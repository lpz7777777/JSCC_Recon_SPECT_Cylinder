"""Exact density tying and descent-checked penalized EM M-steps.

The prior acts once after global backprojection. MAP steps minimize the weighted
Poisson EM surrogate with a diagonal-preconditioned primal-dual method. Keeping
the best feasible primal iterate makes each accepted generalized EM step decrease
its surrogate; capped inner iterations do not claim an exact MAP solution.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


class SpatialGraph:
    def __init__(self, i, j, weight, gradient, n, device="cpu"):
        self.i = torch.as_tensor(np.asarray(i).copy(), dtype=torch.long, device=device)
        self.j = torch.as_tensor(np.asarray(j).copy(), dtype=torch.long, device=device)
        self.weight = torch.as_tensor(np.asarray(weight).copy(), dtype=torch.float64, device=device)
        self.gradient = torch.as_tensor(np.asarray(gradient).copy(), dtype=torch.float64, device=device)
        self.n = n

    def difference(self, x):
        return self.gradient*(x[self.i]-x[self.j])

    def transpose(self, p):
        out = torch.zeros(self.n, dtype=p.dtype, device=p.device)
        out.index_add_(0, self.i, self.gradient*p)
        out.index_add_(0, self.j, -self.gradient*p)
        return out

    def penalty(self, x, delta):
        d = self.difference(x).abs()
        phi = d if delta == 0 else torch.where(d <= delta, d*d/(2*delta), d-delta/2)
        return torch.dot(self.weight, phi)


def kl_value(x, target, a):
    positive = target > 0
    terms = x-target
    terms = terms.clone()
    terms[positive] += target[positive]*(torch.log(target[positive])-torch.log(x[positive].clamp_min(1e-300)))
    return torch.dot(a, terms)


def kl_prox(value, target, a, step):
    b = value-step*a
    product = step*a*target
    root = torch.sqrt(b*b+4*product)
    # Avoid cancellation when b is negative; includes target=0 and positivity.
    return torch.where(b >= 0, .5*(b+root), 2*product/(root-b).clamp_min(1e-300))


def solve_surrogate(current, target, a, graph, strength, delta, max_steps=80,
                    gap_tolerance=1e-6, dual=None):
    if strength == 0:
        return target.clone(), None, {"inner_steps": 0, "gap": 0., "inner_converged": True, "surrogate_delta": float(kl_value(target,target,a)-kl_value(current,target,a))}
    coefficient = strength*graph.weight
    dual = torch.zeros_like(coefficient) if dual is None else dual.clone().clamp(-1, 1)
    incident = graph.transpose(torch.zeros_like(coefficient))
    incident.index_add_(0, graph.i, coefficient*graph.gradient)
    incident.index_add_(0, graph.j, coefficient*graph.gradient)
    tau = .99/(a+incident).clamp_min(1e-300)
    sigma = .99/(2*coefficient*graph.gradient).clamp_min(1e-300)
    initial = kl_value(current, target, a)+strength*graph.penalty(current, delta)
    best_value = initial
    best = current.clone()
    x, extrapolated = current.clone(), current.clone()
    gap = math.inf
    for step in range(1, max_steps+1):
        dual += sigma*coefficient*graph.difference(extrapolated)
        if delta:
            dual /= 1+sigma*coefficient*delta
        dual.clamp_(-1, 1)
        candidate = kl_prox(x-tau*graph.transpose(coefficient*dual), target, a, tau)
        extrapolated = 2*candidate-x
        x = candidate
        if step % 5 == 0 or step == max_steps:
            value = kl_value(x,target,a)+strength*graph.penalty(x,delta)
            if bool(value < best_value):
                best, best_value = x.clone(), value
            # Feasible dual certificate for weighted KL, scaled into q<a.
            q = -graph.transpose(coefficient*dual)
            positive = q > 0
            shrink = min(1., float((a[positive]/q[positive]).min())*.999999) if bool(positive.any()) else 1.
            feasible = dual*shrink
            q = q*shrink
            conjugate = -torch.sum(a*target*torch.log1p(-q/a))
            dual_penalty = .5*delta*torch.dot(coefficient, feasible*feasible)
            gap = max(0., float(best_value+conjugate+dual_penalty))
            if gap <= gap_tolerance*(1+float(best_value)):
                break
    info = {"inner_steps": step, "gap": gap,
            "surrogate_delta": float(best_value-initial),
            "inner_converged": gap <= gap_tolerance*(1+float(best_value))}
    if not torch.isfinite(best).all() or torch.min(best) < 0 or info["surrogate_delta"] > 1e-10:
        raise FloatingPointError("Penalized M-step failed its finite/nonnegative/descent check")
    return best, dual, info


class AblationUpdate:
    def __init__(self, study, spatial_path, variant, device, output=None):
        self.study, self.variant = study, variant
        self.method = variant["method"]
        self.strength = float(variant.get("strength", 0))
        self.delta = float(variant.get("huber_delta", 0)) if self.method == "huber" else 0.
        self.device = device
        with np.load(spatial_path) as model:
            n = len(model["binding_group"])
            self.graph = SpatialGraph(model["edge_i"],model["edge_j"],model["graph_weight"],model["gradient_scale"],n,device)
            self.group = torch.as_tensor(model["binding_group"].copy(),dtype=torch.long,device=device)
        self.group_count = int(self.group.max())+1
        self.output = output
        self.states = {}
        self.rows = []

    @property
    def root(self):
        return not dist.is_initialized() or dist.get_rank() == 0

    def configure(self, name, sensitivity, observed_count):
        if not math.isfinite(observed_count) or observed_count <= 0:
            raise ValueError("Positive observed count normalization required")
        s = sensitivity.reshape(-1).double()
        if torch.min(s) <= 0 or not torch.isfinite(s).all():
            raise ValueError("Positive finite sensitivity required")
        self.states[name] = {"a": s/s.sum(), "alpha": observed_count/float(s.sum()),
                             "count": observed_count, "dual": None, "previous_objective": None,
                             "max_objective_increase": 0., "max_surrogate_increase": 0.,
                             "max_inner_gap": 0., "inner_cap_count": 0, "updates": 0}

    def record_objective(self, name, image, sensitivity, local_logterm, iteration, final=False):
        # All data rows/events are distributed; the sensitivity and image are global.
        logterm = local_logterm.clone().double()
        if dist.is_initialized():
            dist.all_reduce(logterm)
        if not self.root:
            return
        state = self.states[name]
        likelihood = (float(torch.dot(sensitivity.reshape(-1).double(),image.reshape(-1).double()))+float(logterm))/state["count"]
        prior = 0. if self.method == "binding" else self.strength*float(self.graph.penalty(image.reshape(-1).double()/state["alpha"], self.delta))
        objective = likelihood+prior
        previous = state["previous_objective"]
        if previous is not None:
            increase = objective-previous
            state["max_objective_increase"] = max(state["max_objective_increase"], increase)
            if increase > 2e-5:
                raise FloatingPointError(f"{name}: normalized objective increased by {increase}")
        state["previous_objective"] = objective
        if final or iteration == 0 or iteration % 50 == 0:
            self.rows.append({"channel":name,"iteration":iteration,"likelihood_over_count":likelihood,
                              "prior":prior,"objective":objective})

    def __call__(self, image, weight, sensitivity, iteration, name):
        if self.root:
            state = self.states[name]
            x, g, s = (t.reshape(-1).double() for t in (image,weight,sensitivity))
            if not torch.isfinite(g).all() or torch.min(g) < 0:
                raise FloatingPointError("Invalid globally reduced backprojection")
            if self.method == "binding":
                numerator = torch.zeros(self.group_count,dtype=torch.float64,device=self.device)
                denominator = torch.zeros_like(numerator)
                numerator.index_add_(0,self.group,x*g)
                denominator.index_add_(0,self.group,s)
                result = (numerator/denominator)[self.group]
            else:
                target = x*g/s/state["alpha"]
                result, state["dual"], info = solve_surrogate(x/state["alpha"], target,
                    state["a"], self.graph, self.strength, self.delta,
                    self.study["inner_max"],self.study["inner_gap_tolerance"],state["dual"])
                result *= state["alpha"]
                state["max_surrogate_increase"] = max(state["max_surrogate_increase"],info["surrogate_delta"])
                state["max_inner_gap"] = max(state["max_inner_gap"],info["gap"])
                state["inner_cap_count"] += not info["inner_converged"]
                if (iteration+1) % 50 == 0 or iteration == 0:
                    self.rows.append({"channel":name,"iteration":iteration+1,**info})
            state["updates"] += 1
            image = result.reshape_as(image).to(image.dtype)
            if not torch.isfinite(image).all() or torch.min(image) < 0:
                raise FloatingPointError("Invalid updated image")
        if dist.is_initialized():
            dist.broadcast(image,src=0)
        return image

    def save(self):
        if not self.root or self.output is None:
            return
        states = {name:{key:float(value) if isinstance(value, (int,float)) else value
                        for key,value in state.items() if key not in ("a","dual")}
                  for name,state in self.states.items()}
        record = {"variant":self.variant,"states":states,"history":self.rows,
                  "inner_solution":"best feasible primal iterate; descent checked; gap/cap recorded",
                  "prior_applied_once_after_global_backprojection":True}
        Path(self.output,"optimization.json").write_text(json.dumps(record,indent=2,allow_nan=False)+"\n")
