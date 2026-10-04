"""Measurement and actual ellipse-overlap diagnostics; never used by MLEM."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import sys
import numpy as np
from scipy.interpolate import LinearNDInterpolator
import torch

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parents[1])]
from geometry import grid
from compton_event_response import (ComptonEventSettings,build_detector_position_variance,
    build_compton_cone_weights,prepare_compton_events)
from detector_csv import load_detector_coordinates

def quadrature(cell,z,order,radial_order=4,axial_order=4):
    lo,hi,start,end=cell;a,b=250.,150.
    nodes,w=np.polynomial.legendre.leggauss(order)
    theta=(start+end)/2+(end-start)/2*nodes
    hi2=np.minimum(hi**2,1/((np.cos(theta)/a)**2+(np.sin(theta)/b)**2))
    span=np.maximum(hi2-lo**2,0)
    rn,rw=np.polynomial.legendre.leggauss(radial_order)
    zn,zw=np.polynomial.legendre.leggauss(axial_order)
    r=np.sqrt(lo**2+span[:,None]*(rn[None,:]+1)/2)
    x=r*np.cos(theta[:,None]);y=r*np.sin(theta[:,None])
    shape=(order,radial_order,axial_order)
    points=np.stack((np.broadcast_to(x[:,:,None],shape),
                     np.broadcast_to(y[:,:,None],shape),
                     np.broadcast_to(z+1.5*zn[None,None,:],shape)),axis=-1).reshape(-1,3)
    weights=(w[:,None,None]*(end-start)/2*span[:,None,None]/4*
             rw[None,:,None]*zw[None,None,:]*1.5).reshape(-1)
    return points,weights

def energy_layers(inputs,detector,out):
    results=[]
    for folder in sorted(inputs.glob("point_*")):
        for file in sorted(folder.glob("events_v*.csv")):
            with file.open() as f:
                for index,r in enumerate(csv.DictReader(f)):
                    if index>=256:break
                    c1,c2=int(r["c1"])-1,int(r["c2"])-1
                    if min(c1,c2)<0:continue
                    s=np.array([float(r["source_"+k]) for k in "xyz"]);p1=np.array([float(r["p1_"+k]) for k in "xyz"])
                    p2=np.array([float(r["p2_"+k]) for k in "xyz"])
                    # Geant4 world origin and the Factors FOV origin differ by
                    # 345 mm. Compare both true and discretized positions in
                    # the same detector-relative frame.
                    shift=np.array([0.,345.,0.]);s=s+shift;p1=p1+shift;p2=p2+shift
                    def predicted(first,second):
                        v1=first-s;v2=second-first
                        denominator=np.linalg.norm(v1)*np.linalg.norm(v2)
                        if denominator==0:return float("nan")
                        cosine=np.clip(np.dot(v1,v2)/denominator,-1,1)
                        return .440-.440/(1+(.440/.511)*(1-cosine))
                    true=float(r["transfer_mev"]);deposit=float(r["true_e1"]);measured=float(r["measured_e1"])
                    ideal_geometry=predicted(p1,p2);center_geometry=predicted(detector[c1],detector[c2])
                    results.append(dict(dataset=folder.name,view=r["view"],seed=r["seed"],event_id=r["event_id"],
                        legacy=int(r["legacy"]),ideal=int(r["ideal"]),reason=r["reason"],
                        true_geometry_transfer=ideal_geometry,center_geometry_transfer=center_geometry,
                        primary_transfer=true,accumulated_transfer=deposit,measured_transfer=measured,
                        doppler_and_atomic_residual=true-ideal_geometry,
                        crystal_center_residual=center_geometry-ideal_geometry,
                        accumulated_energy_residual=deposit-true,measurement_residual=measured-deposit))
    with (out/"measurement_layers.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]) if results else ["dataset"])
        writer.writeheader();writer.writerows(results)
    return results

def boundary(args,detector):
    cfg=json.loads((HERE/"config.json").read_text());coords,cells,_=grid(cfg)
    geo=np.load(args.geometry);n=3301
    frac=geo["ellipse_fraction"][:n]
    chosen=np.flatnonzero((frac>0)&(frac<.1))
    if len(chosen)>20:chosen=chosen[np.linspace(0,len(chosen)-1,20,dtype=int)]
    volumes=geo["cell_volume_mm3"]
    factor=args.factors/"440keV_RotateNum20"
    raw=np.memmap(factor/"SysMat_polar",dtype="<f4",mode="r",shape=(132040,10496))
    variance=build_detector_position_variance(torch.tensor(detector),0)
    settings=ComptonEventSettings(.440,.13*np.sqrt(511/440),2*.440**2/(.511+2*.440)-.001,.05,.35)
    events=[]
    files=sorted(args.inputs.glob("point_*/legacy_v*.csv"))
    if len(files)>24:files=[files[i] for i in np.linspace(0,len(files)-1,24,dtype=int)]
    for file in files:
        if not file.stat().st_size:continue
        rows=np.loadtxt(file,delimiter=",",usecols=(0,1,2,3),dtype=np.float32,ndmin=2)
        prepared,_=prepare_compton_events(torch.tensor(rows),settings,torch.tensor(detector),variance,variance,
                                         input_energies_already_smeared=True)
        if prepared is None:continue
        index=int(prepared.source_row_indices[0])
        events.append((rows[index],int(file.stem.rsplit("v",1)[1]),file.relative_to(args.inputs).as_posix(),index))
    results=[]
    for event,view,input_file,input_row in events:
        e,_=prepare_compton_events(torch.tensor(event[None,:]),settings,torch.tensor(detector),variance,variance,
                                  input_energies_already_smeared=True)
        c1=int(event[0])-1
        field=(np.array(raw[:,c1],dtype=np.float64)/volumes).reshape(40,n)
        interpolation=LinearNDInterpolator(coords[:n,:2],field.T)
        def spatial_a(points):
            xyvalues=interpolation(points[:,:2])
            layer=(points[:,2]+58.5)/3
            lower=np.clip(np.floor(layer).astype(int),0,38);t=layer-lower
            values=xyvalues[np.arange(len(points)),lower]*(1-t)+xyvalues[np.arange(len(points)),lower+1]*t
            if not np.isfinite(values).all():raise ValueError("Boundary quadrature outside available transverse interpolation")
            return np.maximum(values,0)
        angle=(view-1)*math.pi/10
        def rotate(p):
            result=p.copy();result[:,0]=p[:,0]*math.cos(angle)+p[:,1]*math.sin(angle)
            result[:,1]=p[:,1]*math.cos(angle)-p[:,0]*math.sin(angle);return result
        def response(points):
            world=rotate(points)
            k=build_compton_cone_weights(e,torch.tensor(world,dtype=torch.float32),settings)[0].numpy()
            return k*spatial_a(world)
        for layer in (0,20,39):
            z=-58.5+layer*3
            for cell in chosen:
                previous=None;converged=False;values={};centroid_value=0;true_volume=0
                for order in (16,32,64,128):
                    spatial_order={16:4,32:6,64:8,128:12}[order]
                    points,w=quadrature(cells[cell],z,order,spatial_order,spatial_order);volume=float(w.sum())
                    if volume<=0:raise ValueError("Active boundary cell has zero quadrature volume")
                    value=float(np.dot(response(points),w))
                    values[str(order)]=value
                    if previous is not None and abs(value-previous)<=.01*max(abs(value),1e-30):converged=True
                    else:converged=False
                    previous=value;true_volume=volume
                    centroid=np.sum(points*w[:,None],axis=0)/volume
                    centroid_value=float(response(centroid[None,:])[0]*volume)
                    if converged and order>=64:break
                center_value=float(response(coords[layer*n+cell][None,:])[0]*true_volume)
                results.append(dict(c1=c1+1,c2=int(event[2]),view=view,layer=layer,cell=int(cell),
                    input_file=input_file,input_row=input_row,
                    fraction=float(frac[cell]),overlap_volume=true_volume,representative=center_value,
                    centroid=centroid_value,quadrature=values,converged=converged,
                    axial_a_interpolation="linear with explicit endpoint extrapolation <=1.5 mm",
                    diagnostic_only=True))
    (args.output/"boundary_integration.json").write_text(json.dumps(results,indent=2)+"\n")
    return results

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs",type=Path,required=True);p.add_argument("--factors",type=Path,required=True)
    p.add_argument("--geometry",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    detector=load_detector_coordinates(a.factors/"440keV_RotateNum20/Detector.csv",10496)
    torch.set_num_threads(8)
    layers=energy_layers(a.inputs,detector,a.output)
    with torch.no_grad():integrals=boundary(a,detector)
    (a.output/"offline_summary.json").write_text(json.dumps(dict(layer_events=len(layers),
        boundary_cases=len(integrals),converged=sum(r["converged"] for r in integrals),
        production_kernel_changed=False,production_boundary_operator_changed=False),indent=2)+"\n")
if __name__=="__main__":main()
