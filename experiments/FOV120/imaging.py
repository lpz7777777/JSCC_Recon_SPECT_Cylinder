"""Volume-aware truth, closed-loop projections, and full-height image evaluation."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from fov_config import factor_geometry

OUTPUTS=("440_SinglePhoton","440_ComptonOnly","440_SinglePlusCompton",
         "218_SinglePhoton_CrossTalkCorrected","440SinglePlus218Single","440SingleComptonPlus218Single")


def cell_bounds(coordinates):
    radius=np.hypot(coordinates[:,0],coordinates[:,1])
    rings,index,counts=np.unique(np.round(radius,5),return_inverse=True,return_counts=True)
    nz=len(np.unique(coordinates[:,2])); counts=counts//nz
    edge=np.r_[0,(rings[:-1]+rings[1:])/2,rings[-1]+(rings[-1]-rings[-2])/2]
    return edge[index],edge[index+1],2*np.pi/counts[index],np.arctan2(coordinates[:,1],coordinates[:,0])


def make_truth(factor, dataset, generated, photons, quadrature=8):
    coords,_,_=factor_geometry(factor)
    volume=np.fromfile(factor/"polar_cell_volume_mm3.float64",dtype="<f8")
    lo,hi,dtheta,theta=cell_bounds(coords)
    # Exact cylinder/cell radial overlap; the computational collar remains empty.
    upper=np.maximum(lo,np.minimum(hi,150))
    fraction=(upper**2-lo**2)/(hi**2-lo**2)
    values={218:np.zeros(len(coords)),440:np.zeros(len(coords))}
    masks={}
    if dataset=="XCAT":
        with np.load(generated/"XCAT/truth_3mm.npz") as source:
            arrays={218:source["fr_zyx"],440:source["bi_zyx"]}
            organ={key.removesuffix("_fraction_zyx"):source[key] for key in source.files if key.endswith("_fraction_zyx")}
        source_manifest=json.loads((generated/"XCAT/manifest.json").read_text())
        # MC merges whole interior cells at 3 mm, but keeps native 1.5-mm
        # cubes at the cylinder boundary. Match that hybrid source exactly.
        from Geant4Sim.generate_xcat_ac225_psma_abdomen import native_phantom, whole_target_cell_mask
        with np.load(generated/"XCAT/masks_1p5mm.npz") as source:
            native_low,native_high,native_masks,_=native_phantom(source['xcat_codes_zyx'],
                *source_manifest['crop_half_open']['z'])
        native_arrays={218:native_low,440:native_high}
        whole=whole_target_cell_mask(arrays[218].shape[0])
        integrals={218:source_manifest["integrated_activity_mm3"]["fr"],440:source_manifest["integrated_activity_mm3"]["bi"]}
        masks={name:np.zeros(len(coords)) for name in organ}
    else:
        rods=json.loads((generated/"Simulation/Contrast_truth.json").read_text())["rods"] if dataset=="Contrast" else []
        integrals={e:math.pi*150**2*120+sum(5*math.pi*r["radius_mm"]**2*r["height_mm"] for r in rods if r["energy"]==e) for e in values}
        masks={f"rod_{i+1}":np.zeros(len(coords)) for i in range(len(rods))}
    # Tensor midpoint quadrature uniform in r^2, theta, z. At q=8 -> 512 samples/cell.
    q=quadrature
    dz=float(np.diff(np.unique(coords[:,2]))[0])
    for a in (np.arange(q)+.5)/q:
        r=np.sqrt(lo**2+a*(upper**2-lo**2))
        for b in (np.arange(q)+.5)/q:
            angle=theta+(b-.5)*dtheta
            x,y=r*np.cos(angle),r*np.sin(angle)
            for c in (np.arange(q)+.5)/q:
                z=coords[:,2]+(c-.5)*dz
                if dataset=="XCAT":
                    i=np.floor((x+150)/3).astype(int);j=np.floor((y+150)/3).astype(int)
                    k=np.floor((z+60)/3).astype(int)
                    valid=(i>=0)&(i<100)&(j>=0)&(j<100)&(k>=0)&(k<40)
                    edge=np.zeros(len(coords),dtype=bool)
                    edge[valid]=~whole[k[valid],j[valid],i[valid]]
                    ni=np.floor((x[edge]+150)/1.5).astype(int)
                    nj=np.floor((y[edge]+150)/1.5).astype(int)
                    nk=np.floor((z[edge]+60)/1.5).astype(int)
                    for e in values:
                        values[e][valid]+=arrays[e][k[valid],j[valid],i[valid]]/q**3
                        values[e][edge]+=(native_arrays[e][nk,nj,ni]-arrays[e][k[edge],j[edge],i[edge]])/q**3
                    for name in masks:
                        masks[name][valid]+=organ[name][k[valid],j[valid],i[valid]]/q**3
                        masks[name][edge]+=(native_masks[name][nk,nj,ni]-organ[name][k[edge],j[edge],i[edge]])/q**3
                else:
                    for e in values: values[e]+=1/q**3
                    for index,rod in enumerate(rods):
                        cx,cy,cz=rod["center_mm"]
                        inside=((x-cx)**2+(y-cy)**2<=rod["radius_mm"]**2)&(abs(z-cz)<=rod["height_mm"]/2)
                        values[rod["energy"]]+=5*inside/q**3
                        masks[f"rod_{index+1}"]+=inside/q**3
    for e in values: values[e]*=fraction
    for name in masks: masks[name]*=fraction
    yields={218:.114,440:.259}
    scale=photons/sum(integrals[e]*yields[e] for e in values)
    density={e:values[e]*yields[e]*scale for e in values}
    errors={str(e):float(np.dot(values[e],volume)/integrals[e]-1) for e in values}
    return density, masks, {"dataset":dataset,"photons":photons,"quadrature_per_axis":q,
                            "relative_integral_error":errors,"common_activity_scale":scale,
                            "physical_radius_mm":150,"support_radius_mm":153,
                            "units":"emitted photons/mm3 over all views"}


def project(factor, rho):
    n=len(rho)
    coordinates,_,_=factor_geometry(factor)
    if len(coordinates)!=n:
        raise ValueError("Truth and Factors use different voxel counts")
    detector_count=(factor/"SysMat_polar").stat().st_size//(4*n)
    matrix=np.memmap(factor/"SysMat_polar",dtype="<f4",mode="r",shape=(n,detector_count))
    rot=np.loadtxt(factor/"RotMat_full.csv",delimiter=",",dtype=np.int64)-1
    rotated=rho[rot]/rot.shape[1]
    output=np.zeros((detector_count,rot.shape[1]))
    for start in range(0,n,1024):
        output+=matrix[start:start+1024].T@rotated[start:start+1024]
    return output.T


def genproj(factors,truth_path,out,seed):
    metadata=json.loads(truth_path.with_suffix('.json').read_text())
    photons=int(metadata['photons'])
    exponent=int(math.log10(photons))
    if photons!=10**exponent:
        raise ValueError("Closed-loop filename count levels require a power-of-ten photon budget")
    count_level=f'1e{exponent}'
    with np.load(truth_path) as source:
        low,high=source["rho218"],source["rho440"]
    y440=project(factors/"440keV_RotateNum20",high)
    cross=project(factors/"440keV_to218win_RotateNum20",high)
    y218=project(factors/"218keV_RotateNum20",low)+cross
    rng=np.random.default_rng(seed)
    for kind in ("Noiseless","Poisson"):
        for energy,counts in ((218,y218),(440,y440)):
            directory=out/kind/f"{energy}keV_RotateNum20"
            directory.mkdir(parents=True,exist_ok=True)
            path=directory/f"CntStat_Closure_{count_level}.csv"
            if path.exists(): raise FileExistsError(path)
            np.savetxt(path,counts if kind=="Noiseless" else rng.poisson(counts),delimiter=",")
    np.save(out/"cross_projection.npy",cross)
    (out/"manifest.json").write_text(json.dumps({"seed":seed,"truth":str(truth_path),
        "normalization":"rho is total emitted photon density; each projection divides by 20",
        "count_level":count_level,"photons":photons},indent=2))


def weighted_metrics(image,truth,volume,selection):
    weight=volume*selection
    total=weight.sum()
    if total<=0:return {"volume_mm3":0}
    mean=np.dot(weight,image)/total; ref=np.dot(weight,truth)/total
    variance=np.dot(weight,(image-mean)**2)/total
    return {"volume_mm3":float(total),"mean":float(mean),"truth_mean":float(ref),
            "recovery":float(mean/ref) if ref>0 else None,
            "bias":float(mean/ref-1) if ref>0 else None,
            "cv":float(np.sqrt(variance)/mean) if mean>0 else None,
            "integral":float(np.dot(weight,image)),"truth_integral":float(np.dot(weight,truth)),
            "nrmse":float(np.sqrt(np.dot(weight,(image-truth)**2)/max(np.dot(weight,truth**2),1e-30)))}


def contrast_metrics(image,truth,volume,roi,background,coordinates):
    region=weighted_metrics(image,truth,volume,roi)
    bg=weighted_metrics(image,truth,volume,background)
    if not region["volume_mm3"] or not bg["volume_mm3"]:
        return {"available":False}
    sigma=bg["cv"]*bg["mean"] if bg["cv"] is not None else 0
    contrast_truth=region["truth_mean"]/bg["truth_mean"]-1 if bg["truth_mean"]>0 else 0
    region["cnr"]=(region["mean"]-bg["mean"])/sigma if sigma>0 else None
    region["crc"]=(region["mean"]/bg["mean"]-1)/contrast_truth if bg["mean"]>0 and abs(contrast_truth)>1e-10 else None
    weights=volume*roi*np.maximum(image-bg["mean"],0)
    reference=volume*roi*np.maximum(truth-bg["truth_mean"],0)
    if weights.sum()>0 and reference.sum()>0:
        center=(coordinates*weights[:,None]).sum(axis=0)/weights.sum()
        expected=(coordinates*reference[:,None]).sum(axis=0)/reference.sum()
        region["excess_centroid_error_mm"]=float(np.linalg.norm(center-expected))
    else:region["excess_centroid_error_mm"]=None
    return region


def evaluate(result,factor,truth_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.interpolate import LinearNDInterpolator
    from scipy.spatial import Delaunay
    coords,nxy,nz=factor_geometry(factor)
    volume=np.fromfile(factor/"polar_cell_volume_mm3.float64",dtype="<f8")
    with np.load(truth_path) as data:
        rho218,rho440=data["rho218"],data["rho440"]
        masks={key[5:]:data[key] for key in data.files if key.startswith("mask_")}
    manifest=json.loads((result/"run_manifest.json").read_text())
    iterations,step=manifest["iterations"],manifest["save_step"]
    frames=iterations//step
    out=result/"FullFOV";out.mkdir(exist_ok=True)
    # Reuse one in-plane triangulation for all 40 layers; no axial crop/filter/scaling.
    triangulation=Delaunay(coords[:nxy,:2])
    axis=np.arange(-148.5,150,3);xx,yy=np.meshgrid(axis,axis)
    def cart(values):
        shaped=values.reshape(nz,nxy)
        return np.stack([LinearNDInterpolator(triangulation,plane,fill_value=0)(xx,yy) for plane in shaped])
    summaries={};histories={}
    zabs=np.abs(coords[:,2])
    regions={"center":zabs<=30,"middle":(zabs>30)&(zabs<=45),"edge":(zabs>45)&(zabs<=60)}
    if "body" in masks:
        background=masks["body"]>.999
        for name,mask in masks.items():
            if name!="body":background &= mask<1e-8
    else:
        background=np.hypot(coords[:,0],coords[:,1])<140
        for mask in masks.values():background &= mask<1e-8
    for key in OUTPUTS[:4]:
        path=result/f"Image_{key}_Iter_{iterations}_{frames}"
        history=np.fromfile(path,dtype="<f4")
        if history.size!=frames*len(coords) or not np.isfinite(history).all():
            raise ValueError(f"Invalid history: {path}")
        histories[key]=history.reshape(frames,-1)
    histories[OUTPUTS[4]]=histories[OUTPUTS[0]]+histories[OUTPUTS[3]]
    histories[OUTPUTS[5]]=histories[OUTPUTS[2]]+histories[OUTPUTS[3]]
    for index,key in enumerate(OUTPUTS):
        values=np.fromfile(result/f"Image_{key}",dtype="<f4")
        if values.shape!=rho440.shape or not np.isfinite(values).all() or np.any(values<0):
            raise ValueError(f"Invalid image: {key}")
        truth=rho440 if index<3 else rho218 if index==3 else rho440+rho218
        image3,truth3=cart(values),cart(truth)
        vmax=max(float(truth.max()),1e-12)
        fig,axes=plt.subplots(2,4,figsize=(15,7),layout="constrained")
        for row,(cube,label) in enumerate(((truth3,"Truth"),(image3,key))):
            panels=(cube[nz//2],cube[:,50,:],cube[:,:,50],cube.max(axis=1))
            for col,(panel,title) in enumerate(zip(panels,("axial","coronal","sagittal","coronal MIP"))):
                extent=(-150,150,-150,150) if col==0 else (-150,150,-60,60)
                axes[row,col].imshow(panel,origin="lower",extent=extent,cmap="gray_r",vmin=0,vmax=vmax,aspect="equal")
                axes[row,col].set_title(f"{label}: {title}")
        fig.savefig(out/f"{key}.png",dpi=150);plt.close(fig)
        summary={name:weighted_metrics(values,truth,volume,selection) for name,selection in regions.items()}
        summary["background"]={name:weighted_metrics(values,truth,volume,selection&background) for name,selection in regions.items()}
        summary["organs"]={name:weighted_metrics(values,truth,volume,mask) for name,mask in masks.items()}
        summary["contrast"]={name:contrast_metrics(values,truth,volume,mask,background,coords) for name,mask in masks.items() if name!="body"}
        history_metrics=[]
        for frame,snapshot in enumerate(histories[key]):
            history_metrics.append({"iteration":(frame+1)*step,
                "background":{name:weighted_metrics(snapshot,truth,volume,selection&background) for name,selection in regions.items()},
                "contrast":{name:contrast_metrics(snapshot,truth,volume,mask,background,coords) for name,mask in masks.items() if name!="body"},
                **{name:weighted_metrics(snapshot,truth,volume,selection) for name,selection in regions.items()}})
        summary["iterations"]=history_metrics
        fig,axes=plt.subplots(1,6,figsize=(18,4),layout="constrained")
        chosen=np.unique(np.rint(np.linspace(0,frames-1,6)).astype(int))
        for ax,frame in zip(axes,chosen):
            mip=cart(histories[key][frame]).max(axis=1)
            ax.imshow(mip,origin="lower",extent=(-150,150,-60,60),cmap="gray_r",vmin=0,vmax=vmax)
            ax.set_title(f"Iteration {(frame+1)*step}")
        fig.savefig(out/f"{key}_iterations.png",dpi=150);plt.close(fig)
        summaries[key]=summary
    (out/"metrics.json").write_text(json.dumps({"crop_mm":0,"filter_sigma":0,"colormap":"gray_r",
        "truth_source":str(truth_path),"display_scaling":"none; common truth range",
        "results":summaries},indent=2)+"\n")


def main():
    generated=Path(__file__).resolve().parent/"generated"
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("truth");p.add_argument("--dataset",choices=("Uniform","Contrast","XCAT"),required=True)
    p.add_argument("--factor",type=Path,default=generated/"Factors/440keV_RotateNum20")
    p.add_argument("--photons",type=int,default=1_000_000_000);p.add_argument("--quadrature",type=int,default=8)
    p.add_argument("--output",type=Path,required=True)
    p=sub.add_parser("genproj");p.add_argument("--factors",type=Path,default=generated/"Factors")
    p.add_argument("--truth",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--seed",type=int,default=260924)
    p=sub.add_parser("evaluate");p.add_argument("--result",type=Path,required=True)
    p.add_argument("--factor",type=Path,default=generated/"Factors/440keV_RotateNum20")
    p.add_argument("--truth",type=Path,required=True)
    args=parser.parse_args()
    if args.command=="truth":
        if args.quadrature<1 or args.photons<=0:raise ValueError("Positive quadrature/count required")
        if args.output.exists():raise FileExistsError(args.output)
        density,masks,metadata=make_truth(args.factor,args.dataset,generated,args.photons,args.quadrature)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(args.output,rho218=density[218],rho440=density[440],**{"mask_"+k:v for k,v in masks.items()})
        args.output.with_suffix(".json").write_text(json.dumps(metadata,indent=2)+"\n")
    elif args.command=="genproj":genproj(args.factors,args.truth,args.output,args.seed)
    else:evaluate(args.result,args.factor,args.truth)


if __name__=="__main__":main()
