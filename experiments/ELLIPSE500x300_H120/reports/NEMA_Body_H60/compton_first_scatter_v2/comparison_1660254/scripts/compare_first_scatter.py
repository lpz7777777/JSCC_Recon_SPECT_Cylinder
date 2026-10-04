"""Unsmoothed A/B comparison using the frozen H60 dual-energy 3D sphere truth."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from analyze_nema_result import digest,xy_interpolator,weighted_mean
from mip_projection import axial_mip

HERE=Path(__file__).resolve().parent
CHANNELS=("440_ComptonOnly","440_SinglePlusCompton")
ITERATIONS=(100,500,1000,2000)

def table(path,rows):
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def compare(a,b,output):
    output.mkdir(parents=True,exist_ok=False)
    roots={"legacy":a,"ideal":b};verified={}
    for group,root in roots.items():
        v=json.loads((root/"verification.json").read_text())
        if not v["passed"] or v["mode"]!="formal" or v["iterations"]!=2000:
            raise ValueError("Two verified 2000-iteration results are required")
        if digest(root/"run_manifest.json")!=v["run_manifest_sha256"]:raise ValueError("Verified manifest differs")
        verified[group]=v
    manifests={g:json.loads((p/"run_manifest.json").read_text()) for g,p in roots.items()}
    if manifests["legacy"]["geometry_sha256"]!=manifests["ideal"]["geometry_sha256"]:
        raise ValueError("Paired geometries differ")
    for key,h in manifests["legacy"]["input_sha256"].items():
        if key.startswith("CntStat/") and manifests["ideal"]["input_sha256"][key]!=h:
            raise ValueError("Paired single-photon counts differ")
    source=json.loads((HERE/"reports/NEMA_Body_H60/manifest.json").read_text())
    truthpath=HERE/"generated/NEMA_Body_H60/truth_3mm.npz"
    if digest(truthpath)!=source["truth_sha256"]:raise ValueError("Frozen source truth differs")
    t=np.load(truthpath);x,y,z=(t[f"{k}_mm"] for k in "xyz")
    truth=t["activity_440_zyx"]
    sphere_sum=sum(t[f"sphere_{d}_fraction_zyx"] for d in (10,13,17,22,28,37))
    bgfraction=np.maximum(t["body_fraction_zyx"]-t["lung_fraction_zyx"]-sphere_sum,0)
    background=(bgfraction>=.99)&(np.abs(z[:,None,None])<=25.5)
    g=np.load(HERE/"generated/Geometry/geometry.npz")
    coords=g["coordinates_mm"];active=g["active_indices"]
    volumes=(g["cell_volume_mm3"]*g["ellipse_fraction"])[active]
    tiny=g["ellipse_fraction"][active]<.1;outside=np.abs(coords[active,2])>30
    interpolate=xy_interpolator(coords[:3301,:2],x,y)
    xx,yy=np.meshgrid(x,y);ellipse=(xx/250)**2+(yy/150)**2<=1
    def image(values):
        full=np.zeros(132040,np.float32);full[active]=values
        result=interpolate(full.reshape(40,3301));result[:,~ellipse]=0;return result
    masks={}
    for d in (13,22,37):
        cx,cy,cz=next(s["center_mm"] for s in source["spheres"] if s["diameter_mm"]==d)
        masks[d]=(((xx-cx)**2+(yy-cy)**2<=(d/2+25)**2)[None]&
                   (np.abs(z[:,None,None]-cz)<=d/2+3)&(bgfraction>=.99))
    collection=json.loads((HERE/"generated/compton_first_scatter_v2/analysis_inputs/NEMA/collection.json").read_text())
    photons440=collection["primary_counts"][1]
    rows=[];spheres=[];selected={};normalizers={};history_sha={}
    for channel in CHANNELS:
        paths={gp:root/f"Image_{channel}_history.float32" for gp,root in roots.items()}
        histories={}
        for gp,p in paths.items():
            if p.stat().st_size!=40*82040*4:raise ValueError("Exactly 40 saved frames required")
            sha=digest(p)
            expected=next(r["sha256"]["history"] for r in verified[gp]["outputs"] if r["channel"]==channel)
            if sha!=expected:raise ValueError("Verified history differs")
            history_sha[gp+"/"+channel]=sha
            histories[gp]=np.memmap(p,"<f4",mode="r",shape=(40,82040))
        normalizers[channel]=float(image(histories["legacy"][-1])[background].mean())
        if normalizers[channel]<=0:raise ValueError("Reference background scale is nonpositive")
        for gp,h in histories.items():
            for index,values in enumerate(h):
                if not np.isfinite(values).all() or np.any(values<0):raise ValueError("Invalid history values")
                iteration=(index+1)*50;im=image(values);bg=im[background]
                mean=float(bg.mean());std=float(bg.std(ddof=1));integral=float(values.astype(float)@volumes)
                peak=int(values.argmax());order=np.argsort(values)
                rows.append(dict(group=gp,channel=channel,iteration=iteration,max_density=float(values[peak]),
                    peak_background_ratio=float(values[peak]/mean),peak_x_mm=float(coords[active[peak],0]),
                    peak_y_mm=float(coords[active[peak],1]),peak_z_mm=float(coords[active[peak],2]),
                    p99_density=float(np.quantile(values,.99)),p999_density=float(np.quantile(values,.999)),
                    volume_weighted_p999_density=float(np.interp(.999,np.cumsum(volumes[order])/volumes.sum(),values[order])),
                    background_mean=mean,background_cv=std/mean,total_integral=integral,integral_recovery=integral/photons440,
                    tiny_mass_fraction=float(values[tiny].astype(float)@volumes[tiny]/integral),
                    source_z_leakage=float(values[outside].astype(float)@volumes[outside]/integral)))
                for d in (13,22,37):
                    local=im[masks[d]];local_mean=float(local.mean());local_std=float(local.std(ddof=1))
                    hot=weighted_mean(im,t[f"sphere_{d}_fraction_zyx"])
                    spheres.append(dict(group=gp,channel=channel,iteration=iteration,diameter_mm=d,
                        hot_mean=hot,local_background_mean=local_mean,crc=(hot/local_mean-1)/9,cnr=(hot-local_mean)/local_std))
                if iteration in ITERATIONS:selected[gp,channel,iteration]=im/normalizers[channel]
    table(output/"native_iteration_metrics.csv",rows);table(output/"sphere_iteration_metrics.csv",spheres)
    extent=(x[0]-1.5,x[-1]+1.5,y[0]-1.5,y[-1]+1.5)
    for kind in ("center","mip72"):
        fig,axes=plt.subplots(4,5,figsize=(18,11),layout="constrained")
        for row,(channel,gp) in enumerate((c,p) for c in CHANNELS for p in ("legacy","ideal")):
            for col,im in enumerate([truth]+[selected[gp,channel,i] for i in ITERATIONS]):
                panel=im[20] if kind=="center" else axial_mip(im,z,8)
                color=axes[row,col].imshow(panel,origin="lower",extent=extent,cmap="gray_r",vmin=0,vmax=10,interpolation="nearest")
                axes[row,col].set(xlim=(-252,252),ylim=(-150,150),aspect="equal",title="truth" if col==0 else f"iter {ITERATIONS[col-1]}")
                axes[row,col].tick_params(labelsize=7)
                if col==0:axes[row,col].set_ylabel(channel+"\n"+gp,fontsize=9)
        fig.colorbar(color,ax=axes,shrink=.7,label="Density / common A-group background mean at iteration 2000")
        fig.suptitle("NEMA 1e9 paired first-scatter study: "+("z=+1.5 mm" if kind=="center" else "central 72 mm MIP")+"; no smoothing")
        fig.savefig(output/f"iterations_{kind}.png",dpi=130);plt.close(fig)
    ix=int(np.argmin(abs(x)));iy=int(np.argmin(abs(y)))
    fig,axes=plt.subplots(2,12,figsize=(26,7),layout="constrained")
    for row,channel in enumerate(CHANNELS):
        for gi,(label,im) in enumerate([( "truth",truth)]+[(gp,selected[gp,channel,2000]) for gp in ("legacy","ideal")]):
            panels=[(im[20],extent),(im[:,iy,:],(-252,252,-60,60)),(im[:,:,ix],(-150,150,-60,60)),(axial_mip(im,z,8),extent)]
            for k,(panel,ex) in enumerate(panels):
                color=axes[row,gi*4+k].imshow(panel,origin="lower",extent=ex,cmap="gray_r",vmin=0,vmax=10,interpolation="nearest")
                axes[row,gi*4+k].set_title(label+" "+("axial","coronal","sagittal","MIP72")[k],fontsize=8)
                axes[row,gi*4+k].tick_params(labelsize=6)
        axes[row,0].set_ylabel(channel)
    fig.colorbar(color,ax=axes,shrink=.7);fig.savefig(output/"final_multiplanar.png",dpi=130);plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(16,8),layout="constrained")
    for row,c in enumerate(CHANNELS):
        for gp,style in (("legacy","--"),("ideal","-")):
            rr=[r for r in rows if r["group"]==gp and r["channel"]==c]
            for col,key in enumerate(("peak_background_ratio","background_cv","tiny_mass_fraction","source_z_leakage")):
                axes[row,col].plot([r["iteration"] for r in rr],[r[key] for r in rr],style,label=gp)
                axes[row,col].set(title=c+"\n"+key,xlabel="iteration");axes[row,col].grid(alpha=.2)
        axes[row,0].set_yscale("log");axes[row,0].legend()
    fig.savefig(output/"spike_noise_leakage_curves.png",dpi=145);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout="constrained")
    for row,c in enumerate(CHANNELS):
        for gp,style in (("legacy","--"),("ideal","-")):
            for d,color in zip((13,22,37),("tab:blue","tab:orange","tab:green")):
                rr=[r for r in spheres if r["group"]==gp and r["channel"]==c and r["diameter_mm"]==d]
                for col,key in enumerate(("crc","cnr")):
                    axes[row,col].plot([r["iteration"] for r in rr],[r[key] for r in rr],style,color=color,label=f"{gp} {d}mm")
                    axes[row,col].set(title=c+" "+key,xlabel="iteration");axes[row,col].grid(alpha=.2)
        axes[row,0].legend(fontsize=8)
    fig.savefig(output/"crc_cnr_curves.png",dpi=145);plt.close(fig)
    verdict={}
    for c in CHANNELS:
        old=next(r for r in rows if r["group"]=="legacy" and r["channel"]==c and r["iteration"]==2000)
        new=next(r for r in rows if r["group"]=="ideal" and r["channel"]==c and r["iteration"]==2000)
        reductions={k:1-new[k]/old[k] for k in ("max_density","peak_background_ratio")}
        costs=[]
        for d in (13,22,37):
            aa=next(r["crc"] for r in spheres if r["group"]=="legacy" and r["channel"]==c and r["iteration"]==2000 and r["diameter_mm"]==d)
            bb=next(r["crc"] for r in spheres if r["group"]=="ideal" and r["channel"]==c and r["iteration"]==2000 and r["diameter_mm"]==d)
            costs.append(dict(diameter_mm=d,crc_change=bb-aa,cost_over_5_percentage_points=aa-bb>.05))
        verdict[c]=dict(extreme_spike_improved=all(v>=.5 for v in reductions.values()),reductions=reductions,crc_costs=costs,
                        legacy_final=old,ideal_final=new)
    report=dict(study="compton_first_scatter_v2",through_iteration=2000,selected_iterations=ITERATIONS,
        truth_sha256=digest(truthpath),history_sha256=history_sha,common_normalizers=normalizers,verdict=verdict,
        method="full 120 mm metrics; central 72 mm MIP; no smoothing; gray_r; common A2000 background scale",
        limitation="Truth trajectory selection and 2000 iterations do not establish real-device capability or 10000-iteration stability")
    (output/"comparison.json").write_text(json.dumps(report,indent=2)+"\n")
    scripts=output/"scripts";scripts.mkdir()
    for name in ("compare_first_scatter.py","analyze_nema_result.py","mip_projection.py"):shutil.copy2(HERE/name,scripts/name)
    files={p.relative_to(output).as_posix():digest(p) for p in output.rglob("*") if p.is_file()}
    (output/"artifact_manifest.json").write_text(json.dumps(files,indent=2)+"\n")
    print(json.dumps(verdict,indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ("legacy","ideal","output"):p.add_argument("--"+arg,type=Path,required=True)
    a=p.parse_args();compare(a.legacy,a.ideal,a.output)
if __name__=="__main__":main()
