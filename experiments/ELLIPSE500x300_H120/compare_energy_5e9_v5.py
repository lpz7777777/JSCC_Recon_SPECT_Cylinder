"""Unsmoothed A/B comparison using the verified v5 whole-cell paired results and authoritative H60 3D sphere truth."""
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
from energy_5e9_v5_contract import load_contract,verify_checkpoints

HERE=Path(__file__).resolve().parent
CHANNELS=("440_ComptonOnly","440_SinglePlusCompton")
ITERATIONS=(100,500,1000,2000)

def table(path,rows):
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def compare(a,b,output,geometry_path,summary_path):
    summary=json.loads(summary_path.read_text())
    contract=geometry_path.parent/'contract.json'
    cfg=load_contract(contract,'formal',2000,50)
    if (not summary['passed'] or not summary['paired_imaging_completed'] or
        summary['mode']!='formal' or summary['contract_sha256']!=digest(contract)):
        raise ValueError('Actual paired formal verification is required before analysis')
    output.mkdir(parents=True,exist_ok=False)
    roots={"angular":a,"continuous_energy":b};verified={}
    for group,root in roots.items():
        v=json.loads((root/"verification.json").read_text())
        if not v["passed"] or v["mode"]!="formal" or v["iterations"]!=2000:
            raise ValueError("Two verified 2000-iteration results are required")
        if digest(root/"run_manifest.json")!=v["run_manifest_sha256"]:raise ValueError("Verified manifest differs")
        receipt=summary['models'][group]
        for name in ('verification.json','run_manifest.json'):
            if digest(root/name)!=receipt['sha256'][name]:raise ValueError('Fetched actual proof differs')
        if (v['model']!=group or v['accepted_events']!=cfg['accepted_events'] or
            v['contract_sha256']!=digest(contract)):
            raise ValueError('Actual paired model/event/contract differs')
        verified[group]=v
    manifests={g:json.loads((p/"run_manifest.json").read_text()) for g,p in roots.items()}
    if manifests["angular"]["geometry_sha256"]!=manifests["continuous_energy"]["geometry_sha256"]:
        raise ValueError("Paired geometries differ")
    for group,run in manifests.items():
        for key,expected in [('input_sha256',cfg['input_sha256']),
            ('factor_manifest_sha256',cfg['factor_manifest_sha256']),
            ('factor_payload_sha256',cfg['factor_payload_sha256']),('geometry_sha256',cfg['whole_geometry_sha256']),
            ('sensitivity_sha256',cfg['files'][group+'_Sensi_full']),
            ('source_sha256',cfg['files']['run_energy_5e9_v5.py'])]:
            if run[key]!=expected:raise ValueError('Actual frozen formal identity differs: '+key)
        if (run['model']!=group or run['pixels_active']!=78920 or run['pixels_full']!=132040 or
            run['accepted_compton_events']!=cfg['accepted_events'] or run['accepted_compton_events_per_view']!=cfg['events_per_view'] or
            run['actual_primary_gamma']!=5_000_000_000 or run['workers']!=200 or run['views']!=20):
            raise ValueError('Actual formal dimensions/transport/events differ')
    source=json.loads((HERE/"reports/NEMA_Body_H60/manifest.json").read_text())
    truthpath=HERE/"generated/NEMA_Body_H60/truth_3mm.npz"
    if digest(truthpath)!=source["truth_sha256"]:raise ValueError("Frozen source truth differs")
    t=np.load(truthpath);x,y,z=(t[f"{k}_mm"] for k in "xyz")
    truth=t["activity_440_zyx"]
    sphere_sum=sum(t[f"sphere_{d}_fraction_zyx"] for d in (10,13,17,22,28,37))
    bgfraction=np.maximum(t["body_fraction_zyx"]-t["lung_fraction_zyx"]-sphere_sum,0)
    background=(bgfraction>=.99)&(np.abs(z[:,None,None])<=25.5)
    g=np.load(geometry_path)
    coords=g["coordinates_mm"];active=g["active_indices"]
    if len(active)!=78920 or len(coords)!=132040 or not np.isin(g['ellipse_fraction'],[0,1]).all():
        raise ValueError('Only verified whole-cell polar basis is supported')
    if truth.shape!=(40,100,168) or not np.allclose(coords[:,2].reshape(40,3301)[:,0],z):
        raise ValueError('Authoritative truth and polar axial planes differ')
    volumes=(g["cell_volume_mm3"]*g["ellipse_fraction"])[active]
    outside=np.abs(coords[active,2])>30
    interpolate=xy_interpolator(coords[:3301,:2],x,y)
    xx,yy=np.meshgrid(x,y)
    def image(values):
        full=np.zeros(132040,np.float32);full[active]=values
        return interpolate(full.reshape(40,3301))
    masks={}
    for d in (13,22,37):
        cx,cy,cz=next(s["center_mm"] for s in source["spheres"] if s["diameter_mm"]==d)
        masks[d]=(((xx-cx)**2+(yy-cy)**2<=(d/2+25)**2)[None]&
                   (np.abs(z[:,None,None]-cz)<=d/2+3)&(bgfraction>=.99))
    collection=json.loads((geometry_path.parent/'transport_collection.json').read_text())
    photons440=collection["primary_counts"][1]
    rows=[];spheres=[];selected={};normalizers={};history_sha={}
    for gp,root in roots.items():
        snapshot_history={c:np.memmap(root/f'Image_{c}_history.float32','<f4',mode='r',shape=(40,78920)) for c in CHANNELS}
        checked=verify_checkpoints(root,snapshot_history,active,132040,gp,digest(contract))
        if checked!=verified[gp]['checkpoints']:raise ValueError('Fetched checkpoints differ from actual proof')
    for channel in CHANNELS:
        paths={gp:root/f"Image_{channel}_history.float32" for gp,root in roots.items()}
        histories={}
        for gp,p in paths.items():
            root=roots[gp]
            if p.stat().st_size!=40*78920*4:raise ValueError("Exactly 40 saved frames required")
            sha=digest(p)
            expected=next(r["sha256"]["history"] for r in verified[gp]["outputs"] if r["channel"]==channel)
            if sha!=expected:raise ValueError("Verified history differs")
            history_sha[gp+"/"+channel]=sha
            histories[gp]=np.memmap(p,"<f4",mode="r",shape=(40,78920))
            if (digest(root/f'Image_{channel}_active.float32')!=
                next(item['sha256']['active'] for item in verified[gp]['outputs'] if item['channel']==channel)):
                raise ValueError('Actual final image differs')
            final=np.fromfile(root/f'Image_{channel}_active.float32','<f4')
            if not np.array_equal(final,histories[gp][-1]):raise ValueError('Final frame differs')
        normalizers[channel]=float(image(histories["angular"][-1])[background].mean())
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
                    tiny_mass_fraction="not_applicable_whole_cells",
                    source_z_leakage=float(values[outside].astype(float)@volumes[outside]/integral)))
                for d in (13,22,37):
                    local=im[masks[d]];local_mean=float(local.mean());local_std=float(local.std(ddof=1))
                    hot=weighted_mean(im,t[f"sphere_{d}_fraction_zyx"])
                    spheres.append(dict(group=gp,channel=channel,iteration=iteration,diameter_mm=d,
                        hot_mean=hot,local_background_mean=local_mean,crc=(hot/local_mean-1)/9,cnr=(hot-local_mean)/local_std))
                if iteration in ITERATIONS:selected[gp,channel,iteration]=im/normalizers[channel]
    table(output/"native_iteration_metrics.csv",rows);table(output/"sphere_iteration_metrics.csv",spheres)
    extent=(x[0]-1.5,x[-1]+1.5,y[0]-1.5,y[-1]+1.5)
    ix=int(np.argmin(abs(x)));iy=int(np.argmin(abs(y)))
    views={
        'center':(lambda im:im[20],extent,'z=+1.5 mm'),
        'coronal':(lambda im:im[:,iy,:],(x[0]-1.5,x[-1]+1.5,-60,60),f'y={y[iy]:+.1f} mm'),
        'sagittal':(lambda im:im[:,:,ix],(y[0]-1.5,y[-1]+1.5,-60,60),f'x={x[ix]:+.1f} mm'),
        'mip72':(lambda im:axial_mip(im,z,8),extent,'central 72 mm MIP')}
    for kind,(select,ex,label) in views.items():
        fig,axes=plt.subplots(4,5,figsize=(18,11),layout="constrained")
        for row,(channel,gp) in enumerate((c,p) for c in CHANNELS for p in ("angular","continuous_energy")):
            for col,im in enumerate([truth]+[selected[gp,channel,i] for i in ITERATIONS]):
                panel=select(im)
                color=axes[row,col].imshow(panel,origin="lower",extent=ex,cmap="gray_r",vmin=0,vmax=10,interpolation="nearest")
                axes[row,col].set(aspect="equal",title="truth" if col==0 else f"iter {ITERATIONS[col-1]}")
                axes[row,col].tick_params(labelsize=7)
                if col==0:axes[row,col].set_ylabel(channel+"\n"+gp,fontsize=9)
        fig.colorbar(color,ax=axes,shrink=.7,label="Density / common A-group background mean at iteration 2000")
        fig.suptitle("NEMA 5e9 legacy paired v5 energy-response study: "+label+"; no smoothing")
        fig.savefig(output/f"iterations_{kind}.png",dpi=130);plt.close(fig)
    ix=int(np.argmin(abs(x)));iy=int(np.argmin(abs(y)))
    fig,axes=plt.subplots(2,12,figsize=(26,7),layout="constrained")
    for row,channel in enumerate(CHANNELS):
        for gi,(label,im) in enumerate([( "truth",truth)]+[(gp,selected[gp,channel,2000]) for gp in ("angular","continuous_energy")]):
            panels=[(im[20],extent),(im[:,iy,:],(-252,252,-60,60)),(im[:,:,ix],(-150,150,-60,60)),(axial_mip(im,z,8),extent)]
            for k,(panel,ex) in enumerate(panels):
                color=axes[row,gi*4+k].imshow(panel,origin="lower",extent=ex,cmap="gray_r",vmin=0,vmax=10,interpolation="nearest")
                axes[row,gi*4+k].set_title(label+" "+("axial","coronal","sagittal","MIP72")[k],fontsize=8)
                axes[row,gi*4+k].tick_params(labelsize=6)
        axes[row,0].set_ylabel(channel)
    fig.colorbar(color,ax=axes,shrink=.7);fig.savefig(output/"final_multiplanar.png",dpi=130);plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(16,8),layout="constrained")
    for row,c in enumerate(CHANNELS):
        for gp,style in (("angular","--"),("continuous_energy","-")):
            rr=[r for r in rows if r["group"]==gp and r["channel"]==c]
            for col,key in enumerate(("max_density","peak_background_ratio","background_cv","source_z_leakage")):
                axes[row,col].plot([r["iteration"] for r in rr],[r[key] for r in rr],style,label=gp)
                axes[row,col].set(title=c+"\n"+key,xlabel="iteration");axes[row,col].grid(alpha=.2)
        axes[row,0].set_yscale("log");axes[row,0].legend()
    fig.savefig(output/"spike_noise_leakage_curves.png",dpi=145);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout="constrained")
    for row,c in enumerate(CHANNELS):
        for gp,style in (("angular","--"),("continuous_energy","-")):
            for d,color in zip((13,22,37),("tab:blue","tab:orange","tab:green")):
                rr=[r for r in spheres if r["group"]==gp and r["channel"]==c and r["diameter_mm"]==d]
                for col,key in enumerate(("crc","cnr")):
                    axes[row,col].plot([r["iteration"] for r in rr],[r[key] for r in rr],style,color=color,label=f"{gp} {d}mm")
                    axes[row,col].set(title=c+" "+key,xlabel="iteration");axes[row,col].grid(alpha=.2)
        axes[row,0].legend(fontsize=8)
    fig.savefig(output/"crc_cnr_curves.png",dpi=145);plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(16,8),layout='constrained')
    for row,c in enumerate(CHANNELS):
        for gp,style in (("angular","--"),("continuous_energy","-")):
            rr=[r for r in rows if r['group']==gp and r['channel']==c]
            for col,key in enumerate(('p999_density','background_mean','total_integral','integral_recovery')):
                axes[row,col].plot([r['iteration'] for r in rr],[r[key] for r in rr],style,label=gp)
                axes[row,col].set(title=c+'\n'+key,xlabel='iteration');axes[row,col].grid(alpha=.2)
        axes[row,0].legend(fontsize=8)
    fig.savefig(output/'tail_integral_curves.png',dpi=145);plt.close(fig)
    verdict={}
    for c in CHANNELS:
        old=next(r for r in rows if r["group"]=="angular" and r["channel"]==c and r["iteration"]==2000)
        new=next(r for r in rows if r["group"]=="continuous_energy" and r["channel"]==c and r["iteration"]==2000)
        reductions={k:1-new[k]/old[k] for k in ("max_density","peak_background_ratio")}
        costs=[]
        for d in (13,22,37):
            aa=next(r["crc"] for r in spheres if r["group"]=="angular" and r["channel"]==c and r["iteration"]==2000 and r["diameter_mm"]==d)
            bb=next(r["crc"] for r in spheres if r["group"]=="continuous_energy" and r["channel"]==c and r["iteration"]==2000 and r["diameter_mm"]==d)
            costs.append(dict(diameter_mm=d,crc_change=bb-aa,cost_over_5_percentage_points=aa-bb>.05))
        verdict[c]=dict(extreme_spike_improved=all(v>=.5 for v in reductions.values()),reductions=reductions,crc_costs=costs,
                        angular_final=old,continuous_energy_final=new)
    selected_verdict=[]
    for c in CHANNELS:
        for i in ITERATIONS:
            old=next(r for r in rows if r['group']=='angular' and r['channel']==c and r['iteration']==i)
            new=next(r for r in rows if r['group']=='continuous_energy' and r['channel']==c and r['iteration']==i)
            maximum_drop=1-new['max_density']/old['max_density'];peak_drop=1-new['peak_background_ratio']/old['peak_background_ratio']
            for d in (13,22,37):
                aa=next(r['crc'] for r in spheres if r['group']=='angular' and r['channel']==c and r['iteration']==i and r['diameter_mm']==d)
                bb=next(r['crc'] for r in spheres if r['group']=='continuous_energy' and r['channel']==c and r['iteration']==i and r['diameter_mm']==d)
                selected_verdict.append(dict(channel=c,iteration=i,diameter_mm=d,
                    maximum_density_reduction=maximum_drop,peak_background_reduction=peak_drop,
                    extreme_spike_improved=maximum_drop>=.5 and peak_drop>=.5,
                    crc_change_percentage_points=100*(bb-aa),crc_cost_over_5_percentage_points=aa-bb>.05))
    table(output/'selected_comparison_metrics.csv',selected_verdict)
    joint=list(csv.DictReader((geometry_path.parent/'calibration/independent_joint_categories.csv').open()))
    unresolved=sum(r['adequate']=='False' for r in joint if r['model']=='continuous_energy')
    report=dict(study="compton_energy_probability_v5_5e9",event_policy="legacy",actual_primary_gamma=5000000000,through_iteration=2000,selected_iterations=ITERATIONS,
        job=summary['job'],formal_summary_sha256=digest(summary_path),contract_sha256=digest(contract),
        geometry_sha256=digest(geometry_path),whole_active_cells=78920,tiny_cell_metric='not_applicable_whole_cells',
        truth_sha256=digest(truthpath),history_sha256=history_sha,common_normalizers=normalizers,verdict=verdict,
        selected_verdict=selected_verdict,
        method="native full 120 mm metrics; 3 mm linear XY raster for existing truth ROI and display; central 72 mm MIP; no smoothing or crop; gray_r; common A2000 background scale",
        unresolved_joint_categories=unresolved,
        limitation=f"Frozen ideal-trained material proxy evaluated on legacy events; {unresolved} independent joint categories unresolved. The earlier 1e9 study uses ideal events and is not a dose-only control. 2000 iterations do not establish 10000-iteration stability or actual-device performance.")
    (output/"comparison.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    shutil.copy2(truthpath,output/'truth_3mm.npz')
    scripts=output/"scripts";scripts.mkdir()
    for name in ("compare_energy_5e9_v5.py","analyze_nema_result.py","mip_projection.py"):shutil.copy2(HERE/name,scripts/name)
    files={p.relative_to(output).as_posix():digest(p) for p in output.rglob("*") if p.is_file()}
    (output/"artifact_manifest.json").write_text(json.dumps(files,indent=2)+"\n")
    print(json.dumps(verdict,indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',type=int,required=True)
    a=p.parse_args();report=HERE/'reports/NEMA_Body_H60/compton_energy_probability_v5_5e9'
    data=HERE/'generated/compton_energy_probability_v5_5e9';summary=report/'formal_summary.json'
    current=json.loads(summary.read_text())
    if current['job']!=a.job:raise ValueError('Only actual current verified formal pair')
    compare(data/'formal_results'/str(a.job)/'angular',data/'formal_results'/str(a.job)/'continuous_energy',
        report/f'comparison_{a.job}',data/'formal_payload/whole_geometry.npz',summary)
if __name__=='__main__':main()
