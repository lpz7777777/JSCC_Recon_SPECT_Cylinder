"""Make display-only endpoint-trimmed MIPs from verified NEMA histories."""
import argparse
import json
from pathlib import Path
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from analyze_nema_result import digest, xy_interpolator
from plot_nema_iterations import CHANNELS, SELECTED
from mip_projection import axial_mip, axial_selection, DEFAULT_TRIM_LAYERS

HERE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result")
    parser.add_argument("--trim-layers",type=int,default=DEFAULT_TRIM_LAYERS)
    args=parser.parse_args()
    if "/" in args.result or "\\" in args.result: parser.error("Use one result directory name")
    source=HERE/"reports/NEMA_Body_H60"/args.result
    analysis=json.loads((source/"analysis.json").read_text())
    ip=HERE/"reports"/f"{args.result}_integrity.json"
    integrity=json.loads(ip.read_text())
    if digest(ip)!=analysis["integrity_sha256"]: raise ValueError("Integrity report hash mismatch")
    data=HERE/"generated/RemoteResults"/args.result
    run=json.loads((data/"run_manifest.json").read_text())
    truth_path=HERE/"generated/NEMA_Body_H60/truth_3mm.npz"
    if digest(truth_path)!=analysis["truth_sha256"]: raise ValueError("Truth hash mismatch")
    with np.load(truth_path) as t:
        x,y,z=(t[f"{a}_mm"].copy() for a in "xyz")
        truth={e:t[f"activity_{e}_zyx"].copy() for e in (218,440)}
    truth["sum"]=(.114*truth[218]+.259*truth[440])/.373
    _,policy=axial_selection(z,args.trim_layers)
    out=source/f"mip_trim{args.trim_layers}"
    out.mkdir(exist_ok=True)
    geometry_path=HERE/"generated/Geometry/geometry.npz"
    if digest(geometry_path)!=run["geometry_sha256"]: raise ValueError("Geometry mismatch")
    with np.load(geometry_path) as g:
        xyz=g["coordinates_mm"].copy(); active=g["active_indices"].copy()
    if not np.allclose(np.unique(xyz[:,2]),z): raise ValueError("Axial grid mismatch")
    sampler=xy_interpolator(xyz[:3301,:2],x,y)
    xx,yy=np.meshgrid(x,y,indexing="xy")
    ellipse=(xx/250)**2+(yy/150)**2<=1
    extent=(x[0]-1.5,x[-1]+1.5,y[0]-1.5,y[-1]+1.5)
    images=[]; finals=[]; peak_stats=[]; histories={}
    for channel,title,energy in CHANNELS:
        path=data/f"Image_{channel}_history.float32"
        expected=analysis["summary"][channel]["history_sha256"]
        if path.stat().st_size!=200*len(active)*4 or digest(path)!=expected:
            raise ValueError("History size/hash mismatch")
        histories[channel]=expected
        hist=np.memmap(path,dtype="<f4",mode="r",shape=(200,len(active)))
        scalar=analysis["summary"][channel]["fixed_display_background"]
        if scalar<=0: raise ValueError("Invalid fixed BG normalization")
        panels=[]
        full=np.zeros(len(xyz),dtype=np.float32)
        for iteration in SELECTED:
            values=hist[iteration//50-1]
            if not np.isfinite(values).all() or np.any(values<0): raise ValueError("Invalid frame")
            full.fill(0); full[active]=values
            volume=np.maximum(sampler(full.reshape(len(z),3301)),0)/scalar
            volume[:,~ellipse]=0
            panels.append(axial_mip(volume,z,args.trim_layers))
        images.append((title,energy,panels))
        finals.append((title,energy,volume))
        peak_stats.append(dict(channel=channel,full_mip_max=float(axial_mip(volume,z,0).max()),
            trimmed_mip_max=float(panels[-1].max()),fixed_final_background=scalar))
    lo,hi=policy["retained_slab_bounds_mm"]
    heading=f"MIP: axial slab [{lo:g},{hi:g}] mm; remove {args.trim_layers} layers per end; sigma=0"
    fig,axes=plt.subplots(6,7,figsize=(22,15),layout="constrained")
    for row,(title,energy,panels) in enumerate(images):
        for col,panel in enumerate([axial_mip(truth[energy],z,args.trim_layers)]+panels):
            im=axes[row,col].imshow(panel,origin="lower",extent=extent,cmap="gray_r",vmin=0,vmax=10,interpolation="nearest")
            axes[row,col].set(title="truth MIP" if col==0 else f"iter {SELECTED[col-1]}",xlim=(-252,252),ylim=(-150,150))
            if col==0: axes[row,col].set_ylabel(title)
            axes[row,col].tick_params(labelsize=7)
    fig.colorbar(im,ax=axes,shrink=.6,label="relative gamma density; fixed final BG normalization")
    fig.suptitle(f"NEMA H60 {run['count_level']}: {heading}")
    fig.savefig(out/"mip_iterations.png",dpi=115);plt.close(fig)
    fig,axes=plt.subplots(6,3,figsize=(12,15),layout="constrained")
    for row,(title,energy,volume) in enumerate(finals):
        for col,panel in enumerate((axial_mip(truth[energy],z,args.trim_layers),axial_mip(volume,z,0),axial_mip(volume,z,args.trim_layers))):
            im=axes[row,col].imshow(panel,origin="lower",extent=extent,cmap="gray_r",vmin=0,vmax=10,interpolation="nearest")
            axes[row,col].set(title=("truth MIP","full z MIP","trimmed z MIP")[col],xlim=(-252,252),ylim=(-150,150))
            if col==0: axes[row,col].set_ylabel(title)
    fig.colorbar(im,ax=axes,shrink=.6,label="same scale 0..10; white low, black high")
    fig.suptitle(f"10000 iterations; {heading}")
    fig.savefig(out/"mip_full_vs_trimmed.png",dpi=120);plt.close(fig)
    # Axial/coronal/sagittal views retain all layers; only the MIP excludes ends.
    ix,iy,iz=int(np.argmin(abs(x))),int(np.argmin(abs(y))),20
    fig,axes=plt.subplots(6,8,figsize=(23,17),layout="constrained")
    for row,(title,energy,volume) in enumerate(finals):
        for kind in range(4):
            for col,v in enumerate((truth[energy],volume)):
                if kind==0: panel,ex=v[iz],extent
                elif kind==1: panel,ex=v[:,iy,:],(x[0]-1.5,x[-1]+1.5,-60,60)
                elif kind==2: panel,ex=v[:,:,ix],(y[0]-1.5,y[-1]+1.5,-60,60)
                else: panel,ex=axial_mip(v,z,args.trim_layers),extent
                im=axes[row,2*kind+col].imshow(panel,origin="lower",extent=ex,cmap="gray_r",vmin=0,vmax=10,interpolation="nearest")
                axes[row,2*kind+col].set_title(("axial z=+1.5","coronal full z","sagittal full z","MIP trimmed z")[kind]+(" truth" if col==0 else " recon"),fontsize=8)
                if kind==0 and col==0: axes[row,col].set_ylabel(title)
                axes[row,2*kind+col].tick_params(labelsize=6)
    fig.colorbar(im,ax=axes,shrink=.6,label="relative gamma density")
    fig.suptitle(f"NEMA H60 {run['count_level']}: only MIP trimmed; {heading}")
    fig.savefig(out/"final_multiplanar.png",dpi=115);plt.close(fig)
    metadata=dict(result=args.result,policy=policy,projection="max along z after endpoint selection",selected_iterations=SELECTED,
        truth_sha256=digest(truth_path),geometry_sha256=digest(geometry_path),analysis_sha256=digest(source/"analysis.json"),
        integrity_sha256=digest(ip),history_sha256=histories,display_range=[0,10],colormap="gray_r",gaussian_sigma=0,xy_crop=0,
        normalization="one existing final BG scalar per channel fixed over all iterations; no per-frame fit",peaks=peak_stats,
        warning="Display-only trimmed MIP does not demonstrate full-FOV edge stability; retain untrimmed images and original quantitative metrics")
    (out/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    (out/"scripts").mkdir(exist_ok=True)
    for name in ("plot_nema_mip.py","mip_projection.py"):
        shutil.copy2(HERE/name,out/"scripts"/name)
    (out/"README.md").write_text(f"# NEMA H60：排除轴向端层的MIP\n\n上下各排除{args.trim_layers}层（每端{policy['removed_mm_each_end']:g}mm），保留{policy['retained_layer_count']}层，中心范围{policy['retained_center_range_mm']}mm，对应投影体积z∈[{lo:g},{hi:g}]mm。只改变MIP显示，重建数组、轴冠矢位、CRC/CNR/CV与积分/泄漏指标不变。横向椭圆FOV完整，无平滑，白低黑高，固定色标0..10。附图iterations_z20.png是z=+1.5mm单层轴位图，不是MIP。\n\n![六路逐迭代MIP](mip_iterations.png)\n\n![全轴向MIP与端层排除MIP的同色标对照](mip_full_vs_trimmed.png)\n\n![多平面：仅MIP排除端层](final_multiplanar.png)\n\n排除端层后仍可能存在内部噪声尖峰；改善显示不能作为边缘性能已修复的证据。原始全轴向MIP保留对照，输入/脚本哈希及实际选择见metadata.json。\n",encoding="utf-8")
    print(out)
    print(json.dumps(peak_stats,indent=2))


if __name__=="__main__": main()
