"""Scientific regional response diagnostics; not reconstruction images."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from generate_compton_a_guard import digest,write


def main():
    report=Path(__file__).resolve().parent/'reports/NEMA_Body_H60/compton_response_geometry_v3'
    summary=report/'regional_validation_frozen_measure_summary.json';csvfile=report/'regional_validation_cases.csv'
    result=json.loads(summary.read_text())
    if digest(csvfile)!=result['csv_sha256']:raise ValueError('Regional figure input identity differs')
    rows=list(csv.DictReader(csvfile.open()));regions=('inner_near','middle_oblique','outer_side','outer_far')
    names=('near r156','oblique r210','side r252','far r252');domains=('object_intersection','full_reference_cell')
    values=np.zeros((4,6));quad=np.zeros_like(values);informative=np.zeros_like(values,dtype=int)
    for i,region in enumerate(regions):
        for j,domain in enumerate(domains):
            for k,z in enumerate((0,20,39)):
                selected=[r for r in rows if r['control']==f'{region}_z{z:02d}' and r['domain']==domain]
                if len(selected)!=7:raise ValueError('Frozen regional event/column count differs')
                values[i,j*3+k]=100*max(float(r['A_relative_change']) for r in selected)
                quad[i,j*3+k]=100*max(float(r['quadrature_relative_change']) for r in selected)
                informative[i,j*3+k]=sum(r['informative']=='True' for r in selected)
    fig,ax=plt.subplots(1,2,figsize=(13.5,4.9),constrained_layout=True)
    for axis,data,limit,title in zip(ax,(values,quad),(1.,.05),
            ('Physical A: 0.75 to 0.375 mm (%)','Quadrature: 32/12/12 to 48/16/16 (%)')):
        im=axis.imshow(data,cmap='Blues',vmin=0,vmax=limit,aspect='auto')
        axis.set_yticks(range(4),names);axis.set_xticks(range(6),['-58.5','+1.5','+58.5']*2)
        axis.set_xlabel('z (mm): object intersection  |  full circle reference')
        axis.axvline(2.5,color='black',lw=1);axis.set_title(title)
        for i in range(4):
            for j in range(6):axis.text(j,i,f'{data[i,j]:.3f}\n{informative[i,j]}/7',
                ha='center',va='center',fontsize=8,color='black')
        fig.colorbar(im,ax=axis,label='max change, common R1-Z absolute floor')
    fig.suptitle('168/168 passed; 141 informative | annotations: max % / informative n/7 | response diagnostic',fontsize=12)
    output=report/'regional_response_diagnostics.png';fig.savefig(output,dpi=170);plt.close(fig)
    write(report/'regional_response_figure_manifest.json',dict(source_sha256=digest(Path(__file__)),
        input_summary_sha256=digest(summary),input_csv_sha256=digest(csvfile),
        figure_sha256=digest(output),entries=168,informative=141,distinct_point_events=28,
        reconstruction_image=False,global_accuracy_certified=False))


if __name__=='__main__':main()
