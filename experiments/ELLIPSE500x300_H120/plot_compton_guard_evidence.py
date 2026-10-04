"""A440 interpolation diagnostics; these figures are not reconstructed images."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    evidence={}
    for column,name,title in ((0,'midpoint_gate.json','Axial midpoint A440'),
                              (1,'radial_midpoint_gate.json','Radial midpoint A440')):
        path=a.report/name;value=json.loads(path.read_text());r=value['cases'];x=np.arange(len(r))
        labels=[','.join(f'{v:g}' for v in row['xyz_mm']) for row in r]
        axes[0,column].bar(x,[100*row['relative_l2'] for row in r],color='#4876a7')
        axes[0,column].axhline(1,color='#a33333',linestyle='--',label='1% L2 reference')
        axes[0,column].set_title(title+' (same physics; independent samples)')
        axes[0,column].set_ylabel('Detector-distribution relative L2 (%)');axes[0,column].legend()
        axes[1,column].bar(x,[100*row['total_response_relative_error'] for row in r],color='#4d9471')
        axes[1,column].set_ylabel('Sum of response: relative difference (%)')
        for row in (0,1):
            axes[row,column].set_xticks(x,labels,rotation=70,fontsize=8)
            axes[row,column].grid(axis='y',alpha=.25)
        axes[1,column].set_xlabel('Source coordinate x,y,z (mm)')
        evidence[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    fig.suptitle('Interpolation diagnostics only: global sums can hide detector-bin redistribution',fontsize=13)
    output=a.report/'A440_interpolation_diagnostics.png';fig.savefig(output,dpi=160);plt.close(fig)
    evidence[output.name]=hashlib.sha256(output.read_bytes()).hexdigest()
    (a.report/'interpolation_figure_manifest.json').write_text(json.dumps(dict(
        classification='response diagnostics, not reconstruction',artifacts_sha256=evidence,
        no_smoothing=True),indent=2)+'\n')


if __name__=='__main__':main()
