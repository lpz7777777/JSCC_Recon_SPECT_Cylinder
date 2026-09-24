"""Scan raw Cartesian matrices and fingerprint their central 60-mm columns."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def fingerprint(path,z_layers,detectors=11520):
    expected=detectors*z_layers*51*51*4
    if path.stat().st_size!=expected or z_layers<20 or (z_layers-20)%2:
        raise ValueError('Wrong Cartesian matrix shape')
    values=np.memmap(path,dtype='<f4',mode='r',shape=(detectors,z_layers,51,51))
    start=(z_layers-20)//2
    digest=hashlib.sha256();minimum=float('inf');maximum=0.
    for row in range(0,detectors,16):
        block=values[row:row+16]
        if not np.isfinite(block).all() or np.any(block<0):
            raise ValueError(f'Invalid values at detector row {row}')
        minimum=min(minimum,float(block.min()));maximum=max(maximum,float(block.max()))
        digest.update(block[:,start:start+20].tobytes(order='C'))
    return {'path':str(path),'z_layers':z_layers,'bytes':expected,
            'central_20_layers_sha256':digest.hexdigest(),'min':minimum,'max':maximum,
            'finite_nonnegative':True}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix',type=Path,required=True)
    parser.add_argument('--z-layers',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=fingerprint(args.matrix,args.z_layers)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
