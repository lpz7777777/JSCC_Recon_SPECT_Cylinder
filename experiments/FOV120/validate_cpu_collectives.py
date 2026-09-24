"""Portable two-process GLOO check using FileStore (also works without Windows libuv)."""
import os
from pathlib import Path
import sys
import tempfile

import torch.multiprocessing as mp


def worker(rank,store):
    root=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(root/"distributed/dual_energy_compton_python"))
    os.environ["RANK"]=str(rank)
    os.environ["WORLD_SIZE"]="2"
    os.environ["JSCC_TEST_INIT_METHOD"]=store
    from validate_synthetic_distributed import main
    main()


if __name__=="__main__":
    with tempfile.TemporaryDirectory(prefix="jscc-fov120-gloo-") as folder:
        mp.spawn(worker,args=((Path(folder)/"store").as_uri(),),nprocs=2,join=True)
