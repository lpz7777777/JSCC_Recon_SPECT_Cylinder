"""CPU split-run equivalence and frozen loop-body identity; no HPC job."""
import ast,json
from pathlib import Path
from types import SimpleNamespace
import torch
from jscc_5e10_joint_mlem import joint_mlem as original
from joint_mlem_resume import joint_mlem

def loop(source):
    f=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='joint_mlem')
    return ast.dump(next(n for n in f.body if isinstance(n,ast.For)).body[1],include_attributes=False), [ast.dump(n,include_attributes=False) for n in next(n for n in f.body if isinstance(n,ast.For)).body]

def main():
    root=Path(__file__).parent;torch.set_num_threads(2);torch.manual_seed(35100101)
    assert loop((root/'jscc_5e10_joint_mlem.py').read_text())==loop((root/'joint_mlem_resume.py').read_text())
    matrices=[torch.rand(7,11)+.05 for _ in range(3)]
    response=SimpleNamespace(geometry=SimpleNamespace(active_count=11,views=3),full_rows=matrices[0],matrix=lambda v:matrices[v])
    projection=torch.rand(7,3)*3
    events=[[],[torch.rand(32,11),torch.rand(3,11)],[torch.rand(1,11)]]
    ss=sum(a.sum(0)[:,None] for a in matrices);sc=torch.rand(11,1)+1
    final,history=original(response,projection,events,ss,sc,100,10)
    first,h0=original(response,projection,events,ss,sc,50,10)
    steps=[]
    resumed,h=joint_mlem(response,projection,events,ss,sc,100,10,True,lambda i,h:steps.append(i),None,
                         initial_image=first,initial_history=list(h0),completed_iterations=50)
    assert torch.equal(final,resumed) and torch.equal(history,h) and steps==[60,70,80,90,100]
    fresh,hfresh=joint_mlem(response,projection,events,ss,sc,100,10)
    assert torch.equal(final,fresh) and torch.equal(history,hfresh)
    nonroot,hnone=joint_mlem(response,projection,events,ss,sc,100,10,False,None,None,
                            initial_image=first,initial_history=[],completed_iterations=50)
    assert torch.equal(final,nonroot) and hnone is None
    rejected=0
    for kwargs in [dict(completed_iterations=55),dict(completed_iterations=50),
                   dict(completed_iterations=50,initial_image=first,initial_history=[]),
                   dict(completed_iterations=50,initial_image=first+1,initial_history=list(h0)),
                   dict(completed_iterations=100,initial_image=final,initial_history=list(history))]:
        try:joint_mlem(response,projection,events,ss,sc,100,10,**kwargs)
        except ValueError:rejected+=1
    assert rejected==5
    print('WIDTH_RESUME_PASS',json.dumps(dict(loop_AST_identical=True,split_history_bitwise=True,
        fresh_bitwise=True,nonroot_bitwise=True,invalid_resume_rejected=rejected,synthetic_small_matrices=True)))

if __name__=='__main__':main()
