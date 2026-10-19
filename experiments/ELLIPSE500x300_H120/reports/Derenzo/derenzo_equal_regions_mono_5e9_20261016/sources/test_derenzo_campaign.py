"""Exact design-to-macro and independently counted monoenergy acquisition checks."""
import copy
import ast
import math
import tempfile
import time
import unittest
import numpy as np
from derenzo_campaign_common import *


class DerenzoCampaignTests(unittest.TestCase):
    def test_joint_branch_matches_original_all_saved_values(self):
        import torch
        from torch_active_operator import ActiveGeometry,ViewResponse,compton_and_joint_mlem,_event_weight,_reduce_sum,_update
        tree=ast.parse((HERE/'derenzo_campaign_reconstruction.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='joint_mlem')
        env=dict(torch=torch,time=time,_event_weight=_event_weight,_reduce_sum=_reduce_sum,_update=_update)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'<joint-branch>','exec'),env)
        torch.manual_seed(391)
        g=ActiveGeometry([0,2],[1.,0.,1.],np.tile(np.arange(3)[:,None],(1,20)))
        response=ViewResponse(torch.tensor([[.2,.3,.4],[.5,.1,.3]]),g)
        projection=torch.arange(40,dtype=torch.float32).reshape(2,20)+1
        blocks=[[torch.rand(3,2)+.01,torch.rand(2,2)+.01] for _ in range(20)]
        s=response.sensitivity();cs=torch.tensor([[2.],[3.]])
        _,reference=compton_and_joint_mlem(response,projection,blocks,s,cs,100,50)
        checkpoints=[]
        actual=env['joint_mlem'](response,projection,blocks,s,cs,100,50,True,lambda i,h:checkpoints.append((i,h[-1].clone())))
        for x,y in zip(actual,reference):torch.testing.assert_close(x,y,rtol=0,atol=0)
        self.assertEqual([x[0] for x in checkpoints],[50,100])
        self.assertTrue(torch.equal(checkpoints[-1][1],actual[0]))

    def test_every_macro_cylinder_matches_referenced_geometry_and_volume_weight(self):
        for energy in (218,440):
            design=read(DESIGN/'data'/f'derenzo_{energy}keV_geometry.json')
            for view in range(1,21):
                text=macro_text(design,view,PER_WORKER)
                self.assertIn(f'/ellipse/angle {(view-1)*18}\n',text)
                self.assertIn('/ellipse/centerY -345\n',text)
                self.assertEqual(text.count('/run/beamOn '),1)
                self.assertIn('/run/beamOn 25000000\n',text)
                rows=[list(map(float,line.split()[1:])) for line in text.splitlines() if line.startswith('/ellipse/addRod ')]
                self.assertEqual(len(rows),114 if energy==218 else 53)
                for a,r in zip(rows,design['rods']):
                    expected=[energy,r['x_mm'],r['y_mm'],0,r['diameter_mm']/2,30,r['volume_mm3']]
                    np.testing.assert_array_equal(a,expected)
                self.assertEqual({a[0] for a in rows},{energy})
                density=[a[-1]/(math.pi*a[4]**2*(2*a[5])) for a in rows]
                np.testing.assert_allclose(density,1,rtol=1e-14)

    def test_four_acquisitions_have_disjoint_seeds_and_each_exactly_5e9(self):
        all_seeds=[]
        for d in DATASETS:
            info=dataset_info(d)
            jobs=[dict(index=i,view=i//10+1,worker=i%10,photons=PER_WORKER,
                       seed=info['seed_base']+i,energy_keV=info['energy_keV']) for i in range(WORKERS)]
            registry=dict(study=STUDY,dataset=d,total_primary_photons=TOTAL,views=20,workers_per_view=10,jobs=jobs)
            validate_registry(registry,d)
            self.assertEqual(sum(j['photons'] for j in jobs),5_000_000_000)
            all_seeds += [info['seed_base']-1]+[j['seed'] for j in jobs]
            bad=copy.deepcopy(registry);bad['jobs'][1]['seed']=bad['jobs'][0]['seed']
            with self.assertRaises(ValueError):validate_registry(bad,d)
        self.assertEqual(len(all_seeds),len(set(all_seeds)))

    def test_actual_monoenergy_dose_and_system_bins_cannot_be_substituted(self):
        for d in DATASETS:
            info=dataset_info(d)
            c=dict(passed=True,study=STUDY,dataset=d,total_primary_photons=TOTAL,
                   primary_counts=[TOTAL,0,0] if info['energy_keV']==218 else [0,TOTAL,0],
                   bins=info['bins'],workers=200,views=20,data_kind='actual_Geant4_transport',source_solid_angle='4pi')
            self.assertIs(validate_collection(c,d),c)
            for key,value in [('total_primary_photons',TOTAL//4),('data_kind','forward_projection_plus_Poisson'),('bins',1),('primary_counts',[TOTAL//2,TOTAL//2,0])]:
                with self.assertRaises(ValueError):validate_collection(dict(c,**{key:value}),d)


if __name__=='__main__':unittest.main()
