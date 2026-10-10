import copy, unittest
from jscc_5e10_array import array_accounting, array_receipt_identity

def rows(workers=1000):
    out=[]
    for i in range(workers):
        for suffix in ('','batch','extern','0'):
            s='.'+suffix if suffix else ''
            out.append(f'2000_{i}{s}|{3000+i}{s}|COMPLETED|0:0|1|1|100K|00:20:00|cpu=1,node=1|cn01')
    return '\n'.join(out)

class ArrayIdentityTests(unittest.TestCase):
    def test_all_independent_workers_and_steps_required(self):
        self.assertTrue(array_accounting(rows(),2000)['passed'])
        self.assertFalse(array_accounting(rows(999),2000)['passed'])
        self.assertFalse(array_accounting(rows().replace('2000_8.batch|3008.batch|COMPLETED','2000_8.batch|3008.batch|FAILED'),2000)['passed'])
        self.assertFalse(array_accounting('\n'.join(x for x in rows().splitlines() if not x.startswith('2000_4.0|')),2000)['passed'])
    def test_receipt_binds_child_not_parent(self):
        p=array_accounting(rows(),2000)
        r=dict(array_job='2000',array_task=4,allocation_job='3004')
        a=dict(job='3004',scontrol='JobId=3004 ArrayJobId=2000 ArrayTaskId=4 NumNodes=1 NumCPUs=1 NumTasks=1 CPUs/Task=1')
        self.assertTrue(array_receipt_identity(r,a,p,4))
        for key,value in [('array_task',5),('array_job','2001'),('allocation_job','2000')]:
            bad=dict(r);bad[key]=value
            with self.assertRaises(ValueError):array_receipt_identity(bad,a,p,4)
        bad=dict(a,scontrol=a['scontrol'].replace('NumCPUs=1','NumCPUs=2'))
        with self.assertRaises(ValueError):array_receipt_identity(r,bad,p,4)
    def test_duplicate_or_other_project_rejected(self):
        with self.assertRaises(ValueError):array_accounting(rows()+'\n'+rows().splitlines()[0],2000)
        with self.assertRaises(ValueError):array_accounting(rows().replace('2000_0|','2001_0|',1),2000)

if __name__=='__main__':unittest.main()
