"""Export frozen paired identities and q rejections for review; never changes selection."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / 'generated/compton_first_scatter_v2'
OUT = HERE / 'reports/NEMA_Body_H60/compton_first_scatter_v2/event_manifests'
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def export():
    OUT.mkdir(exist_ok=True)
    identities={g:{tuple(map(int,x)) for x in np.load(DATA/f'analysis/NEMA_{g}_event_identities.npy')} for g in ('legacy','ideal')}
    a,b=identities['legacy'],identities['ideal']; changed=a^b
    if (len(a),len(b),len(a&b),len(a-b),len(b-a))!=(97078,91231,90778,6300,453):raise ValueError('Frozen accepted identities changed')
    rejected={};sources={};rows=[];cuts={g:[] for g in identities}
    for g in identities:
        p=DATA/f'analysis/NEMA_{g}_q_rejected.csv';sources[str(p.relative_to(HERE))]=sha(p)
        rejected[g]={(int(r['view']),int(r['row'])):r for r in csv.DictReader(p.open())}
        p=DATA/f'analysis/NEMA_{g}_event_identities.npy';sources[str(p.relative_to(HERE))]=sha(p)
    fields=('dataset','worker','seed','view','event_id','legacy_row','ideal_row','global_legacy_row','global_ideal_row','reason','primary_mev','c1','c2','legacy_c1','legacy_c2')
    for p in sorted((DATA/'analysis_inputs/NEMA').glob('events_v*.csv')):
        sources[str(p.relative_to(HERE))]=sha(p)
        with p.open(newline='') as f:
            for r in csv.DictReader(f):
                identity=(int(r['seed']),int(r['event_id']))
                if identity in changed:
                    rows.append(dict(group='A_only' if identity in a else 'B_only',**{k:r[k] for k in fields}))
                for g in identities:
                    key=(int(r['view']),int(r['global_'+g+'_row']))
                    if key in rejected[g]:
                        cuts[g].append({'group':g,**{k:r[k] for k in fields},**rejected[g][key]})
    if len(rows)!=6753 or len(cuts['legacy'])!=221 or len(cuts['ideal'])!=154:raise ValueError('Incomplete metadata joins')
    for name,records in [('paired_only_events.csv',rows),('legacy_q_rejected_events.csv',cuts['legacy']),('ideal_q_rejected_events.csv',cuts['ideal'])]:
        with (OUT/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    manifest=dict(study='compton_first_scatter_v2',post_q_counts=dict(A=97078,B=91231,common=90778,A_only=6300,B_only=453),q_rejected=dict(legacy=221,ideal=154),row_convention='Zero based local List rows and per-view merged global List rows; (seed,event_id) is shared paired identity',sources=sources,files={p.name:sha(p) for p in OUT.glob('*.csv')})
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('PAIRED_EVENT_MANIFESTS_VERIFIED',len(rows),[len(cuts[g]) for g in identities])
if __name__=='__main__':export()
