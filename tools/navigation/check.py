"""Read-only paths, conditions, protected bytes and command-safety validation."""
from __future__ import annotations
import ast
from collections import Counter
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote,urlsplit
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.navigation.build import read,sha,put
from tools.project_index import check as check_catalog

class Links(HTMLParser):
    def __init__(self):super().__init__();self.urls=[]
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        for key in ['href','src']:
            if key in attrs:self.urls.append(attrs[key])

def validate():
    errors=[];checked=0
    catalog=read(ROOT/'docs/ac225/catalog.json');routes=read(ROOT/'Analysis_Plots/results/data/current_results.json')['routes']
    check=check_catalog(catalog,read(ROOT/'docs/ac225/source_index.json'),read(ROOT/'docs/ac225/directory_index.json'))
    errors.extend(check['errors'])
    managed=read(ROOT/'docs/organization/20261117/build_manifest.json')['outputs_sha256']
    extra=['README.md','docs/README.md','docs/ac225/README.md','docs/ac225/OPERATIONS.md','docs/ac225/CODE_GUIDE.md','docs/ac225/SCIENCE_CONVENTIONS.md','docs/ac225/WORKFLOW.md','docs/ac225/RESULTS.md','tools/navigation/README.md','Auxiliary_Studies/README.md','Geant4Sim/README.md','experiments/README.md','experiments/ELLIPSE500x300_H120/README.md','Analysis_Plots/README.md','Analysis_Plots/research/README.md','Analysis_Plots/results/README.md','docs/organization/20261117/README.md']
    for name in sorted(set(managed)|set(extra)):
        p=ROOT/name
        if not p.is_file():errors.append('Missing managed document '+name);continue
        if name in managed and sha(p)!=managed[name]:errors.append('Stale generated file SHA '+name)
        if p.suffix not in {'.md','.html'}:continue
        text=p.read_text(encoding='utf8');urls=[]
        if p.suffix=='.md':urls=re.findall(r'\]\(([^\n]+?)\)',text)
        else:
            parser=Links();parser.feed(text);urls=parser.urls
        for value in urls:
            value=value.strip('<>')
            if not value or value.startswith('#') or urlsplit(value).scheme:continue
            target=(p.parent/unquote(value.split('#')[0])).resolve();checked+=1
            if not target.exists():errors.append('Broken link '+name+' -> '+value)
    ids=[r['id'] for r in routes]
    for name,digest in read(ROOT/'Analysis_Plots/results/data/artifact_sha256.json').items():
        if sha(ROOT/name)!=digest:errors.append('Stale parent result artifact SHA '+name)
    if len(ids)!=len(set(ids)):errors.append('Duplicate condition IDs')
    for r in routes:
        for key in ['grid','material','projection_source','correction','width_percent','numerical_qa','visual_qa','physical_certification']:
            if key not in r:errors.append(r['id']+' missing '+key)
        for key in ['report','evidence','pointer']:
            if not (ROOT/r[key]).exists():errors.append(r['id']+' missing '+key+': '+r[key])
        if r['method']=='JSCC' and r['energy_keV']!=440:errors.append('Unsubstantiated 218 joint route '+r['id'])
        if r['method']=='Compton' and r.get('channel')!='440_Compton':errors.append('Pure Compton mislabeled '+r['id'])
        if r['projection_source']=='forward+Poisson' and r['kind']!='synthetic':errors.append('Synthetic marked real '+r['id'])
        if r['material']=='water' and r['method']!='SC':errors.append('Water detector prefix mislabeled as joint '+r['id'])
        if r['physical_certification']!='not-certified':errors.append('Unreviewed physical certification '+r['id'])
        if r['id'].startswith('Cartesian-') and r['delivery']!='not-delivered':errors.append('Unsubstantiated Cartesian delivery '+r['id'])
        if str(r.get('job'))=='1771220' and (r['iterations']!=9950 or r['status']!='failed-preview-only'):errors.append('9950 failed preview promoted '+r['id'])
    baseline=read(ROOT/'docs/organization/20261117/protected_baseline.json')['protected_sha256']
    for name,digest in baseline.items():
        p=ROOT/name
        if not p.is_file() or sha(p)!=digest:errors.append('Protected file changed: '+name)
    moves=read(ROOT/'docs/organization/20261117/moves.json')
    for m in moves:
        if sha(ROOT/m['destination'])!=m['sha256']:errors.append('Archived bytes changed '+m['destination'])
    # Guard actual write targets that previously posed a real regression risk.
    for name in ['experiments/ELLIPSE500x300_H120/AGENTS.md','experiments/ELLIPSE500x300_H120/reports/NEMA_Body_H60/nema5e9_cartesian_he_20261115/freeze.json']:
        before=sha(ROOT/name)
        # Intercept writes even if the destination guard regresses. Validation
        # must never damage the real protected files it is checking.
        with patch.object(Path,'write_bytes',side_effect=AssertionError('protected write attempted')), patch.object(Path,'write_text',side_effect=AssertionError('protected write attempted')):
            try:put(ROOT/name,'MUST NEVER WRITE')
            except ValueError:pass
            except AssertionError:errors.append('Writer attempted protected destination '+name)
            else:errors.append('Writer accepted protected destination '+name)
        if sha(ROOT/name)!=before:errors.append('Write guard modified protected bytes '+name)
    for path in (ROOT/'tools/navigation').glob('*.py'):
        ast.parse(path.read_text(encoding='utf8'),filename=str(path))
    return dict(passed=not errors,links_checked=checked,studies=len(catalog['experiments']),condition_routes=len(routes),route_kinds=dict(Counter(r['kind'] for r in routes)),protected_files=len(baseline),archived_tools=len(moves),catalog_check=check,errors=errors,no_scientific_runner_imports_or_execution=True)

def main():
    result=validate();print(json.dumps(result,ensure_ascii=False,indent=2))
    if not result['passed']:raise SystemExit(1)
if __name__=='__main__':main()
