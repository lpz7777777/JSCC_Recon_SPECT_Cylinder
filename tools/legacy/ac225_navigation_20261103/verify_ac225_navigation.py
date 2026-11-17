"""Local headless navigation/filter/links QA; never opens the user's profile."""
from pathlib import Path
from collections import Counter
from html.parser import HTMLParser
import hashlib
import json
import subprocess
import tempfile
import os
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/organization/20261103'

def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
class HTML(HTMLParser):
    def __init__(self):super().__init__();self.qa=None
    def handle_starttag(self,t,a):
        a=dict(a)
        if t=='html' and 'data-navigation-qa' in a:self.qa=json.loads(a['data-navigation-qa'])

def main():
    chrome=next(p for p in [Path('C:/Program Files/Google/Chrome/Application/chrome.exe'),Path('C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe')] if p.is_file())
    results=read(ROOT/'Analysis_Plots/results/data/current_results.json')['routes']
    catalog=read(ROOT/'docs/ac225/catalog.json')['experiments']
    assert len(results)==len({r['id'] for r in results})==36
    counts=Counter(r['kind'] for r in results);assert counts=={'actual':20,'synthetic':16}
    assert all(r['views'] in [20,60] and r['energy_keV'] in [218,440] for r in results)
    assert all(r['kind']=='synthetic' for r in results if r['views']==60)
    assert all(r['energy_keV']==440 for r in results if r['method']=='JSCC')
    # Every completed actual EHE route uses the dense matrix, never the old
    # EHE rows retained in the historical15-route gallery.
    ehe=[r for r in results if r['kind']=='actual' and r['method']=='EHE' and r['status'].startswith('completed')]
    assert len(ehe)==6 and all('dense65536' in r['matrix'] for r in ehe)
    image_dir=OUT/'visual_qa';image_dir.mkdir(exist_ok=True)
    checks={}
    with tempfile.TemporaryDirectory(prefix='codex_ac225_nav_') as work:
        work=Path(work).resolve()
        # TemporaryDirectory's recursive cleanup is restricted to this verified
        # newly created task directory directly under the system temp root.
        assert work.parent==Path(tempfile.gettempdir()).resolve() and work.name.startswith('codex_ac225_nav_')
        (work/'created_for_this_qa.txt').write_text('Localnavigationtestonly',encoding='utf8')
        def run(uri,flags):
            args=[str(chrome),'--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check',
                  '--disable-background-networking','--disable-extensions','--disable-sync','--disable-component-update',
                  '--hide-scrollbars','--window-size=1700,1200','--virtual-time-budget=1500','--user-data-dir='+str(work/'profile')]+flags+[uri]
            r=subprocess.run(args,capture_output=True,timeout=50,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            if r.returncode:raise RuntimeError('Headless navigation QA exit '+str(r.returncode))
            return r.stdout.decode('utf8',errors='replace')
        for key,path,total in [('project',ROOT/'docs/ac225/INDEX.html',len(catalog)),('results',ROOT/'Analysis_Plots/results/REPORT.html',len(results))]:
            page=path.read_text('utf8')
            if key=='results':
                script='''const checks={};function visible(){return [...document.querySelectorAll('tbody tr')].filter(r=>!r.hidden).length;}
checks.initial=visible();q.value='NEMA 5e12 60';q.dispatchEvent(new Event('input'));checks.energyPair60=visible();
q.value='';k.value='actual';k.dispatchEvent(new Event('change'));checks.actual=visible();
k.value='synthetic';k.dispatchEvent(new Event('change'));checks.synthetic=visible();
k.value='';q.value='1761195';q.dispatchEvent(new Event('input'));checks.pendingJoint=visible();
checks.noHorizontalOverflow=document.documentElement.scrollWidth<=innerWidth;
document.documentElement.setAttribute('data-navigation-qa',JSON.stringify(checks));'''
            else:
                script='''const checks={};function visible(){return [...document.querySelectorAll('tbody tr')].filter(r=>!r.hidden).length;}
checks.initial=visible();q.value='60view';q.dispatchEvent(new Event('input'));checks.study60=visible();
q.value='';k.value='synthetic';k.dispatchEvent(new Event('change'));checks.syntheticStudies=visible();
checks.noHorizontalOverflow=document.documentElement.scrollWidth<=innerWidth;
document.documentElement.setAttribute('data-navigation-qa',JSON.stringify(checks));'''
            fixture=page.replace('<meta charset="utf-8">','<meta charset="utf-8"><base href="'+path.parent.as_uri()+'/">',1)
            fixture=fixture.replace('</html>','<script>'+script+'</script></html>')
            test=work/(key+'.html');test.write_text(fixture,encoding='utf8')
            parsed=HTML();parsed.feed(run(test.as_uri(),['--dump-dom']));assert parsed.qa is not None
            c=parsed.qa;assert c['initial']==total and c['noHorizontalOverflow'],c
            if key=='results':assert (c['energyPair60'],c['actual'],c['synthetic'],c['pendingJoint'])==(2,20,16,1),c
            else:assert (c['study60'],c['syntheticStudies'])==(1,2),c
            png=image_dir/(key+'_index.png');run(path.as_uri(),['--screenshot='+str(png)])
            assert png.is_file() and png.stat().st_size>10000
            checks[key]=dict(passed=True,DOM=c,page_sha256=sha(path),screenshot=str(png.relative_to(ROOT)).replace('\\','/'),screenshot_sha256=sha(png),direct_visual_review_pending=True)
    audit=read(OUT/'organization_manifest.json');assert audit['passed'] and audit['protected_original_source_SHA_unchanged']
    cli=subprocess.run(['python','-X','utf8',str(ROOT/'tools/project_index.py'),'check'],capture_output=True,text=True,encoding='utf8',check=True)
    query=read(ROOT/'docs/ac225/catalog.json');assert any(s['id']=='ehe-nema-poisson-60view' for s in query['experiments'])
    proof=dict(passed=True,checked_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),route_kind_counts=dict(counts),completed_actual_EHE_routes_all_dense=True,
        no60view_claimed_actual_transport=True,JSCC_joint_only440=True,local_CLI_check=json.loads(cli.stdout),browser=checks,
        no_scientific_execution=True,no_array_moves=True,temporary_profile_was_isolated=True,direct_visual_review_pending=True)
    (OUT/'navigation_qa.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('NAVIGATION_BROWSER_AND_SEMANTIC_QA_PASS',json.dumps({k:v['DOM'] for k,v in checks.items()}))

if __name__=='__main__':main()
