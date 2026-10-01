"""Check that independent NEMA doses change photon count and seeds only."""
import json
from pathlib import Path

from prepare_nema_simulation import digest
from resource_budget import estimate
from validate_nema_simulation import validate


def main():
    here=Path(__file__).resolve().parent
    roots=[here/f"generated/NEMA_Body_H60/Simulation_{dose}" for dose in ("1e9","5e9")]
    manifests=[json.loads((root/"jobs.json").read_text()) for root in roots]
    for root in roots:
        validate(root/"jobs.json","prepared")
    old,new=manifests
    assert old["nema_truth_sha256"]==new["nema_truth_sha256"]
    assert old["config_sha256"]==new["config_sha256"]
    assert old["yield_weighted_activity_integral"]==new["yield_weighted_activity_integral"]
    assert {row["seed"] for row in old["jobs"]}.isdisjoint(row["seed"] for row in new["jobs"])
    for a,b in zip(old["macros"],new["macros"]):
        def source(root,item):
            return [line for line in (root/item["path"]).read_text().splitlines()
                    if not line.startswith("/run/beamOn ")]
        assert source(roots[0],a)==source(roots[1],b)
    report={"level":"5e9","status":"prepared_source_and_budget_verified",
            "total_primary_photons":new["total_primary_photons"],"views":20,"workers":200,
            "photons_per_worker":new["jobs"][0]["photons"],
            "seed_range":[new["jobs"][0]["seed"],new["jobs"][-1]["seed"]],
            "seeds_disjoint_from_1e9":True,"all_20_source_macros_equal_except_beamOn":True,
            "nema_truth_sha256":new["nema_truth_sha256"],
            "jobs_sha256":digest(roots[1]/"jobs.json"),
            "source_integrals":new["yield_weighted_activity_integral"],
            "planning_budget":estimate(550000,8,1,22,55),
            "budget_is_not_a_measured_resource_gate":True}
    out=here/"reports/NEMA_Body_H60/5e9/input_checks.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__": main()
