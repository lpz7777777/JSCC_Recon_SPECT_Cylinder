# transport源码

职责索引用于打开源码；冻结副本和历史控制器不作为新的执行入口。

| 文件 | 身份 | 说明 |
|---|---|---|
| [Geant4Sim/Geant4Code/src/EventAction.cc](../../../../Geant4Sim/Geant4Code/src/EventAction.cc) | library-or-historical-tool | EventAction |
| [Geant4Sim/Geant4Code/src/RunAction.cc](../../../../Geant4Sim/Geant4Code/src/RunAction.cc) | library-or-historical-tool | RunAction |
| [Geant4Sim/Geant4Code/src/SteppingAction.cc](../../../../Geant4Sim/Geant4Code/src/SteppingAction.cc) | library-or-historical-tool | SteppingAction |
| [Geant4Sim/Geant4Code_CntStatOnly/src/EventAction.cc](../../../../Geant4Sim/Geant4Code_CntStatOnly/src/EventAction.cc) | library-or-historical-tool | EventAction |
| [Geant4Sim/Geant4Code_CntStatOnly/src/RunAction.cc](../../../../Geant4Sim/Geant4Code_CntStatOnly/src/RunAction.cc) | library-or-historical-tool | RunAction |
| [Geant4Sim/Geant4Code_CntStatOnly/src/SteppingAction.cc](../../../../Geant4Sim/Geant4Code_CntStatOnly/src/SteppingAction.cc) | library-or-historical-tool | SteppingAction |
| [Geant4Sim/Geant4Code_CntStatResponseStudy/src/EventAction.cc](../../../../Geant4Sim/Geant4Code_CntStatResponseStudy/src/EventAction.cc) | library-or-historical-tool | EventAction |
| [Geant4Sim/Geant4Code_CntStatResponseStudy/src/RunAction.cc](../../../../Geant4Sim/Geant4Code_CntStatResponseStudy/src/RunAction.cc) | library-or-historical-tool | RunAction |
| [Geant4Sim/Geant4Code_CntStatResponseStudy/src/SteppingAction.cc](../../../../Geant4Sim/Geant4Code_CntStatResponseStudy/src/SteppingAction.cc) | library-or-historical-tool | SteppingAction |
| [Geant4Sim/Geant4Code_EHE/src/EventAction.cc](../../../../Geant4Sim/Geant4Code_EHE/src/EventAction.cc) | library-or-historical-tool | EventAction |
| [Geant4Sim/Geant4Code_EHE/src/RunAction.cc](../../../../Geant4Sim/Geant4Code_EHE/src/RunAction.cc) | library-or-historical-tool | RunAction |
| [Geant4Sim/Geant4Code_EHE/src/SteppingAction.cc](../../../../Geant4Sim/Geant4Code_EHE/src/SteppingAction.cc) | library-or-historical-tool | SteppingAction |
| [experiments/ELLIPSE500x300_H120/ac225_corrections/transport.py](../../../../experiments/ELLIPSE500x300_H120/ac225_corrections/transport.py) | library-or-historical-tool | Actual water transport/collection; success is bound to every worker and step. |
| [experiments/ELLIPSE500x300_H120/check_nema_transport_progress.py](../../../../experiments/ELLIPSE500x300_H120/check_nema_transport_progress.py) | library-or-historical-tool | Audit finished transport workers while other array tasks are still running. |
| [experiments/ELLIPSE500x300_H120/collect_energy_candidate_v5.py](../../../../experiments/ELLIPSE500x300_H120/collect_energy_candidate_v5.py) | library-or-historical-tool | Hash-checked collection and hard gate; never submits a reconstruction. |
| [experiments/ELLIPSE500x300_H120/derenzo_campaign_transport.py](../../../../experiments/ELLIPSE500x300_H120/derenzo_campaign_transport.py) | library-or-historical-tool | Run immutable accepted binaries and collect every independent CPU worker. |
| [experiments/ELLIPSE500x300_H120/ehe_5e10_recovered_transport.py](../../../../experiments/ELLIPSE500x300_H120/ehe_5e10_recovered_transport.py) | library-or-historical-tool | Strict mixed-allocation acceptance for launcher-only recovery of interrupted workers. The original worker executable/wrapper are reused without modification. |
| [experiments/ELLIPSE500x300_H120/ehe_5e10_transport.py](../../../../experiments/ELLIPSE500x300_H120/ehe_5e10_transport.py) | library-or-historical-tool | Independent 5e10 acquisition using the accepted EHE Geant4 executable. |
| [experiments/ELLIPSE500x300_H120/jscc_5e10_transport.py](../../../../experiments/ELLIPSE500x300_H120/jscc_5e10_transport.py) | library-or-historical-tool | Bounded immutable JSCC Geant4 workers and exact full-worker collection. |
| [experiments/ELLIPSE500x300_H120/maty_collect_calibration.sh](../../../../experiments/ELLIPSE500x300_H120/maty_collect_calibration.sh) | library-or-historical-tool | maty_collect_calibration |
| [experiments/ELLIPSE500x300_H120/maty_collect_imaging.sh](../../../../experiments/ELLIPSE500x300_H120/maty_collect_imaging.sh) | library-or-historical-tool | maty_collect_imaging |
| [experiments/ELLIPSE500x300_H120/maty_collect_xcat.sh](../../../../experiments/ELLIPSE500x300_H120/maty_collect_xcat.sh) | library-or-historical-tool | maty_collect_xcat |
| [experiments/ELLIPSE500x300_H120/maty_nema_collect.sh](../../../../experiments/ELLIPSE500x300_H120/maty_nema_collect.sh) | library-or-historical-tool | maty_nema_collect |
| [experiments/ELLIPSE500x300_H120/opengate_ehe_collect_20261113.py](../../../../experiments/ELLIPSE500x300_H120/opengate_ehe_collect_20261113.py) | library-or-historical-tool | Strict, non-overlapping collection of formal OpenGATE primaries and CDRs. |
| [experiments/ELLIPSE500x300_H120/opengate_ehe_transport_20261113.py](../../../../experiments/ELLIPSE500x300_H120/opengate_ehe_transport_20261113.py) | library-or-historical-tool | Actual OpenGATE transport of the original EHE, with event/crystal readout. |
| [experiments/ELLIPSE500x300_H120/studies/nema5e9_cartesian_he_20261115/transfer_20261117/transport_retries_r3.py](../../../../experiments/ELLIPSE500x300_H120/studies/nema5e9_cartesian_he_20261115/transfer_20261117/transport_retries_r3.py) | study-source-preserved; executed identity in freeze | Retry only recoverable I/O; never retry data-integrity or scientific failures. |
| [experiments/ELLIPSE500x300_H120/studies/opengate_he_derenzo_20261116/collect.py](../../../../experiments/ELLIPSE500x300_H120/studies/opengate_he_derenzo_20261116/collect.py) | study-source-preserved; executed identity in freeze | Collect only new monoenergetic HE imaging, reuse accepted HE response bytes. |
| [experiments/FOV120/collect_point_responses.py](../../../../experiments/FOV120/collect_point_responses.py) | library-or-historical-tool | Verify point workers and export compact detector-count responses (no List duplication). |
| [experiments/FOV120/deploy_collected_qc.py](../../../../experiments/FOV120/deploy_collected_qc.py) | library-or-historical-tool | Deploy a verified FOV120 QC archive to the reconstruction cluster. The reconstruction credential stays in Windows DPAPI storage. The archive is uploaded to a temporary name, checked remotely |
| [experiments/FOV120/maty_collect.sh](../../../../experiments/FOV120/maty_collect.sh) | library-or-historical-tool | maty_collect |
| [experiments/FOV120/maty_collect_qc.sh](../../../../experiments/FOV120/maty_collect_qc.sh) | library-or-historical-tool | maty_collect_qc |
| [experiments/FOV120/package_collected_qc.py](../../../../experiments/FOV120/package_collected_qc.py) | library-or-historical-tool | Package collected FOV120 Uniform/Contrast data with file-level SHA256 checks. |
| [experiments/FOV120/validate_cpu_collectives.py](../../../../experiments/FOV120/validate_cpu_collectives.py) | library-or-historical-tool | Portable two-process GLOO check using FileStore (also works without Windows libuv). |

[算法与流程路线](../../CODE_GUIDE.md)
