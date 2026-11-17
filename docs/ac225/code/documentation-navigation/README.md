# documentation-navigation源码

职责索引用于打开源码；冻结副本和历史控制器不作为新的执行入口。

| 文件 | 身份 | 说明 |
|---|---|---|
| [experiments/ELLIPSE500x300_H120/catalog_native_spikes_v4.py](../../../../experiments/ELLIPSE500x300_H120/catalog_native_spikes_v4.py) | navigation-only | Read-only 3D native-cell spike catalogue; no interpolation or filtering. |
| [tests/test_project_index.py](../../../../tests/test_project_index.py) | navigation-only | Read-only catalog boundaries and navigation correctness. |
| [tools/baseline/verify_dual_energy_baseline.py](../../../../tools/baseline/verify_dual_energy_baseline.py) | navigation-only | Read-only verification of the accepted 2026-10-07 dual-energy baseline. No SSH, simulation, event selection, training, submission or reconstruction. Optional data/result checks read complete |
| [tools/legacy/ac225_navigation_20261103/ac225_navigation_documents.py](../../../../tools/legacy/ac225_navigation_20261103/ac225_navigation_documents.py) | retired-navigation-renderer | Repository documentation templates for the curated225Ac navigation registry. |
| [tools/legacy/ac225_navigation_20261103/prepare_ac225_registry_20261103.py](../../../../tools/legacy/ac225_navigation_20261103/prepare_ac225_registry_20261103.py) | retired-navigation-renderer | Curate navigation identities from existing evidence; no scientific execution. |
| [tools/legacy/ac225_navigation_20261103/refresh_ac225_navigation.py](../../../../tools/legacy/ac225_navigation_20261103/refresh_ac225_navigation.py) | retired-navigation-renderer | Rebuild local225Ac navigation and pointer folders without importing runners. Only documentation, JSON/CSV indexes and result pointer folders are written. All original source, frozen releases |
| [tools/legacy/ac225_navigation_20261103/verify_ac225_navigation.py](../../../../tools/legacy/ac225_navigation_20261103/verify_ac225_navigation.py) | retired-navigation-renderer | Local headless navigation/filter/links QA; never opens the user's profile. |
| [tools/project_index.py](../../../../tools/project_index.py) | navigation-only | Read-only project/Ac225 directory catalog; never imports experiment runners. |
| [tools/research/ac225_navigation_documents.py](../../../../tools/research/ac225_navigation_documents.py) | navigation-only | Retired template API; dated original is archived under tools/legacy. |
| [tools/research/build_project_catalog.py](../../../../tools/research/build_project_catalog.py) | navigation-only | Compatibility entry: refresh current local navigation; preserve old galleries. |
| [tools/research/build_reconstruction_results_index.py](../../../../tools/research/build_reconstruction_results_index.py) | navigation-only | Compatibility entry: refresh current local navigation; preserve old galleries. |
| [tools/research/prepare_ac225_registry_20261103.py](../../../../tools/research/prepare_ac225_registry_20261103.py) | navigation-only | Compatibility entry for the maintained local navigation tools. |
| [tools/research/refresh_ac225_navigation.py](../../../../tools/research/refresh_ac225_navigation.py) | navigation-only | Compatibility entry for the maintained local navigation tools. |
| [tools/research/verify_ac225_navigation.py](../../../../tools/research/verify_ac225_navigation.py) | navigation-only | Compatibility entry for the maintained local navigation tools. |

[算法与流程路线](../../CODE_GUIDE.md)
