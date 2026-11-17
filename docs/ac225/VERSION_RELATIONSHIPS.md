# 当前入口与保留版本

| 研究 / 类别 | 当前证据入口 | 原位保留及替代关系 |
|---|---|---|
| NEMA Cartesian 5090 | [1822462](studies/nema/nema-cartesian-5090.md)，科学95022158ffedcdd7 | 原矩阵修复009433dd7e2b3b50与传输r1–r6均保留；旧“未提交”已被formal_registration/Slurm更新，不能改冻结字节 |
| Cartesian 4090 | [有效r2 1822534](studies/nema/nema-cartesian-4090-r2.md)，执行a502c70af71dea0e | r2正式启动后失败；相邻无r2目录仅为未提交预检失败，两者都保留，不自动补交 |
| Derenzo5e10联合 | [1771220失败9950预览](studies/derenzo/derenzo-jscc-5e10-440-joint.md) | 1761195旧64卡、1769020及ec1677ba41cce0f9续算记录均保留；旧“排队/运行”被11月12日失败预览证据更新 |
| Derenzo5e10 EHE | [1769915两能量完整400帧](studies/derenzo/derenzo-ehe-dense-5e10.md) | 原传输中断与旧“GPU未提交”属于历史，完整交付不删除partial与失败日志 |
| EHE矩阵 | [65536点/面](studies/ehe/ehe-dense-matrix.md) | 原256点结果及冻结C保留，物理HOLD没有因自身S或加密而解除 |
| OpenGATE EHE NEMA | [有效源旋转修复](studies/nema/opengate-ehe-5e9.md) | 首轮源旋转失效结果保留，不能把其图像与修复后的真实投影混作同一版本 |
| HE NEMA / Derenzo | [HE NEMA](studies/nema/opengate-he-nema.md)、[HE Derenzo](studies/derenzo/opengate-he-derenzo.md) | Derenzo复用46d438089710a6ce响应，新源34a52489b6362ba0，不覆盖EHE源/矩阵/图集 |
| 带水SC | [首轮32路](studies/nema/water-sc-first.md)、[供体v2](studies/nema/water-scatter-v2.md) | r5有效路由更新原submitted；14路预览是子集，完整交付6400帧仍不自动升级全组视觉QA；v2独立模型不覆盖首轮 |
| Compton展宽 | [1814153三档完成](studies/nema/nema-width-sweep.md) | 1813383失败/9950帧和历史差异标志保留；100/80/60%各自身S，不把80%直接升级生产物理模型 |
| 旧511/440、FOV120、本地复现 | [历史复现](../../Reproduction/README.md)、[FOV120](../../experiments/FOV120/README.md) | 根目录通用入口、distributed和旧Factors/CntStat/List接口留原路径；未经证据不能合并到当前225Ac条件 |
| 导航维护 | [tools/navigation](../../tools/navigation/README.md) | 四份20261103工具按SHA归档，旧命令兼容；旧AGENTS写入API已关闭，人工授权与撤销记录原字节保留 |

源码精确重复组见[duplicate_source_groups.json](duplicate_source_groups.json)，Python引用见[source_dependencies.csv](source_dependencies.csv)。二者是维护证据，不表示科学等价、可删除或可移动。冻结payload、图集scripts和同名工作区源码各有独立身份。
