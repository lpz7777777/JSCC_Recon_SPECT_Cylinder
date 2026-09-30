# 独立点源与 Factors 图样核查（未完成的诊断）

此项用于诊断椭圆热柱欠恢复，**不属于 NEMA 体模输入或校准**。maty 独立点源 `15502273` 共 10 个位置/能量数据集、各 20 视角，合计 1e8 初级 γ；实际效率及哈希见 [`selected_point_efficiency.json`](selected_point_efficiency.json)。点源的 CntStat 与 scxi717 上已校准的 218/440 Factors 对比，最初使用最近的单个极坐标采样点，报告与图见 [`selected_point_factor_alignment.json`](selected_point_factor_alignment.json) 和 [`selected_point_factor_alignment.png`](selected_point_factor_alignment.png)。

进一步用最近 **8** 个三维极坐标点按距离平方倒数插值，脚本分别为 `upload_point_alignment_inputs.py`、`point_factor_interpolation_remote.py`、`fetch_point_alignment_report.py`，结果为 [`selected_point_factor_interpolation.json`](selected_point_factor_interpolation.json)。在选出的 10 个点上，插值对余弦相似度改变量通常仅约 0.001–0.004；四层总计数分数的差异已另列于报告。这样排除了“仅取最近一个网格点”作为低原始余弦值的主要解释，但**不能直接证明系统矩阵模型失配**：每点仅 1e7 初级 γ，探测器直方图稀疏，Poisson 噪声本身会降低原始图样余弦值。下一步应按各个 Factors 预测均值生成 Poisson 零假设分布，与实测余弦和分层残差比较；再进行点源定位/分辨率分析。

输入 bundle 位于受 Git 忽略的 `generated/PointQA/selected_point_counts.zip`，其 SHA-256、逐点 CntStat 哈希和 Factors manifest 哈希写在结果 JSON。脚本复用 `docs/REMOTE_COMPUTE_ACCESS.md` 中的本机安全 SSH 凭据，不写入密码或私钥。不要将此探索性报告作为 NEMA 生产矩阵的重新校准依据。
