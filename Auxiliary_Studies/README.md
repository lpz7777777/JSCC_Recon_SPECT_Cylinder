# 系统矩阵与辅助研究

| 目录 | 实际职责与地位 |
|---|---|
| [CUDA矩阵工程](GPU-Based-System-Matrix-Calculation-for-SPECT-PET-main/ReadMe.md) | 探头/准直器响应、PE与散射；正式版本由各研究freeze和二进制SHA绑定 |
| [Factors转换](GPU-Based-System-Matrix-Calculation-for-SPECT-PET-main/GenFactors/README.md) | 原始响应→柱坐标/密度Factors，必须匹配体积和网格 |
| [Compton sensitivity](Sensitivity_SPECT_PolarCoor/README.md) | 独立校准事件、灵敏度与验证；不能用成像观测Cᵀ1替代 |
| [Compton原型](ComptonSystemMatrixPrototype/README.md) | 独立原型，不能自动替代生产v5 |
| [事件顺序](EventOrderInference_Experiment/README.md) | 事件顺序推断实验，独立方法/输入与输出 |
| [CRCVAR](CRCVAR_SinglePhoton/README.md) | 单光子定量/方差研究 |
| [FreePath](FreePath)、[LinePofile](LinePofile)、[Reference](Reference) | 历史路径、剖面与参考程序；拼写保留兼容，不据目录外观合并 |

当前[完整数据流](../docs/ac225/WORKFLOW.md)、[源码分类](../docs/ac225/SOURCE_INDEX.md)、[各研究](../docs/ac225/EXPERIMENTS.md)连接实际代码与证据。运行器、矩阵服务器部署、成功响应与冻结路径均未移动。相似版本和重复实现只登记关系，不擅自替换科学模型。
