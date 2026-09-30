# 中心放置的 NEMA-like Body Phantom（几何预览）

本目录只包含体模设计和真值预览。**尚未运行 Geant4，也没有进行任何成像或重建。** 后续若用于成像，源与重建支持域仍须采用 `ELLIPSE500x300_H120` 的物理椭圆柱 FOV：`(x/250)^2+(y/150)^2≤1，|z|≤60 mm`，不能将长方形真值画布当作有效源范围。

![体模在椭圆 FOV 内的轴位布局](layout_in_ellipse.png)

![三向真值切片](activity_slices.png)

![三维几何示意](geometry_3d.png)

## 尺寸与来源

- 主腔横向约 290 mm、前后约 220 mm；肺部插件外径 51 mm；六个球的内径依次为 10、13、17、22、28、37 mm。这些尺寸参照 [Data Spectrum NEMA IEC PET Body Phantom 数据表](https://www.spect.com/pdf/NEMA-IEC-PET-Body-Phantom.pdf)，并对应用户提供的示意图。
- 球心位于中心半径 57 mm 的圆上，按示意图顺序分别位于 0°、60°、120°、180°、240°、300°；主体下缘轮廓由配置中的控制点作单调样条插值。这两项是**按图近似**，并非厂家 CAD 尺寸。
- 主体和中心肺部插件均以 `(0,0,0) mm` 为中心。主体 z 范围 `[-30,+30] mm`，置于 120 mm 高 FOV 的正中央。此 **60 mm 缩短版不符合 NEMA NU 2 标准体模至少 180 mm 的内部轴向长度**，仅用于本项目的研究比较；标准参见 [NEMA NU 2-2007 §7.3.3](https://psec.uchicago.edu/library/applications/PET/chien_min_NEMA_NU2_2007.pdf)。
- 预览活度：主体背景 1；10/13/17/22 mm 球为 4；28/37 mm 球及中心插件为 0。该活度比只是直观显示和后续方案候选，未生成 218/440 keV 初级光子。无 PMMA 壁、球壳或组织衰减模型。

## 可复现数据与检查

从本实验目录运行 `python make_nema_body_h60.py`。配置为 `nema_body_h60_config.json`；体素真值输出到忽略 Git 的 `generated/NEMA_Body_H60/truth_3mm.npz`，内容包括 x/y/z 坐标、主体/肺部/六球体积占比及相对活度。全 FOV 画布为 `[z,y,x]=[40,100,168]`，间距 3 mm；主体恰占中心 20 层。球边界以每轴 8 个子体素积分，六球数值体积与解析球体积的最大偏差约 0.23%。

`manifest.json` 记录尺寸、球心、积分体积、状态，以及配置、预览图和体素真值的 SHA-256。体素真值文件较大，故不纳入 Git；重新生成后可对照 manifest 哈希。当前文件仅为几何/源分布准备，不代表生产模拟已验证。
