# 对比方法的洪水深度估计文献来源

核查日期：2026-10-21。覆盖 `compare/common/comparison_factory.py` 与 `compare/common/traditional_registry.py` 注册的全部 **8 个对比方法**。

本文件列出实际将对应方法用于洪水深度估计的已发表论文，而非网络最初用于医学分割、图像分类或道路分割的架构论文。每项同时说明原文证据与当前代码的关系：**任务文献依据不等于原作者代码移植记录，也不等于完整复现原文实验**。对于仓库只提供架构实现、未记录具体论文移植过程的方法，以下文献作为可核验的洪水深度应用依据，不追溯性地声称代码最初来自该论文。

## 1. 方法与来源对应表

| 当前对比方法 | 实现文件 | 洪水深度应用论文 | 原文可核查内容 |
|---|---|---|---|
| FwDET v2 | [`fwdet_v2.py`](traditional/fwdet_v2.py) | **[R1] Cohen et al., 2019**：[DOI](https://doi.org/10.5194/nhess-19-2053-2019) | FwDET v2.0 从淹没边界高程重建水面并估计水深。 |
| TSA | [`tsa.py`](traditional/tsa.py) | **[R2] Chimata et al., 2025**：[DOI](https://doi.org/10.5194/nhess-25-2455-2025) | 第 4.4–4.5 节、Fig. 5：用边界高程拟合一阶趋势水面，再减 DEM。 |
| FlDepth | [`fldepth.py`](traditional/fldepth.py) | **[R3] Akkimi et al., 2026**：[DOI](https://doi.org/10.1007/s11269-025-04405-1) | 出版社摘要：沿淹没区中心线布设横断面，利用边界高程构建水面。 |
| U-Net | [`unet_depth_regression.py`](deep_learning/unet_depth_regression.py) | **[R4] Löwe et al., 2021（U-FLOOD）**：[DOI](https://doi.org/10.1016/j.jhydrol.2021.126898) | 将 U-Net 改为二维最大洪水深度预测网络；目标为连续水深。 |
| Attention U-Net | [`dlsim_attention_unet.py`](deep_learning/dlsim_attention_unet.py) | **[R5] Yokoya et al., 2022（DLSIM）**：[DOI](https://doi.org/10.1109/TGRS.2020.3035469) | 回归模型小节、网络结构图及水深实验表明确包含 Attention U-Net。 |
| LinkNet | [`dlsim_linknet.py`](deep_learning/dlsim_linknet.py) | **[R5] Yokoya et al., 2022（DLSIM）**：[DOI](https://doi.org/10.1109/TGRS.2020.3035469) | 与 Attention U-Net 分别训练和评估，包含洪水最大水深回归结果。 |
| U-Net++ | [`unetplusplus_depth_regression.py`](deep_learning/unetplusplus_depth_regression.py) | **[R6] Blay et al., 2026（Inundation2Depth）**：[DOI](https://doi.org/10.1016/j.dib.2025.112347) | 第 4.5 节及 Table 7：以 RMSE 为损失，实际训练 U-Net++ 预测水深。 |
| ResNet18 | [`resnet18_depth_regression.py`](deep_learning/resnet18_depth_regression.py) | **[R7] Blay & Hashemi-Beni, 2026**：[DOI](https://doi.org/10.3390/rs18010060) | 第 2.4.1、2.6、3.1–3.2 节：ResNet-18 连续水深回归及其评估。 |

## 2. 逐项核查与本地适配说明

### 2.1 FwDET v2

- **洪水深度证据**：R1 是 FwDET v2.0 的方法论文，直接研究由遥感淹没范围和地形数据估计连续水深；不只是洪水范围识别论文。可查阅[期刊全文](https://nhess.copernicus.org/articles/19/2053/2019/)。
- **当前实现**：以淹没区外侧边界高程作水面样本，通过欧氏距离最近邻分配得到水面，计算非负水深。
- **适配边界**：原文的 ArcGIS 实现使用代价分配处理永久水体等情况；当前代码没有该完整过程，也没有复现原文的全部后处理。应表述为 **FwDET v2 的本地 DSM 适配版**，不能把欧氏分配宣称为所有地形与代价条件下的严格等价实现。

### 2.2 TSA

- **洪水深度证据**：R2 第 4.4 节说明用淹没外边界高程作样本，以最小二乘法拟合一阶水面 `W(x, y) = ax + by + c`；第 4.5 节由水面减去 DEM 计算水深。[原文及 Fig. 5](https://nhess.copernicus.org/articles/25/2455/2025/)。
- **当前实现**：对每个八连通淹没分量分别拟合一阶趋势水面；边界样本不足时回退至最近边界高程分配。
- **适配边界**：当前只移植水深重建部分，使用项目提供的范围和 DSM，没有复现原文的 EOS-04 洪水提取与 HAND 筛选工作流。按连通分量拟合及回退规则为本地实现选择。

### 2.3 FlDepth

- **洪水深度证据**：R3 的[出版社摘要](https://link.springer.com/article/10.1007/s11269-025-04405-1)明确说明，从洪水范围中心线横断面及边界地形高程重建水面，以估计河流洪水和降雨积水深度。
- **当前实现**：用距离变换的局部脊线近似中心线，在脊线点附近选取边界高程的中位数，再以最近脊线样本传播水面。
- **适配边界**：这是 **FlDepth 思路的简化本地适配**；局部脊线、径向边界采样和最近样本传播不构成原文横断面生成及插值流程的完整复现。此次证据核查依据出版社摘要，不声称已逐步核验付费全文中的全部算法细节。

### 2.4 U-Net

- **洪水深度证据**：R4（U-FLOOD）明确将 U-Net 用于二维最大洪水深度图预测，输入包含降雨过程和地形、地表变量，研究未参与训练的区域与降雨事件。[论文页面](https://www.sciencedirect.com/science/article/pii/S0022169421009483)；[作者机构提供的全文](https://mediatum.ub.tum.de/doc/1638826/1638826.pdf)。
- **当前实现**：常规 U-Net 编码器—解码器，使用拼接式跳跃连接和本地范围约束水深输出头。
- **适配边界**：当前使用本项目的 SAR、地形和已知范围输入，未实现 U-FLOOD 的降雨输入及原文完整设置。R4 是采用 U-Net 进行洪水深度回归的任务依据，不能据此将当前模型直接命名为完整 U-FLOOD 复现。
- **另一项直接应用证据**：R6 第 4.5 节及 Table 7 也实际评估了 U-Net 的洪水深度回归，可作为范围与地形输入路线的补充引用。

### 2.5 Attention U-Net

- **洪水深度证据**：R5 研究模拟数据与遥感观测结合的洪水深度反演，明确使用 Attention U-Net 回归洪水最大水深。作者[公开预印本全文](https://arxiv.org/pdf/2006.05180)的 CNN-based Regression 小节、Fig. 4(a) 及 Table I 可核查网络和水深结果；[作者代码](https://github.com/nyokoya/dlsim)提供水深任务入口。
- **当前实现**：在 U-Net 跳跃连接上加入注意力门，采用项目统一输入及范围约束回归头。
- **适配边界**：不包含 DLSIM 的原始模拟数据生成过程、原始二值变化输入与完整训练配置，也未与 LinkNet 融合输出。原文是该架构用于洪水深度估计的直接来源，不应以胰腺分割领域的 Attention U-Net 原始论文替代。

### 2.6 LinkNet

- **洪水深度证据**：与 Attention U-Net 共用 R5；公开全文 Fig. 4(b) 与 Table I 分别列出 LinkNet 结构及最大水深任务结果，故可以用同一篇洪水深度论文支持这两个独立对比方法。
- **当前实现**：残差编码块、相加式跳跃连接及本地水深输出头。
- **适配边界**：DLSIM 公开全文说明，其 LinkNet 变体没有沿用原始 LinkNet 的残差编码块；当前代码使用 `ResidualBlock`，因此应明确为 **受 DLSIM 启发的 LinkNet 本地适配版**，不能宣称与 DLSIM 原文网络逐层一致。

### 2.7 U-Net++

- **洪水深度证据**：R6 虽为数据论文，但第 4.5 节确实训练并比较了 FCN、U-Net、U-Net++ 和 Attention U-Net 的**水深回归**，采用 RMSE 损失并在 Table 7 报告测试误差；并非只在数据生成阶段使用分割网络。[开放全文及实验表](https://pmc.ncbi.nlm.nih.gov/articles/PMC12765332/)。
- **当前实现**：四级嵌套 U-Net++ 解码器和密集跳跃连接，使用本项目统一输入与范围约束输出头。
- **适配边界**：R6 证明 U-Net++ 在洪水深度任务中的实际应用，但没有据此确认当前层宽、输入通道、损失或训练配置与原文相同；本文献是任务应用依据，不是已核验的代码移植来源。引用时应保留“本地适配”的表述。

### 2.8 ResNet18

- **洪水深度证据**：R7 第 2.4.1 节明确将 ResNet-18/34/50 改为连续水深回归；第 2.6 节描述范围、DTM 及地形派生量的输入实验，第 3.1–3.2 节提供水深预测结果。[期刊全文](https://www.mdpi.com/2072-4292/18/1/60)。
- **当前实现**：随机初始化的 Torchvision ResNet18 编码器，增加逐级上采样解码器，恢复全分辨率水深；代码明确使用 `weights=None`。
- **适配边界**：论文将分类输出改为水深回归的描述，不足以证明当前解码器与其结构相同。R7 可支持 ResNet18 的洪水深度应用来源；本项目的解码器、输入与训练协议仍是本地适配。

## 3. 正式参考文献

以下按正式期刊卷期年份列出；DOI 中的年份、在线发表年份与卷期年份可能不同。

**[R1]** Cohen, S., et al. (2019). **The Floodwater Depth Estimation Tool (FwDET v2.0) for improved remote sensing analysis of coastal flooding.** *Natural Hazards and Earth System Sciences*, **19**, 2053–2065. [DOI：10.5194/nhess-19-2053-2019](https://doi.org/10.5194/nhess-19-2053-2019).

**[R2]** Chimata, L. A., Anuvala Setty Venkata, S. B., Patlolla, S. V. R., Korada Hari Venkata, D. R., Kandrika, S., & Chauhan, P. (2025). **Automated rapid estimation of flood depth using a digital elevation model and Earth Observation Satellite (EOS-04)-derived flood inundation.** *Natural Hazards and Earth System Sciences*, **25**, 2455–2472. [DOI：10.5194/nhess-25-2455-2025](https://doi.org/10.5194/nhess-25-2455-2025).

**[R3]** Akkimi, A., Jella, S., Peesapati, R. M., Singh, A., KHV, D. R., & Chauhan, P. (2026). **FlDepth: A New Method for Estimating Fluvial and Pluvial Flood Depths from Near Real-Time Satellite-Derived Inundation Map and Topography.** *Water Resources Management*, **40**, Article 128. [DOI：10.1007/s11269-025-04405-1](https://doi.org/10.1007/s11269-025-04405-1). 正式发表日期为 2026-02-16，不应仅依据 DOI 写成 2025 年论文。

**[R4]** Löwe, R., Böhm, J., Jensen, D. G., Leandro, J., & Rasmussen, S. H. (2021). **U-FLOOD – Topographic deep learning for predicting urban pluvial flood water depth.** *Journal of Hydrology*, **603**, 126898. [DOI：10.1016/j.jhydrol.2021.126898](https://doi.org/10.1016/j.jhydrol.2021.126898).

**[R5]** Yokoya, N., Yamanoi, K., He, W., Baier, G., Adriano, B., Miura, H., & Oishi, S. (2022). **Breaking Limits of Remote Sensing by Deep Learning From Simulated Data for Flood and Debris-Flow Mapping.** *IEEE Transactions on Geoscience and Remote Sensing*, **60**, 1–15. [DOI：10.1109/TGRS.2020.3035469](https://doi.org/10.1109/TGRS.2020.3035469). 在线发表于 2020-11-16；此处采用正式卷期年份 2022。作者 2020 年预印本及代码 README 使用较早标题，引用期刊版时采用本条正式标题。

**[R6]** Blay, J., Gebregziabher, Y., Jha, M. K., & Hashemi Beni, L. (2026). **Inundation2Depth: A multi-source dataset for floodwater depth estimation in urban areas.** *Data in Brief*, **64**, 112347. [DOI：10.1016/j.dib.2025.112347](https://doi.org/10.1016/j.dib.2025.112347). 在线发表于 2025-12-01，卷期为 2026 年 2 月；[PubMed 记录](https://pubmed.ncbi.nlm.nih.gov/41492553/)同时列出在线日期与卷期日期。

**[R7]** Blay, J., & Hashemi-Beni, L. (2026). **Geospatial and Deep Learning Approaches for Modeling Floodwater Depth in Urbanized Areas.** *Remote Sensing*, **18**(1), 60. [DOI：10.3390/rs18010060](https://doi.org/10.3390/rs18010060). 在线发表于 2025-12-24，正式卷期年份为 2026。

## 4. 引用口径

- U-Net、Attention U-Net、LinkNet、U-Net++、ResNet18 的洪水深度应用来源，分别使用 R4、R5、R5、R6、R7；不能仅用对应通用架构的首发论文替代这些任务文献。
- 当前所有对比模型均按本项目的数据和实验协议适配；本文献清单不改变任何实现、配置或已有结果。对外描述应区分“参考某论文的任务方法”和“完整复现某论文”。
- 当前协议将 `valid_depth_mask` 作为已知洪水范围输入。应将其结果表述为已知范围条件下的水深估计，不能凭这些文献将其描述为包含独立范围提取的端到端结果。
