# 折叠 Q 回归样本

三个已有真实状态分别覆盖专家终态、HIL 成功标记、HIL 介入失败终态。
NPZ 保存原粒子位置、参考和拓扑，以及验收实验独立计算的五维特征和分数。
测试不加载实验目录，也不现场重新拟合期望值。

来源：`outputs/evaluations/fold_hil_quality_ssl_20260908/surface_response/refinement_20260909/`
下 `material_axis/temporal_features.json` 与 `temporal/windows/`，每类首个窗口的最后状态。
每个 NPZ 的 `episode_id` 保存原轨迹标识。期望分数使用
`surface_response/recommended_model.json` 的已冻结权重计算。

这些是数值一致性回归样本，不是新增测试集或泛化证据。
