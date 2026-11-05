# 本次参数确认记录

参数版本：`VDIE-FULL-v1.0-review-1`。

用户在收到参数审阅文件后回复：**“确认通过”**。确认日期：2026-11-05（Asia/Shanghai）。执行许可保存在 `configs/review_options.yaml`，正式运行入口检查该许可。

批准范围包括全部 211 项原规范参数、40 项补充执行约定、37 项派生量、4,020 条实验记录，以及向量能耗倍率 2/4/8、广播全体监听的物理半双工解释、Prefill 复用率无效应对照。

`parameter_review.docx`、`parameter_review.md` 和其 CSV 为审批时提交的原稿，因此原稿中的“待审批”字样保留。实际执行状态与参数见 `configs/review_options.yaml` 和 `results/parameters.csv`。

FP8 NRMSE 门槛始终为 0.05。用户的参数确认不构成对后续数值失败的豁免；时序实验是否完成与所有验收是否通过分别在最终 manifest 中记录。
