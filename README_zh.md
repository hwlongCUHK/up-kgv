# UP-KGV：不确定性优先的知识图谱验证系统

面向大规模自动抽取生物医学知识图谱的人机协同错误审核流水线。

## 概述

UP-KGV 解决的核心问题是：给定一个包含数百万自动抽取三元组的知识图谱，以及有限的人工审核预算 *K*，哪些三元组应当优先送交领域专家检查？

核心洞见在于**模式—语义分歧具有信息价值**。当一条三元组在结构上异常、但语义上合理（或反之），它更可能是真实错误。UP-KGV 利用这一信号构建有序审核队列，无需来自目标知识图谱的标注训练数据。

在 CPubMed-KGv2\_0（458 万条三元组）上评估，UP-KGV 达到 **AUPRC = 0.782**，在审核预算 *K* = 100 时比随机抽查多发现 **2.7 倍**的真实错误。

## 流水线结构

```
知识图谱（458 万条三元组）
    │
    ▼ 阶段一 — 确定性检测器
    │   自环过滤器、模式频率分析器、基数异常分析器、
    │   拓扑评分器、实体类型冲突检测器
    │   → 约 12,000 条 Machine_Suspected 三元组
    │
    ▼ 阶段二 — 结构异常评分
    │   为每条三元组计算综合异常分 C(x)
    │   低分三元组 → AUTO_CLEAN（跳过 LLM 审核）
    │
    ▼ 阶段三 — 证据包构建
    │   每条三元组包含：实体类型、模式统计、邻域摘要、
    │   矛盾对、对称边缺失、拓扑特征
    │
    ▼ 阶段四 — 专家 LLM 路由
    │   语义可信度集成（两轮：模式感知 + 纯语义）
    │   关系标签审核器（模式极端异常三元组）
    │   实体链接与类型检查器（边界模式 + 类型冲突）
    │   矛盾仲裁器（存在矛盾对的三元组）
    │   上下文连贯性验证器（不确定的枢纽实体）
    │
    ▼ 阶段五 — 优先级评分
    │   S(x) = 0.55·A(x) + 0.35·C(x) + 0.10·R(x) + λ·1[V(x)=UNCERTAIN]·C(x)
    │   输出 Top-K 审核队列供人工审核
    │
    ▼ 阶段六 — 补丁写回（人工决策后）
        已批准的修复写回知识图谱
```

## 安装

```bash
pip install -e .
# 或使用 uv：
uv sync
```

需在项目根目录创建 `.env` 文件并填写 API 密钥：

```env
DEEPSEEK_API_KEY=你的密钥
# 可选：为金标准标注使用独立模型族
OPENAI_API_KEY=你的密钥
```

## 使用方法

```python
from kg_verify.orchestrator import KGVerifyHITLOrchestrator

pipeline = KGVerifyHITLOrchestrator()
pipeline.run()
# 输出 review_items.tsv 至 outputs/ 目录，供人工审核
```

**启用两轮集成模式**（生产环境推荐）：

```bash
ENSEMBLE_AGENT_A=1 python -m kg_verify.orchestrator
```

## 配置说明

所有阈值、权重和模型设置均集中于 `config.py`，主要参数如下：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `AUTO_CLEAN_SCORE` | 0.15 | C(x) 低于此值时跳过 LLM 审核 |
| `SCHEMA_THRESHOLD_EXTREME` | 0.005 | 模式频率低于此值时路由至关系标签审核器 |
| `HUB_DEGREE_THRESHOLD` | 5000 | 节点度高于此值时调用上下文连贯性验证器 |
| `USE_ENSEMBLE_AGENT_A` | False | 是否启用两轮模式—语义集成 |
| `PRIORITY_WEIGHTS` | 见 config | S(x) 各项权重 |

## 目录结构

```
kg_verify/
├── agents/          # LLM 专家智能体（5 个专家）
├── audit/           # 审计日志
├── patches/         # 知识图谱补丁写回（审核后修复）
├── pipeline/        # 统计检测器与证据包构建
├── review/          # 审核队列构建与导出
├── tools/           # KG 索引与模式统计工具
├── config.py        # 全局配置
└── orchestrator.py  # 6 阶段流水线入口
```

## 许可证

MIT
