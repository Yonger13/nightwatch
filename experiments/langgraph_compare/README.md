# LangGraph 对照实验

用 LangGraph 重写 Nightwatch 的一条核心链路，与手写版做逐项对照。

**完整文档见 [`docs/08-LangGraph对照实现.md`](../../docs/08-LangGraph对照实现.md)** ——
包含设计思路、实现步骤、关键代码说明、与原方案的对照关系。

---

## 为什么做这个

Nightwatch 的主项目**零第三方依赖**（核心链路全手写）。这个选择有理由，
但有个风险：

> **"我没用框架"和"我不会用框架"从外面看是一样的。**

这个实验的作用是**把"手写 vs 框架"从观点变成实测** ——
两个都做过，并且能讲出差异。

## 怎么跑

```bash
cd "D:/agent program"

# 1. 装依赖（只装到实验环境，主项目不受影响）
pip install langgraph

# 2. 跑对照
python experiments/langgraph_compare/compare.py

# 3. 看导出的流程图（可粘贴到 https://mermaid.live 渲染）
cat experiments/langgraph_compare/graph.mmd
```

## 文件

| 文件 | 作用 |
|---|---|
| `shared.py` | **两版共用的部分** —— 提示词 / 动作解析 / 组件 / 样本 |
| `custom_agent.py` | 手写版（基线） |
| `graph_agent.py` | LangGraph 版 |
| `compare.py` | 对比运行器 |
| `compare_result.json` | 对照结果（运行时生成） |
| `graph.mmd` | 导出的流程图（运行时生成） |

## ★ 这个命令才是关键

```bash
python experiments/langgraph_compare/compare.py
```

它会产出**五类可验证的数据**：

1. **功能等价性** —— 两版的预测是否一致（不一致说明对照不成立）
2. **编排代码量** —— 只算代码行，不算注释
3. **运行时开销** —— 多次重复的中位耗时
4. **框架多给了什么** —— 逐条**实测**，不引用文档
5. **框架藏起了什么** —— 从实写遇到的摩擦点提取

## 依赖隔离

```
src/nightwatch/          ← 零第三方依赖，主项目不受影响
experiments/             ← 这里才用 langgraph
```

`pip install -r requirements.txt` **不会**拉 langgraph。
两个环境互不干扰 —— 因为"想用某个工具做实验"和"想让项目依赖它"是两件事。
