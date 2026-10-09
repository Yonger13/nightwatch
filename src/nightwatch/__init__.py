"""
Nightwatch · 带长程记忆的开源项目维护 Agent
============================================

一句话定位
----------
接管 issue 队列的一级分流与 PR 初审：在工具失败、任务循环、记忆冲突的情况下
能可靠降级，并用**维护者的真实动作**作为 ground truth 验证效果。

学习入口
--------
建议按这个顺序读代码（每一步都能独立理解，不依赖后面的模块）：

    1. models.py              数据模型 —— 先搞清楚"系统里有哪些东西"
    2. config.py              所有可调的设计决策集中在这里
    3. memory/lifecycle.py    ★ 核心：记忆的六个环节
    4. memory/store.py        存储与混合检索
    5. context/budget.py      上下文预算调度
    6. loop/reliability.py    四层可靠性
    7. loop/react.py          ★ 主循环：把上面所有东西串起来
    8. tools/registry.py      6 个工具与沙箱
    9. eval/                  评测：怎么证明它有效

配套文档在 docs/ 目录，每个模块都有对应的"为什么这么设计"说明。
"""

__version__ = "0.3.0"
__all__ = ["config", "models"]
