"""
Nightwatch · 评测集构造

================================================================================
这个文件解决的是一道必问题："你的评测集怎么造的？会不会过拟合自己的实现？"
================================================================================
三道防线：

    ① **分层采样**。先枚举问题类型，每类保证最少样本数。
       为什么？因为标签分布是长尾的 —— 如果随机采样，
       稀有类型可能只有 1-2 条，你根本测不出它在稀有类型上的表现。

    ② **留出集**。先划出 30% 完全不参与任何调试，最后只用它报数。
       这是防止"调参调到评测集上"的标准做法。

    ③ **数据来自真实来源且脚本化**。评测集由脚本生成、进版本控制，
       别人能复现。而不是手工挑几十条"看起来合适"的样本。

================================================================================
本项目的评测集有两个来源
================================================================================
    A. **真实来源（首选）**：某个开源项目的历史 issue
       —— 维护者打的标签、标记的重复关系、关闭方式，就是天然的 ground truth。
          你不需要自己标注，这是这个场景最大的优势。

    B. **合成数据（演示/兜底）**：人工构造的分层样本
       —— 用于在没有网络/没有选定项目时能跑通全流程。

**必须诚实区分这两者。** 报告里要写清楚指标是在哪个数据集上得的 ——
用合成数据跑出来的漂亮数字没有说服力，面试时必须说明。
================================================================================
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from ..models import Issue


# =============================================================================
# 分层定义
# =============================================================================
# 这些分层不是随便分的，它们对应"问题类型"这个业务概念。
# 分层的目的是**保证每一类都被测到**，而不是被常见类型淹没。

STRATA = {
    "timeout":    "超时、限流、性能问题",
    "permission": "权限、鉴权、访问控制",
    "config":     "配置错误、环境问题",
    "dependency": "依赖冲突、安装失败",
    "crash":      "崩溃、异常退出",
    "duplicate":  "重复问题（用于测判重能力）",
    "feature":    "功能请求（非 bug，需要区分出来）",
    "unclear":    "信息不足，应转人工",
}


@dataclass
class EvalDataset:
    """评测集。

    包含 train（可用来调试）和 holdout（只用于最终报数）两部分。
    """
    train: list[Issue] = field(default_factory=list)
    holdout: list[Issue] = field(default_factory=list)
    version: str = "v1"
    source: str = ""                 # ★ 必须记录来源（真实 / 合成）
    notes: str = ""

    def all_issues(self) -> list[Issue]:
        return self.train + self.holdout

    def summary(self) -> dict:
        def strata_count(issues):
            c: dict[str, int] = {}
            for i in issues:
                key = (i.gold_labels[0] if i.gold_labels else "unlabeled")
                c[key] = c.get(key, 0) + 1
            return dict(sorted(c.items(), key=lambda x: -x[1]))

        return {
            "版本": self.version,
            "来源": self.source,
            "train规模": len(self.train),
            "holdout规模": len(self.holdout),
            "train分层分布": strata_count(self.train),
            "holdout分层分布": strata_count(self.holdout),
            "来源说明": self.notes,
        }

    def to_json(self, path: str | Path) -> None:
        from ..models import to_dict
        Path(path).write_text(json.dumps({
            "version": self.version,
            "source": self.source,
            "notes": self.notes,
            "train": [to_dict(i) for i in self.train],
            "holdout": [to_dict(i) for i in self.holdout],
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "EvalDataset":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        def mk(items):
            out = []
            for d in items:
                from ..models import IssueState
                out.append(Issue(
                    issue_id=d["issue_id"], title=d["title"], body=d["body"],
                    author=d["author"], repo=d.get("repo", ""),
                    labels=d.get("labels", []), stack_trace=d.get("stack_trace", ""),
                    state=IssueState(d.get("state", "open")),
                    gold_labels=d.get("gold_labels", []),
                    gold_is_duplicate_of=d.get("gold_is_duplicate_of"),
                    gold_action=d.get("gold_action", ""),
                ))
            return out
        return cls(train=mk(raw.get("train", [])), holdout=mk(raw.get("holdout", [])),
                   version=raw.get("version", "v1"), source=raw.get("source", ""),
                   notes=raw.get("notes", ""))


# =============================================================================
# 构造器
# =============================================================================

class DatasetBuilder:
    """从 issue 列表构造分层评测集。"""

    def __init__(self, seed: int = 42, holdout_ratio: float = 0.3,
                 min_per_stratum: int = 3):
        self.seed = seed
        self.holdout_ratio = holdout_ratio
        self.min_per_stratum = min_per_stratum
        random.seed(seed)

    def build(self, issues: Iterable[Issue], source: str = "synthetic",
              version: str = "v1", notes: str = "") -> EvalDataset:
        """分层采样 + 划分 holdout。"""
        # --- 第一步：按层归类 ---
        buckets: dict[str, list[Issue]] = {}
        for issue in issues:
            key = self._stratum_of(issue)
            buckets.setdefault(key, []).append(issue)

        train, holdout = [], []
        for key, items in buckets.items():
            # 层内先打乱（用固定 seed 保证可复现）
            random.shuffle(items)
            n = len(items)
            if n < self.min_per_stratum:
                # 样本不足的层：**全部放进 train**，并在总结里标注。
                # 为什么不做数据增强凑数？因为凑出来的样本会污染指标，
                # 让「稀有类型表现差」这个真实问题被掩盖。
                train.extend(items)
                continue

            n_holdout = max(1, int(n * self.holdout_ratio))
            holdout.extend(items[:n_holdout])
            train.extend(items[n_holdout:])

        random.shuffle(train)
        random.shuffle(holdout)

        return EvalDataset(
            train=train, holdout=holdout, version=version, source=source,
            notes=notes or (
                f"分层采样，seed={self.seed}，holdout 比例={self.holdout_ratio}。"
                "holdout 集在调试期间不得使用，只用于最终报数。"
            ),
        )

    @staticmethod
    def _stratum_of(issue: Issue) -> str:
        """判断一个 issue 属于哪一层。

        ============================================================
        ★ 分层的依据必须来自**数据本身**，不能来自对某个数据集的假设
        ============================================================
        原实现硬编码了一套 STRATA（timeout / permission / config / ...），
        那是我**为合成数据设计的**。换到真实数据（pandas）后，
        所有样本都不匹配那套关键词，于是**全部落到 unclear 层** ——
        分层报告看起来"分层了"，实际上只有一层，毫无意义。

        这是个很隐蔽的 bug：**代码没报错、输出格式也对、只有内容是错的。**

        修法：优先用维护者的标签作为层名（那是权威分类），
        因为 `gold_labels` 在上游已经收敛过口径（见 prepare_dataset.py）。

        这个改动同时兼容合成数据 —— 因为合成数据的 gold_labels
        本来就是 ["timeout"] / ["permission"] 这类，直接当层名正好。

        教训：**别把"某个数据集的特性"写进通用逻辑里。**
        如果确实需要项目相关的配置，应该做成参数，而不是写死。
        """
        if issue.gold_is_duplicate_of:
            return "duplicate"
        if issue.gold_labels:
            # 用第一个标签作为层名 —— gold_labels 已收敛为"问题类型"层
            return str(issue.gold_labels[0])
        # 没有标签 → 独立一层（这些样本的正确答案是"转人工"）
        return "unlabeled"


# =============================================================================
# 合成数据集（演示 + 兜底）
# =============================================================================

def build_synthetic_dataset() -> EvalDataset:
    """构造合成评测集。

    **这只是一个让流程能跑起来的占位实现。**

    它的局限必须说清楚：
      - 症状描述是人写的，比真实 issue 规整，模型表现会虚高
      - 重复关系是人为指定的，比真实场景容易识别
      - 没有真实作者的历史行为，记忆效果测不准

    所以：**任何在这个数据集上的指标都不能写进简历。**
    要写进简历的数字必须来自真实项目的 issue 队列（见 docs/01）。
    """
    raw = [
        # --- timeout 层 ---
        ("#101", "接口在高峰时段频繁超时",
         "我们的服务在每天上午 10 点左右调用报表接口时经常超时，返回 504。"
         "持续了大约一周，非高峰时段正常。",
         "alice", ["timeout"], [],
         "Traceback (most recent call last):\n  File \"api/report.py\", line 88\n    resp = client.get(url, timeout=30)\nrequests.exceptions.Timeout: HTTPSConnectionPool read timed out",
         None, "打标签 bug/timeout，建议提升配额"),

        ("#102", "批量导出大数据量时响应超时",
         "导出超过 5000 条记录时接口超时，1000 条以下是正常的。",
         "bob", ["timeout", "performance"], [],
         "requests.exceptions.ReadTimeout: Read timed out. (read timeout=30)",
         None, "打标签 bug/timeout，建议分批导出"),

        ("#103", "定时任务偶发执行超时",
         "每天的定时同步任务偶尔会超时失败，不是每次都出现，大概三天一次。",
         "carol", ["timeout"], [],
         "TimeoutError: task exceeded 300s limit",
         None, "打标签 bug/timeout"),

        # --- permission 层 ---
        ("#104", "调用接口返回 403 权限不足",
         "按照文档配置了 token，但调用用户列表接口一直返回 403。"
         "token 是今天新生成的。",
         "dave", ["permission"], [],
         "HTTPError: 403 Client Error: Forbidden for url: /api/v1/users",
         None, "打标签 bug/permission，需要检查 scope"),

        ("#105", "新用户无法访问仪表盘",
         "创建的新账号登录后看不到仪表盘，提示无权限访问。"
         "老账号正常。",
         "erin", ["permission"], [],
         "PermissionError: user lacks scope dashboard:read",
         None, "打标签 bug/permission"),

        # --- config 层 ---
        ("#106", "配置文件里的环境变量不生效",
         ".env 文件里设置了 API_BASE_URL，但程序读到的还是默认值。",
         "frank", ["config"], [],
         "KeyError: 'API_BASE_URL' not found in environment",
         None, "打标签 bug/config，检查加载顺序"),

        ("#107", "Docker 部署时无法读取本地配置",
         "本地跑没问题，打包成 docker 之后配置文件读不到。",
         "grace", ["config", "deploy"], [],
         "FileNotFoundError: /app/config.yaml",
         None, "打标签 bug/config"),

        # --- dependency 层 ---
        ("#108", "安装时依赖冲突无法解决",
         "pip install 时提示 requests 版本冲突，一个依赖要 2.25，另一个要 2.31。",
         "henry", ["dependency"], [],
         "ERROR: Cannot install -r requirements.txt because these package versions have conflicting dependencies.",
         None, "打标签 bug/dependency"),

        ("#109", "升级 Python 版本后无法导入模块",
         "从 3.9 升到 3.11 之后，import 报错说找不到某个模块属性。",
         "ivan", ["dependency"], [],
         "ImportError: cannot import name 'cached_property' from 'functools'",
         None, "打标签 bug/dependency"),

        # --- crash 层 ---
        ("#110", "处理空文件时程序崩溃退出",
         "上传一个 0 字节的文件时进程直接退出，没有报错信息。",
         "judy", ["crash"], [],
         "Segmentation fault (core dumped)",
         None, "打标签 bug/crash"),

        ("#111", "并发请求时偶发进程崩溃",
         "压测到 100 并发时进程会崩溃，50 并发正常。",
         "kevin", ["crash", "performance"], [],
         "Fatal Python error: Aborted",
         None, "打标签 bug/crash"),

        # --- duplicate 层 ---
        ("#112", "高峰时段接口超时（重复）",
         "上午高峰期调用接口超时。",
         "laura", ["timeout"], [],
         "requests.exceptions.Timeout",
         "#101", "标记为 duplicate of #101"),

        ("#113", "403 权限错误（重复）",
         "配置了 token 还是 403。",
         "mike", ["permission"], [],
         "403 Forbidden",
         "#104", "标记为 duplicate of #104"),

        ("#114", "环境变量不生效（重复）",
         ".env 配置不生效。",
         "nancy", ["config"], [],
         "KeyError: 'API_BASE_URL'",
         "#106", "标记为 duplicate of #106"),

        # --- feature 层 ---
        ("#115", "希望能支持导出为 Excel 格式",
         "现在只能导出 CSV，希望增加 Excel 导出。",
         "oscar", ["feature"], [],
         "",
         None, "打标签 enhancement"),

        ("#116", "建议增加暗色主题",
         "夜间使用时白色背景太刺眼，希望能切换暗色主题。",
         "peggy", ["feature"], [],
         "",
         None, "打标签 enhancement"),

        # --- unclear 层 ---
        ("#117", "用不了",
         "用不了，帮我看看。",
         "quinn", [], [],
         "",
         None, "转人工（信息不足）"),

        ("#118", "有 bug",
         "有 bug，快修。",
         "rita", [], [],
         "",
         None, "转人工（信息不足）"),
    ]

    issues = []
    for iid, title, body, author, labels, _, stack, dup, action in raw:
        issues.append(Issue(
            issue_id=iid, title=title, body=body, author=author,
            repo="demo/repo", labels=[], stack_trace=stack,
            gold_labels=labels, gold_is_duplicate_of=dup, gold_action=action,
        ))

    return DatasetBuilder(seed=42).build(
        issues,
        source="synthetic",
        version="syn-v1",
        notes=("⚠️ 合成数据，仅用于跑通流程与演示。"
               "症状描述比真实 issue 规整，指标会虚高。"
               "**不要把这个数据集上的数字写进简历。**"),
    )
