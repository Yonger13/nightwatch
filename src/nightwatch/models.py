"""
Nightwatch · 数据模型

================================================================================
这个文件为什么值得单独存在
================================================================================
Agent 项目和普通 CRUD 项目最大的区别之一：**状态结构决定能力上限**。

普通项目的数据模型是"业务实体的映射"，Agent 项目的数据模型还要回答：
  - 我做过什么？（轨迹）
  - 我记住了什么？（记忆）
  - 我为什么这么做？（决策依据）
  - 我之前错了没有？（失败与纠正）

所以这里的每个字段都值得想一想"为什么要存它"。下面会在关键字段上标注。

================================================================================
关键认知：字段设计暴露你的判断力
================================================================================
面试官看你的数据模型，其实在看一个问题：
    **"这个人知不知道什么信息未来会有用？"**

举例：如果你在记忆里只存"问题 → 答案"，那你的系统永远不会进步。
如果你存了"试过什么但失败了"，下次就能跳过弯路。
如果你存了"人工改了什么"，你就有了一条不需要花钱标注的训练信号。
================================================================================
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash(*parts: Any) -> str:
    """生成稳定的短哈希。用于幂等键 —— 同样的输入必须得到同样的键。"""
    raw = "||".join(str(p) for p in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


# =============================================================================
# 枚举：状态空间
# =============================================================================
# 为什么用枚举而不是字符串？
# 因为这些是"有限且互斥"的状态，写死在类型里能在编码期就发现错误。
# 更重要的是：它逼你想清楚"到底有几种可能的结果"。
# 很多 Agent 项目的失败恰恰是因为作者只考虑了"成功"这一种情况。

class IssueState(str, Enum):
    """issue/工单的状态机。注意 open → triaged 是 Agent 的产出，closed 是人工决策。"""
    OPEN = "open"
    TRIAGED = "triaged"       # Agent 已给出分类与建议
    AWAITING_HUMAN = "awaiting_human"  # 转人工（Agent 不确定或超出边界）
    CLOSED = "closed"


class Outcome(str, Enum):
    """
    一次处理的结果。
    注意 MISDIAGNOSED 是单独一类 —— "误诊"和"升级"是两种完全不同的失败，
    值得分开统计，因为它们的修复手段不一样。
    """
    RESOLVED = "resolved"        # 判断正确，建议被采纳
    ESCALATED = "escalated"      # 超出能力边界，正确转人工（这不是失败）
    MISDIAGNOSED = "misdiagnosed"  # 判断错误
    UNFINISHED = "unfinished"    # 达到轮数上限，未完成


class HumanVerdict(str, Enum):
    """
    人工反馈。**这是整个项目里质量最高的信号来源。**
    它不花钱、真实、且直接可用 —— 直接决定了记忆的权重和评测的 ground truth。
    """
    ACCEPTED = "accepted"   # 采纳
    EDITED = "edited"       # 修改后采纳 —— 修改内容是最有价值的纠错样本
    REJECTED = "rejected"   # 驳回
    UNKNOWN = "unknown"


class MemoryTier(str, Enum):
    """
    三层记忆。不是随便分的 —— 每层解决一个不同时间尺度的问题：
      WORKING  —— 这次会话里"我刚才做了什么"（秒-分钟）
      EPISODIC —— "过去发生过什么"（天-月，跨会话）
      SEMANTIC —— "我们约定怎么处理这类问题"（月-年，稳定）
    """
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


# =============================================================================
# Issue：被处理的对象
# =============================================================================

@dataclass
class Issue:
    """一个待处理的 issue / 工单。

    设计要点：**输入要尽量是半结构化的**。
    业务报告里"高 ROI 画像"明确要求"半结构化输入：邮件、表单、单据，不是纯自由文本"。
    issue 天然满足这一点（正文 + 标签 + 报错堆栈 + 附件），这比纯聊天场景更容易评估。
    """
    issue_id: str
    title: str
    body: str
    author: str                     # 提交者。**记忆要能回答"这个作者报过什么"**
    repo: str = ""
    labels: list[str] = field(default_factory=list)      # 当前已有标签
    comments: list[dict] = field(default_factory=list)
    attachments: list[str] = field(default_factory=list)
    stack_trace: str = ""           # 报错堆栈，是判重的强信号
    state: IssueState = IssueState.OPEN
    created_at: str = field(default_factory=_now)

    # --- ground truth（来自维护者的真实动作，不是我们标的）---
    # 这几个字段是"真实用户"这个需求的直接产物：
    # 开源项目的维护者会真实地打标签、判定重复、关闭 issue，
    # 这些动作就是天然的标注，不需要我们花钱请人标。
    gold_labels: list[str] = field(default_factory=list)
    gold_is_duplicate_of: Optional[str] = None   # 维护者标记的"重复于哪个 issue"
    gold_action: str = ""                        # 维护者最终做了什么

    def fingerprint(self) -> str:
        """内容指纹。用于幂等：同一个 issue 重复处理不应产生重复记忆。"""
        return _hash(self.repo, self.issue_id, self.title)

    def search_text(self) -> str:
        """用于向量化/检索的文本。**这个拼接顺序是有讲究的**：
        标题 + 堆栈优先，因为它们包含最多判别信息；正文可能很长且噪音多。"""
        parts = [self.title, self.stack_trace[:500], self.body[:1500]]
        return "\n".join(p for p in parts if p)


# =============================================================================
# ToolCall / 轨迹
# =============================================================================

@dataclass
class ToolCall:
    """一次工具调用的记录。

    注意 tool_input 会被用来做**循环检测**（比对最近 N 轮的参数是否完全相同）。
    所以这里存的是"归一化后的参数摘要"，而不是原始字符串 ——
    否则 "查询 A  " 和 "查询 A" 会被当成两次不同的调用。
    """
    step: int
    tool_name: str
    tool_input: dict
    ok: bool
    output: Any = None
    error: str = ""
    error_kind: str = ""       # timeout / rate_limit / invalid_args / permission / unknown
    latency_ms: int = 0
    offloaded: bool = False    # 结果是否被卸载到磁盘（体积过大时）
    offload_ref: str = ""      # 卸载后的文件指针，需要细节时可以读回来
    ts: str = field(default_factory=_now)

    def norm_key(self) -> str:
        """归一化的调用签名，用于循环检测。"""
        try:
            args = json.dumps(self.tool_input, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError):
            args = str(self.tool_input)
        return f"{self.tool_name}::{args}"


@dataclass
class Step:
    """ReAct 循环中的一步：思考 → 行动 → 观察。

    为什么三步都要存？
    因为排查失败时你需要知道"它是想错了还是做错了"：
      - thought 错了 → 提示词问题
      - action 错了 → 工具描述问题（模型不知道这个工具是干嘛的）
      - observation 错了 → 工具实现问题
    这三类问题的修复手段完全不同。只存结果的话，你只能看到"最后答案不对"。
    """
    step: int
    thought: str = ""
    action: Optional[ToolCall] = None
    observation: str = ""
    ts: str = field(default_factory=_now)


@dataclass
class Trace:
    """一次完整处理的轨迹。

    这是"可观测性"的载体，也是评测的数据来源。
    没有 trace 的 Agent 项目 = 无法调试 = 也无法改进。
    """
    trace_id: str
    issue_id: str
    steps: list[Step] = field(default_factory=list)
    final_answer: str = ""
    predicted_labels: list[str] = field(default_factory=list)
    predicted_duplicate_of: Optional[str] = None
    confidence: float = 0.0
    outcome: Outcome = Outcome.UNFINISHED
    # 上下文占用快照：每轮记录各桶用了多少，用于排查"是哪次压缩丢了信息"
    context_usage: list[dict] = field(default_factory=list)
    # 记忆召回的明细：召回了哪些、用了哪些、哪些是无效召回
    memory_recalled: list[str] = field(default_factory=list)
    memory_used: list[str] = field(default_factory=list)
    total_tokens: int = 0
    started_at: str = field(default_factory=_now)
    finished_at: str = ""

    def record(self, step: Step) -> None:
        self.steps.append(step)

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    def ineffective_recall_rate(self) -> float:
        """无效召回率 = 召回了但没用的比例。

        **这是记忆系统最值得盯的指标之一。**
        很多人只看"召回命中率"，但那会误导你：
        有些记忆被频繁召回，仅仅因为它的向量"吸引"了很多不相关查询（噪音吸铁石）。
        真正要优化的是"召回来之后真的被用上"的比例。
        """
        if not self.memory_recalled:
            return 0.0
        unused = set(self.memory_recalled) - set(self.memory_used)
        return len(unused) / len(self.memory_recalled)


# =============================================================================
# Memory：记忆记录
# =============================================================================

@dataclass
class MemoryRecord:
    """一条情景记忆 = 一次「问题 → 根因 → 处置 → 结果」的完整快照。

    下面注释标了 **★** 的字段是"三个容易被忽略但很值钱的字段"。
    它们体现的是"什么信息未来会有用"这个判断。
    """
    memory_id: str
    tier: MemoryTier = MemoryTier.EPISODIC

    # --- 溯源 ---
    issue_id: str = ""            # 能回到原始 issue
    author: str = ""              # ★ 精确键：支持"同一作者报过吗"的查询
    repo: str = ""

    # --- 内容 ---
    symptom: str = ""             # 症状描述，**向量检索的主字段**
    symptom_tags: list[str] = field(default_factory=list)   # 结构化标签，用于过滤
    root_cause: str = ""          # 根因。**允许为空** —— "没定位到根因"本身也是信息
    resolution: str = ""

    # --- ★ 值钱字段 ①：保留失败的尝试 ---
    # 大多数项目只记成功路径。但"试过 A 不行，应该试 B"恰恰是下次省时间的地方 ——
    # 这是情景记忆比知识库更有价值的部分：知识库告诉你"正确答案"，
    # 记忆告诉你"在你这个环境下，哪条路走不通"。
    actions: list[dict] = field(default_factory=list)

    # --- 结果 ---
    outcome: Outcome = Outcome.UNFINISHED

    # --- ★ 值钱字段 ②：人工反馈 ---
    # human_verdict / human_edit 是唯一"不花钱、真实、可用"的标注信号。
    # 它同时服务三件事：① 评测的 ground truth ② 记忆权重的校正 ③ 提示词优化的样本
    human_verdict: HumanVerdict = HumanVerdict.UNKNOWN
    human_edit: str = ""          # 工程师改了什么 —— 最高质量的纠错样本

    # --- ★ 值钱字段 ③：访问统计 ---
    # 既是"这条记忆有没有用"的信号，也是衰减计算的输入。
    # 注意要同时记"被召回次数"和"召回后实际使用次数" —— 两者之差就是无效召回。
    access_count: int = 0         # 被召回次数
    use_count: int = 0            # 召回后真的被用上的次数
    last_access: str = ""

    # --- ★ 值钱字段 ⑤：误诊计数（结果反馈闭环）---
    #
    # "这条记忆被用了、但那次判断错了" 的次数。
    #
    # ★ 它必须和 use_count 配对看：
    #   单看 misuse_count 没有意义（用 100 次错 2 次是好事），
    #   要看的是**误诊率** = misuse / (use + 平滑项)。
    misuse_count: int = 0

    # --- ★ 值钱字段 ④：代码状态绑定（仓库变更时效性的地基）---
    #
    # 这一组字段是"**检测记忆过时**"的前提条件。
    #
    # ============================================================
    # 为什么必须有它们
    # ============================================================
    # 在这个字段出现之前，记忆只记了 `repo="pandas-dev/pandas"` 一个字符串 ——
    # **完全没有绑定任何代码状态**。这导致一个根本性的问题：
    #
    #     当仓库发生变更（重构、API 改名、模块拆分）时，
    #     系统**无法回答**"哪些记忆因此过时了" ——
    #     因为它不知道每条记忆当初依赖的是哪些代码。
    #
    # 这不是"漏了一个 if"，而是**缺了一整类信号通道**。
    #
    # （项目自己已经用实验证明了这种"过时记忆"的危害：
    #   `eval/poison.py` 的 STALE 类型污染 —— "曾经对、现在过时"，
    #   检测难度标为最高级，实测暴露次数 4 次 vs 其他类型全部 0 次。
    #   详见 docs/06 的 P2-1。）
    #
    # ============================================================
    code_refs: list[str] = field(default_factory=list)
    """这条记忆**依赖过的代码文件路径**（相对仓库根）。

    来源：处理该 issue 时调过 `read_code` 的那些路径
    （见 `lifecycle.py` 的 `_extract_code_refs`）。

    ★ 为什么是"读过哪些文件"而不是"改过哪些文件"？
      因为 Agent 是**只读**的（它只做分诊，不改代码）。
      它判断的依据就是它读过的那些文件 —— 那些文件变了，判断可能就不成立了。

    ★ 它的用途**不是**自动作废记忆，而是**缩小人工复核的范围**：
      "这次改动碰到了 3 个文件，其中 2 个被 5 条记忆引用过"
      → 把那 5 条推进待复核队列，而不是让 5000 条记忆全部重审。
    """

    repo_head: str = ""
    """写下这条记忆时，仓库的 commit（或快照标识）。

    ★ 为什么只存一个"标识"而不存完整的代码快照？
      因为快照的成本极高（几 MB × 每天几十条），而**收益递减** ——
      真正需要回答的问题是"有没有变"，而不是"变成什么样了"。

      **一个 commit hash 就能回答"有没有变"** —— 成本几乎为零。

    ★ 诚实说明：本项目**没有实现完整的代码快照比对**。
      这个字段的作用是：
        ① 标记"这条记忆是基于哪个时点的代码得出的"
        ② 支持未来做"从 A 到 B 之间改了哪些文件"的精确查询
      现在它只是个标识 —— 但**先把位置留出来**，
      否则以后要加时，历史数据全都是空的。
    """

    # --- 元数据 ---
    confidence: float = 0.0       # Agent 自评置信度（**不能直接当事实**）
    importance: float = 0.0       # 重要性评分（写入决策的依据）
    decay_weight: float = 1.0     # 衰减权重
    archived: bool = False        # 是否已移入冷存储
    embedding: list[float] = field(default_factory=list)
    created_at: str = field(default_factory=_now)

    def idempotency_key(self) -> str:
        """幂等键。

        为什么需要？因为记忆抽取是**异步**的（不阻塞主链路），异步就会有重试。
        没有幂等键的话，一次网络抖动就可能写进两条重复记忆。
        这就是"幂等"这个后端概念在 Agent 场景的具体落点。
        """
        return _hash(self.repo, self.issue_id, self.symptom[:80])


@dataclass
class PendingConflict:
    """无法自动裁决的记忆冲突，等待人工处理。

    **这个类的存在本身就是设计观点**：
    记忆系统不可能永远自洽。与其让 Agent 自己拍板（可能错上加错），
    不如把矛盾明确地挂出来，等人来判。
    面试时这一点是加分项 —— 它说明你不相信"全自动"。
    """
    conflict_id: str
    memory_a: str
    memory_b: str
    reason: str
    created_at: str = field(default_factory=_now)
    resolved_by: str = ""      # 谁解决的
    resolution: str = ""       # 保留哪条


@dataclass
class PendingPromotion:
    """待人工审核的"SOP 提炼"提案。

    ============================================================================
    ★ 这个类存在的理由：语义层的风险量级和情景层完全不同
    ============================================================================
    情景记忆写错了 → 最多影响**一次召回**（下次检索到一条错的参考）。
    语义记忆写错了 → 它是"我们约定这类问题这么处理"，
                      会被**所有后续同类任务**召回，污染面成倍放大。

    两者风险量级不同，所以门槛也要不同：
        情景记忆 → 自动写入（评分够就写）
        语义记忆 → **必须人工批准**（这是本类存在的意义）

    所以「待审核」这个状态是**一等公民**，不是流程上的临时态 ——
    它把"人的判断"变成了数据模型的一部分。

    ============================================================================
    为什么要有 `status` 而不是"待审核 = 表里存在，已审核 = 表里删除"
    ============================================================================
    因为**审核结果本身是重要的记录**：
      · 被拒绝的提案如果以后又被提出来，你应该能看到"上次为什么拒"
      · 审核历史能回答"这套 SOP 是谁、什么时候定的"
      · 审计需要（谁在什么时候批准了什么）

    **删除是最差的"状态表达"方式** —— 它丢掉了过程信息。
    这也是为什么很多系统用"软删除"而不是物理删除。
    """
    promotion_id: str
    pattern: str                       # 标签组合（如 "timeout|权限"）
    occurrences: int                   # 出现过多少次（支撑"高频"判断）
    sample_ids: list[str] = field(default_factory=list)
    sample_summary: str = ""           # 给审核人看的摘要（不用点进每条）
    status: str = "pending"            # pending / approved / rejected
    created_at: str = field(default_factory=_now)
    reviewed_at: str = ""
    reviewed_by: str = ""
    review_note: str = ""              # 审核备注（拒绝理由也要记）
    promoted_memory_id: str = ""       # 批准后生成的语义记忆 ID

    @property
    def is_pending(self) -> bool:
        return self.status == "pending"

    @property
    def is_approved(self) -> bool:
        return self.status == "approved"


# =============================================================================
# 序列化辅助
# =============================================================================

def to_dict(obj: Any) -> dict:
    """把 dataclass 转成可 JSON 化的 dict（枚举转字符串）。"""
    d = asdict(obj)
    for k, v in list(d.items()):
        if isinstance(v, Enum):
            d[k] = v.value
    return d


# =============================================================================
# 反序列化（服务拆分时新增）
# =============================================================================
# ★ 为什么原来不需要、现在需要？
#
# 原来记忆是**进程内**的，对象直接传递，不需要序列化往返。
# 拆成独立服务后，对象要经过 HTTP/JSON 传输 —— 于是必须能"重建"。
#
# 这里**没有用"通用反射式反序列化"**（那种按类型注解自动转换的写法），
# 而是每个类显式写一个 from_dict。理由：
#
#   ① `from __future__ import annotations` 让字段类型变成**字符串**，
#      通用实现要额外解析字符串类型名，反而更脆
#   ② 跨服务的数据契约**应该显式可见** —— 读代码就能看到"网络上传了什么字段"
#   ③ 显式版本能在字段缺失时给出合理默认值（向前兼容）
#
# **跨服务的数据契约不要用魔法。** 它会让你在排查"为什么字段丢了"时
# 完全找不到线索。

def _enum_or_default(enum_cls, value, default):
    """把字符串转成枚举，非法值回退到默认（而不是抛异常）。

    ★ 为什么容错而不是抛？
    因为服务端和客户端的版本可能不一致（滚动升级期间）。
    如果客户端收到了一个"服务端新增的枚举值"，抛异常会让整个
    降级链路崩掉 —— 而正确的行为是"尽力而为"。
    """
    try:
        return enum_cls(value)
    except (ValueError, TypeError):
        return default


def memory_from_dict(d: dict) -> "MemoryRecord":
    """从 dict 重建 MemoryRecord。

    字段缺失时用 dataclass 的默认值 —— 这是**向前兼容**的关键：
    服务端加了新字段后，老客户端仍能工作（只是忽略新字段）。
    """
    import dataclasses

    valid = {f.name for f in dataclasses.fields(MemoryRecord)}
    kwargs = {k: v for k, v in d.items() if k in valid}

    if "tier" in kwargs:
        kwargs["tier"] = _enum_or_default(MemoryTier, kwargs["tier"],
                                          MemoryTier.EPISODIC)
    if "outcome" in kwargs:
        kwargs["outcome"] = _enum_or_default(Outcome, kwargs["outcome"],
                                             Outcome.UNFINISHED)
    if "human_verdict" in kwargs:
        kwargs["human_verdict"] = _enum_or_default(
            HumanVerdict, kwargs["human_verdict"], HumanVerdict.UNKNOWN)
    return MemoryRecord(**kwargs)


def conflict_from_dict(d: dict) -> "PendingConflict":
    """从 dict 重建 PendingConflict。"""
    import dataclasses

    valid = {f.name for f in dataclasses.fields(PendingConflict)}
    return PendingConflict(**{k: v for k, v in d.items() if k in valid})


def to_json(obj: Any, indent: int = 2) -> str:
    return json.dumps(to_dict(obj), ensure_ascii=False, indent=indent)
