"""
Nightwatch · 错误分类框架（error taxonomy）

================================================================================
为什么"错了 4 条"不是一个可用的结论
================================================================================
评测报告最容易停在这一步：

    "61 条里有 4 条没命中。"

这句话信息量为零 —— 因为它把**四种性质完全不同**的失败混成了一个数字：

    · 模型不知道有这个类别        → 要补词表 / 提示词
    · 模型被无关信息带偏了        → 要改提示词的注意力引导
    · **评测代码自己把对的判成错的** → 要修评测，跟模型没关系
    · 工具/接口挂了导致没跑完     → 要修工程，也不是模型问题

四条里**有两条根本不是模型的问题**。如果报告只说"错了 4 条"，
下一步的动作会指向模型 —— 而真正的病灶在别处。

★ 这就是 error analysis 的意义：
  **它把"错了多少"变成"错在哪一层"，而后者才是能指导行动的信息。**

================================================================================
五个正交类别
================================================================================
    knowledge_gap    领域知识缺失：类别名/缩写不在词表里，模型**没机会**答对
    reasoning_error  推理被误导：信息都在，但被无关内容带偏
    evaluation_bug   评测侧缺陷：模型答对了，是映射/归约环节把它丢了
    tool_failure     工具调用失败：流程没跑完，谈不上推理对错
    no_error         预测正确（或 gold 本身存疑）

**"正交"是什么意思**：一条错误只应该落在**一个**类别里。
如果分类结果里出现大量"既像 A 又像 B"，说明类别定义有重叠，
需要重新划分；本模块用**有序规则**保证每条记录只命中一个类别。

================================================================================
★★ 分类是关于"归因"的判断，不是关于"对错"的判断 ★★
================================================================================
区分 `knowledge_gap` 和 `evaluation_bug` 是本模块最重要的价值，
也是**最容易搞反**的一处：

    knowledge_gap ：模型输出里**根本没有**正确信号 → 它不知道
    evaluation_bug：模型输出里**已经有**正确信号，但最终预测没有 → 被丢了

判据是"**原始预测里有没有指向正确答案的线索**"，
而不是"预测对不对" —— 后者两个类别都一样错。

★ 卡点在于：要判断"原始输出里的某个词是不是指向 gold"，
  就需要知道标签别名的对应关系（`ci` → Build）。
  而这张表**正属于评测侧的 `to_coarse`** —— 如果本模块自己再存一份，
  就又变成"同一件事两份实现"（本项目已经踩过四次）。

  所以本模块**不存别名表**，而是由调用方通过 `context["coarsen"]`
  传一个函数进来。这样：
      · 别名知识始终只有一份（在评测侧）
      · 本模块可以在**不知道标签体系**的情况下工作（也能被单测）
      · 换标签体系时不用改本模块

================================================================================
已知局限（写在这里，而不是等被问）
================================================================================
  ① **这五个类别不是 MECE 的穷尽划分**。比如"模型答对了但 gold 标错了"
     会被归到 `no_error` —— 但那是**数据的问题**，需要人工复核才能发现，
     规则分类器识别不出来。`context` 里可以传 `gold_suspect=True`
     把这类显式标出来。
  ② 分类靠**规则 + 调用方提供的上下文**，不是模型判断。
     好处是完全确定、可复现、可单测；代价是**上下文给得不全就会误分**。
     所以 `taxonomy_report` 会把"信息不足"的比例单独报出来，
     而不是把它悄悄塞进某个类别里。
  ③ 一条真实错误往往**多因叠加**（既缺词表、又被干扰）。
     本模块按优先级只报**最靠前的一个**（见规则顺序），
     `evidence` 里会列出命中该规则的依据，便于人工复核。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# 类别定义
# =============================================================================

class ErrorCategory(str, Enum):
    """错误的五个正交类别。

    继承 `str` 是为了让它能直接进 JSON（`json.dumps` 认得 str 子类），
    报告导出时不需要额外转换。
    """
    KNOWLEDGE_GAP = "knowledge_gap"
    REASONING_ERROR = "reasoning_error"
    EVALUATION_BUG = "evaluation_bug"
    TOOL_FAILURE = "tool_failure"
    NO_ERROR = "no_error"


@dataclass(frozen=True)
class CategoryInfo:
    """单个类别的说明。**修法写在数据里**，而不是散在文档各处 ——
    这样报告可以直接把"该怎么办"打出来，不用人去翻文档对照。"""
    name_cn: str
    definition: str
    typical_signal: str
    action: str
    blame: str          # 责任归属：模型 / 评测 / 工程 —— 决定了该找谁修


CATEGORY_INFO: dict[ErrorCategory, CategoryInfo] = {
    ErrorCategory.KNOWLEDGE_GAP: CategoryInfo(
        name_cn="领域知识缺失",
        definition="类别名或缩写不在词表里，模型没有机会答对",
        typical_signal="标题前缀（如 DEPR:）不在已知前缀表里",
        action="补词表 / 在提示词里列出该类别名",
        blame="模型（知识侧）",
    ),
    ErrorCategory.REASONING_ERROR: CategoryInfo(
        name_cn="推理被误导",
        definition="信息都在，但被无关内容（模板、勾选项、旧描述）带偏",
        typical_signal="正文含模板化干扰文本；模型照抄了标题前缀当结论",
        action="改提示词的注意力引导 / 明确告诉它忽略模板段落",
        blame="模型（推理侧）",
    ),
    ErrorCategory.EVALUATION_BUG: CategoryInfo(
        name_cn="评测侧缺陷",
        definition="模型输出里已有正确信号，但映射或归约环节把它丢了",
        typical_signal="原始预测里的词能对应到 gold，最终预测里却没有",
        action="修评测代码（映射表 / 归约逻辑），**不要动模型**",
        blame="评测（与模型无关）",
    ),
    ErrorCategory.TOOL_FAILURE: CategoryInfo(
        name_cn="工具调用失败",
        definition="流程没跑完（异常、超时、接口报错），谈不上推理对错",
        typical_signal="记录里的 error 字段非空",
        action="修工程（重试 / 降级 / 超时），必要时从指标里剔除",
        blame="工程（与模型无关）",
    ),
    ErrorCategory.NO_ERROR: CategoryInfo(
        name_cn="无错误",
        definition="预测正确；或预测与 gold 不符但 gold 本身存疑",
        typical_signal="预测命中 gold；或 context 显式标记 gold_suspect",
        action="无需修复；gold 存疑的需人工复核",
        blame="—",
    ),
}

#: 类别优先级（数字越小越先判定）。
#
# ★ 顺序不是随便排的，它决定了"多因叠加时报哪一个"：
#     tool_failure 最前 —— 流程都没跑完，后面的推理讨论没有意义
#     no_error     次之   —— 先确认"到底错没错"，再谈错在哪
#     evaluation_bug 第三 —— **必须先排除评测侧的锅，再去说模型的不是**。
#                            否则会把评测的 bug 记在模型头上，
#                            然后照着错误的方向去优化模型。
#     然后是 knowledge_gap（没机会答对）和 reasoning_error（有机会但没抓住）——
#     这两者的分界是"该不该怪它"：不知道可以补，被误导才需要调引导。
_PRIORITY = [
    ErrorCategory.TOOL_FAILURE,
    ErrorCategory.NO_ERROR,
    ErrorCategory.EVALUATION_BUG,
    ErrorCategory.KNOWLEDGE_GAP,
    ErrorCategory.REASONING_ERROR,
]


@dataclass
class ErrorVerdict:
    """一条错误的分类判定（含依据）。"""
    category: ErrorCategory
    reason: str
    evidence: list[str] = field(default_factory=list)
    insufficient_info: bool = False     # 上下文不足，分类结果不可靠
    #: 证据强度：`"strong"` = 找到了具体判据；`"weak"` = 落到了兜底分支。
    #: ★ 为什么要单独标出来：兜底分支总是会接住"其他原因"的那些，
    #:   如果把它和"已定位到原因"的混在一起报，读者会以为每条都查清了。
    evidence_strength: str = "strong"

    def as_dict(self) -> dict:
        info = CATEGORY_INFO[self.category]
        return {
            "category": self.category.value,
            "name_cn": info.name_cn,
            "blame": info.blame,
            "reason": self.reason,
            "evidence": self.evidence,
            "insufficient_info": self.insufficient_info,
            "evidence_strength": self.evidence_strength,
        }


# =============================================================================
# 归一化工具
# =============================================================================

def _norm(x) -> str:
    """把标签压成可比较的小写形式（只做大小写和分隔符统一，不做语义映射）。

    ⚠️ 刻意**不在这里做粗化**（如 `ci` → Build）——
       那一层属于评测侧的 `to_coarse`，本模块不复制它的知识。
    """
    s = str(x).strip().lower()
    for sep in ("/", ":", "-", "_", " "):
        s = s.replace(sep, "|")
    return s.split("|")[-1] if "|" in s else s


def _title_prefix(title: str) -> str:
    """取标题前缀（`DEPR: xxx` → `depr`）。

    ★ 只认"全大写字母 + 冒号"这种形式 —— 这是本项目跟踪的仓库
      （pandas 等）的真实惯例。宽松匹配会把 `Fix: ...` 这类
      普通英文句首也当成前缀，反而引入噪声。
    """
    head = str(title).split(":", 1)[0].strip()
    if head and head.isupper() and head.isalpha() and len(head) <= 6:
        return head.lower()
    return ""


# =============================================================================
# 核心：分类
# =============================================================================

def _evaluate(pred, gold, context: dict | None) -> ErrorVerdict:
    """有序规则分类。**这是唯一的实现**，两个公开入口都走它。"""
    ctx = dict(context or {})
    p_raw = list(pred or [])
    g_raw = list(gold or [])
    p_norm = {_norm(x) for x in p_raw if str(x).strip()}
    g_norm = {_norm(x) for x in g_raw if str(x).strip()}

    # ---------- ① 工具失败：流程没跑完，后面都别谈了 ----------
    err = str(ctx.get("error", "") or "").strip()
    if err:
        return ErrorVerdict(
            ErrorCategory.TOOL_FAILURE,
            "记录里带 error 字段 —— 流程没跑完，不属于推理或知识问题",
            [f"error: {err[:80]}"],
        )

    # ---------- ② 到底错没错 ----------
    if p_norm & g_norm:
        return ErrorVerdict(
            ErrorCategory.NO_ERROR,
            "预测命中 gold（Top-1 有交集）",
            [f"预测 {sorted(p_norm)} ∩ 真值 {sorted(g_norm)}"],
        )
    if ctx.get("gold_suspect"):
        return ErrorVerdict(
            ErrorCategory.NO_ERROR,
            "调用方标记 gold 存疑 —— 无法判定为模型错误",
            ["context.gold_suspect = True"],
        )

    # ---------- ③ 先排除评测侧的锅 ----------
    #
    # ★★ 这是本模块最关键的一条规则 ★★
    #
    # 判据不是"预测对不对"（那和别的类别一样），而是
    # "**原始预测里有没有指向正确答案的线索**"。
    #
    # ⚠️ 判据必须看 `predicted_raw`（粗化**之前**的原始输出），
    #    不能看 `pred` —— `pred` 是粗化**之后**的结果，
    #    线索正是在这一步被丢掉的。拿 pred 去判会永远判不出来。
    #    （这一点是本模块第一版写错的地方：写成用 pred 遍历，
    #      结果 evaluation_bug 这条规则**永远不会触发** ——
    #      因为若 coarsen(pred) 命中 gold，规则②早就返回 NO_ERROR 了。
    #      一个永不触发的分支比没有分支更糟：它看起来在守。）
    #
    # 真实例证（data/prospective_eval.json 里的 4 条未命中）：
    #     #67982  title="CI: ..."  raw=['CI','config','dependency']  gold=Build → 预测 []
    #     #66735  title="BLD: ..." raw=['BLD','needs-info']          gold=Build → 预测 []
    #   模型**明明输出了 `CI` / `BLD`**，是 to_coarse 没收录这两个缩写，
    #   归约时把它们丢了 → 最终预测为空。
    #   这四条如果只报"错了 4 条"，会被当成模型的分类能力问题 —— 而实际不是。
    coarsen = ctx.get("coarsen")
    raw_pred = ctx.get("predicted_raw")
    if callable(coarsen) and raw_pred:
        hit_tokens = []
        for tok in raw_pred:
            try:
                coarse = coarsen(str(tok))
            except Exception:                               # noqa: BLE001
                coarse = None
            if coarse and _norm(coarse) in g_norm:
                hit_tokens.append(f"{tok}→{coarse}")
        if hit_tokens:
            return ErrorVerdict(
                ErrorCategory.EVALUATION_BUG,
                "原始预测里已含指向正确答案的信号，但最终预测丢失了它 —— "
                "问题出在映射/归约环节，不是模型",
                [f"原始预测 {tok}" for tok in hit_tokens]
                + [f"最终预测 {sorted(p_norm) or '空'}，真值 {sorted(g_norm)}"],
            )

    # ---------- ④ 知识缺失：词表里根本没这个类别 ----------
    prefix = str(ctx.get("prefix") or _title_prefix(ctx.get("title", ""))).lower()
    known = ctx.get("known_prefixes")
    if prefix and known is not None:
        known_norm = {str(k).lower() for k in known}
        if prefix not in known_norm:
            return ErrorVerdict(
                ErrorCategory.KNOWLEDGE_GAP,
                f"标题前缀 `{prefix.upper()}:` 不在已知前缀表里 —— "
                f"模型没有渠道知道它该归到 {sorted(g_norm)}",
                [f"前缀 {prefix!r} ∉ 已知前缀 {sorted(known_norm)}"],
            )

    # ---------- ⑤ 兜底：推理被误导 ----------
    #
    # 走到这里说明：信息是有的（词表里有、评测也没丢），但模型没用对。
    #
    # ★ 但要分清两件事 —— "找到了具体的误导源" vs "只是归到这一类"：
    #   这两者的证据强度差很多，直接混在一起报会让读者以为
    #   "每一条都定位到了原因"，而实际可能只是落到了兜底分支。
    ev: list[str] = [f"预测 {sorted(p_norm) or '空'}，真值 {sorted(g_norm)}"]
    positive = False                      # 是否找到了**具体的**误导源

    # 误导源一：模型照抄了标题前缀。
    # 判据要求"预测 == 前缀所指的类别"且"前缀 ≠ 真值" ——
    # 只有这样才能说"是前缀把它带偏的"。
    # （如果预测与前缀所指不同，那说明它没照抄前缀，就不能怪前缀。）
    if prefix and callable(coarsen):
        try:
            pc = coarsen(prefix)
        except Exception:                                   # noqa: BLE001
            pc = None
        if pc and _norm(pc) in p_norm and _norm(pc) not in g_norm:
            ev.append(f"模型照抄了标题前缀 `{prefix.upper()}:`（→{pc}），"
                      f"但真值不是它 —— **前缀本身就是误导源**")
            positive = True

    # 误导源二：正文里的模板/勾选项。
    distractors = ctx.get("distractors") or []
    body = str(ctx.get("body", "") or "")
    if body and distractors:
        found = [d for d in distractors if d and str(d) in body]
        if found:
            ev.append(f"正文命中干扰词 {found} —— 疑似模板段落把结论带偏")
            positive = True

    reason = ("信息齐备但结论不对，且**已定位到具体误导源**"
              if positive else
              "信息齐备但结论不对，**未定位到具体误导源** —— "
              "只能归到这一类（需要人工看原文才能进一步细分）")
    return ErrorVerdict(
        ErrorCategory.REASONING_ERROR,
        reason,
        ev + ([] if ctx else ["⚠ 未提供 context，无法进一步定位（见模块 docstring 局限②）"]),
        insufficient_info=not ctx,
        evidence_strength="strong" if positive else "weak",
    )


def classify_error(pred, gold, context: dict | None = None) -> ErrorCategory:
    """把一条错误分类到五个正交类别之一。

    参数
    ----
    pred    : 最终预测的标签列表（已粗化）
    gold    : 真值标签列表
    context : 可选上下文。**提供的字段越多，分类越准**：

        `error`           工具/接口的错误信息（非空 → tool_failure）
        `title`           标题（用于取前缀、定位干扰）
        `body`            正文（用于检查干扰词）
        `prefix`          显式指定前缀（不传则从 title 里提取）
        `known_prefixes`  系统已知的前缀集合（判断 knowledge_gap 的必要输入）
        `predicted_raw`   粗化**之前**的原始预测 ★ 判断 evaluation_bug 的必要输入
                          （最终预测 `pred` 是被丢掉线索之后的结果，
                            拿它判断会永远判不出 evaluation_bug）
        `coarsen`         可调用对象 `str -> str|None`，做标签粗化
                          （★ 由调用方传入，本模块**不复制**这张表 ——
                            理由见模块 docstring）
        `distractors`     干扰词列表（如模板里的 "Feature Type"）
        `gold_suspect`    显式标记 gold 存疑

    返回
    ----
    `ErrorCategory`。需要看**判定依据**时用 `classify_error_detailed`。

    什么时候用
    ----------
    只在**预测与 gold 不符**时调用才有意义（命中的会返回 `NO_ERROR`，
    但那只是兜底行为，正常流程应该先过滤掉命中的样本）。
    """
    return _evaluate(pred, gold, context).category


def classify_error_detailed(pred, gold,
                            context: dict | None = None) -> ErrorVerdict:
    """同 `classify_error`，但返回带**依据**和**责任归属**的完整判定。

    ★ 为什么要有这个"详细版"：报告里只写"3 条 evaluation_bug"是不够的 ——
      读者需要知道**凭什么这么判**，否则无法复核。
      判定依据（`evidence`）会逐条列出命中的规则和原始数据。
    """
    return _evaluate(pred, gold, context)


# =============================================================================
# 汇总报告
# =============================================================================

def taxonomy_report(errors: list) -> dict:
    """把一批错误汇总成分类报告。

    `errors` 里每项可以是 `ErrorVerdict`、`ErrorCategory`，
    或形如 `{"category": "..."}` 的字典（方便从 JSON 读回来）。

    返回各类别的**数量 / 占比 / 责任归属**，外加：
      · `样本量`（总错误数，不是总样本数 —— 两者别混）
      · `信息不足比例`：上下文没给全、分类不可靠的比例
      · `小样本提示`

    ========================================================================
    为什么一定要把"责任归属"单独聚合一次
    ========================================================================
    因为**"模型的问题"和"评测的问题"要分开计数**。

    假设 4 条错误里有 3 条是 `evaluation_bug`：
      只看"错了 4 条"       → 去改模型（改错方向了）
      看责任归属 3/4 在评测 → 先修映射表（改对方向了）

    ★ 本条来自真实数据：`data/prospective_eval.json` 的 4 条未命中里，
      3 条其实是 `to_coarse` 没收录 `CI`/`BLD` 导致的 ——
      模型输出了正确的信号，是评测丢的。
    """
    if not errors:
        return {"样本量": 0, "提示": "没有错误可供分类", "分类": {}}

    counts: dict[str, int] = {}
    blames: dict[str, int] = {}
    insufficient = 0
    weak = 0
    n = 0

    for e in errors:
        if isinstance(e, ErrorVerdict):
            cat, ins = e.category, e.insufficient_info
            weak += int(e.evidence_strength == "weak")
        elif isinstance(e, ErrorCategory):
            cat, ins = e, False
        elif isinstance(e, dict):
            raw = str(e.get("category", "")).strip()
            try:
                cat = ErrorCategory(raw)
            except ValueError:
                continue                     # 未知类别：跳过，不猜
            ins = bool(e.get("insufficient_info"))
            weak += int(str(e.get("evidence_strength", "strong")) == "weak")
        else:
            continue
        n += 1
        counts[cat.value] = counts.get(cat.value, 0) + 1
        blames[CATEGORY_INFO[cat].blame] = \
            blames.get(CATEGORY_INFO[cat].blame, 0) + 1
        insufficient += int(ins)

    if n == 0:
        return {"样本量": 0, "提示": "没有可识别的错误类别",
                "分类": {}}

    detail = {}
    for key, cnt in sorted(counts.items(), key=lambda kv: -kv[1]):
        cat = ErrorCategory(key)
        info = CATEGORY_INFO[cat]
        detail[info.name_cn] = {
            "category": key,
            "数量": cnt,
            "占比": round(cnt / n, 4),
            "责任归属": info.blame,
            "定义": info.definition,
            "典型信号": info.typical_signal,
            "该怎么做": info.action,
        }

    return {
        "样本量": n,
        "说明": "样本量是**错误条数**，不是评测集大小 —— 两者别混。",
        "分类": detail,
        "责任归属聚合": dict(sorted(blames.items(), key=lambda kv: -kv[1])),
        "信息不足比例": round(insufficient / n, 4),
        # ★ 兜底分支总是会接住"其他原因"的那些。把它单独报出来，
        #   读者才知道"这条结论有多实"——否则会以为每条错误都定位到了原因。
        "证据强度不足比例": round(weak / n, 4),
        # 延续项目一贯的"小样本提示"风格：不拦着你用，但要说清边界
        "提示": ("错误条数 < 5，占比仅作方向参考，不宜据此下结论"
                 if n < 5 else
                 "分类靠规则 + 调用方提供的上下文，上下文不全时可能误分；"
                 "优先看「责任归属聚合」决定下一步动作"),
        "已知局限": [
            "五类不是 MECE 的穷尽划分：模型答对但 gold 标错的会被归到 no_error，"
            "那属于数据问题，需要人工复核",
            "多因叠加时只报优先级最高的一个（tool_failure > no_error > "
            "evaluation_bug > knowledge_gap > reasoning_error）",
        ],
    }
