"""
Nightwatch · 工具层

================================================================================
为什么工具要单独成层
================================================================================
在 Agent 架构里，工具是**模型与真实世界的唯一接口**。它有三个特殊性：

  1. **工具描述就是提示词。** 模型只能通过 description 和参数 schema 理解
     这个工具能干什么。描述写得含糊，模型就会用错 ——
     而"模型用错工具"看起来像模型问题，实际是描述问题。
     所以下面的每个 description 都写得很具体（含边界和反例）。

  2. **工具是权限边界。** Agent 能做什么，完全等于它能调哪些工具。
     所以"工具分级"不是代码整洁问题，是安全问题。

  3. **工具数量要克制。** Gartner 把"用 50+ 工具的通用 Agent"点名为失败模式；
     JPMorgan 450+ 用例进生产的洞察是"规模来自大量小而受治理的应用"。
     所以本项目定稿 6 个工具，每加一个都要能回答"为什么非它不可"。

================================================================================
面试必问题：为什么 Skill 要放进沙箱？沙箱是数据隔离还是执行隔离？
                                    —— 公开面经（广州某厂 Python 后端实习）
================================================================================
**关键是别被那个"二选一"限死。** 这两个是不同层面的问题，要同时做：

    数据隔离 —— 它能看到什么
      文件系统可见范围、网络白名单、凭证隔离、能读哪些仓库路径

    执行隔离 —— 它能做什么
      CPU/内存/时间配额、能不能 fork、能不能发网络请求、能跑哪些命令

本项目的分级策略（见 config.TOOLS["sandbox"]）：
    只读查询类（search_issues / search_similar_issues / read_code / search_docs）
        → 直接执行。风险可控：它们只读数据、不改状态
    有副作用类（apply_labels / post_comment）
        → 进沙箱 + 幂等保证 + 可回滚
        理由：写错了会污染真实仓库，必须能被撤回
================================================================================
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..models import Issue, MemoryRecord, _hash


# =============================================================================
# 工具定义
# =============================================================================

@dataclass
class ToolResult:
    ok: bool
    data: Any = None
    error: str = ""
    error_kind: str = ""       # 喂给 RetryPolicy 判断能不能重试
    meta: dict = field(default_factory=dict)


@dataclass
class Tool:
    """一个工具。

    注意 input_schema 用的是 JSON Schema 子集 —— 因为它会被**直接送给模型**。
    这不是给程序员看的文档，是给模型看的说明书。
    """
    name: str
    description: str
    input_schema: dict
    handler: Callable[..., Any]
    readonly: bool = True          # 只读 → 直接执行；否则进沙箱
    idempotent: bool = True        # 写工具必须显式声明是否幂等

    def to_llm_spec(self) -> dict:
        """转成给模型看的工具描述。"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


# =============================================================================
# 沙箱
# =============================================================================

class Sandbox:
    """执行隔离 + 数据隔离。

    ============================================================
    诚实说明：这是一个**教学用的轻量实现**
    ============================================================
    真实生产环境的沙箱应该是：
      容器 / microVM，带 seccomp 或 gVisor，独立文件系统命名空间，
      网络策略由 iptables 或 sidecar 控制

    这里用进程内的策略模拟，原因是：
      - 让项目零依赖可跑
      - 但**把接口和概念立起来**，换成真沙箱时只需替换这个类

    所以重点不是这里的实现有多强，而是你能讲清楚"如果上生产该怎么做"。
    面试时这么说是加分的 —— 它说明你知道当前实现是权宜的，且知道正解是什么。
    """

    # 数据隔离：允许访问的路径前缀
    ALLOWED_PATH_PREFIX = ("data/", "issues/", "repos/")

    # 网络白名单（执行隔离的一部分）
    ALLOWED_HOSTS = ("api.github.com",)

    def __init__(self, timeout: float = 10.0, allow_network: bool = False):
        self.timeout = timeout
        self.allow_network = allow_network
        self.violations: list[str] = []

    def check_path(self, path: str) -> bool:
        """数据隔离检查：这个路径能读吗？"""
        normalized = path.replace("\\", "/").lstrip("./")
        ok = any(normalized.startswith(p) for p in self.ALLOWED_PATH_PREFIX)
        if not ok:
            self.violations.append(f"path_denied:{path}")
        return ok

    def check_host(self, host: str) -> bool:
        """执行隔离检查：这个主机能连吗？"""
        if not self.allow_network:
            self.violations.append(f"network_denied:{host}")
            return False
        ok = host in self.ALLOWED_HOSTS
        if not ok:
            self.violations.append(f"host_not_allowed:{host}")
        return ok

    def run(self, fn: Callable[[], Any], *, extra_check: Optional[Callable[[], bool]] = None):
        """在沙箱约束下执行。

        真实实现会把 fn 丢进子进程并加资源限制。这里做的是：
          ① 执行前置检查（由调用方提供）
          ② 捕获所有异常并转成结构化错误
          ③ 记录违规（用于可观测 —— 沙箱拦了什么，是重要的安全信号）
        """
        if extra_check is not None and not extra_check():
            return ToolResult(ok=False, error="沙箱策略拒绝",
                              error_kind="permission")
        try:
            return ToolResult(ok=True, data=fn())
        except TimeoutError as e:
            return ToolResult(ok=False, error=str(e), error_kind="timeout")
        except PermissionError as e:
            return ToolResult(ok=False, error=str(e), error_kind="permission")
        except (TypeError, ValueError, KeyError) as e:
            # 参数类错误：**不可重试**。重试一万次都一样，快速失败更快
            return ToolResult(ok=False, error=str(e), error_kind="invalid_args")
        except Exception as e:                      # noqa: BLE001
            return ToolResult(ok=False, error=str(e), error_kind="unknown")


# =============================================================================
# 数据源抽象
# =============================================================================

class IssueRepository:
    """内存版数据源。

    为什么要抽这一层？因为工具的实现不该关心"数据从哪来"。
    离线跑（用合成数据）和在线跑（用真实 GitHub API）应该共用同一套工具逻辑。
    """

    def __init__(self, issues: list[Issue] | None = None):
        self.issues: dict[str, Issue] = {}
        self._docs: dict[str, str] = {}
        self._pr_diffs: dict[str, dict] = {}
        for i in issues or []:
            self.issues[i.issue_id] = i

    def add(self, issue: Issue) -> None:
        self.issues[issue.issue_id] = issue

    def get(self, issue_id: str) -> Optional[Issue]:
        return self.issues.get(issue_id)

    def all(self) -> list[Issue]:
        return list(self.issues.values())

    def add_doc(self, name: str, content: str) -> None:
        self._docs[name] = content

    def docs(self) -> dict[str, str]:
        return dict(self._docs)

    # ------------------------------------------------ PR diff（P2-3 新增）
    # ★ 和 docs 一样，PR diff 也属于"**数据源**"而不是"工具"。
    #
    #   工具负责"解析和判断"，数据源负责"东西从哪来"。
    #   分开之后，离线跑（用预置的 diff）和在线跑（从 GitHub API 拉）
    #   可以共用同一套工具逻辑 —— 这是 `IssueRepository` 一开始就定下的边界。

    def add_pr_diff(self, pr_id: str, diff_text: str,
                    meta: Optional[dict] = None) -> None:
        """登记一个 PR 的 diff。

        meta 可带 title / author / base 等 —— 用于生成更完整的初审报告。
        """
        self._pr_diffs[pr_id] = {"diff": diff_text, "meta": meta or {}}

    def pr_diff(self, pr_id: str) -> Optional[dict]:
        return self._pr_diffs.get(pr_id)

    def pr_ids(self) -> list[str]:
        return list(self._pr_diffs)


# =============================================================================
# 六个工具的实现
# =============================================================================

class ToolRegistry:
    """工具注册表。共 6 个工具 —— 这是定稿数量，不再增加。

    学习建议：注意每个工具 description 的写法。
    对比一下"查 issue"和真实的描述，你会发现差异在哪：
    好的描述包含 **用途 + 边界 + 反例**，而不只是"这是干什么的"。
    """

    def __init__(self, repo: IssueRepository, store=None, sandbox: Sandbox | None = None,
                 current_issue_id: str = "", encoder=None):
        self.repo = repo
        self.store = store                        # MemoryStore，供工具查记忆
        self.sandbox = sandbox or Sandbox()
        # ★ 编码器必须与记忆库用同一个 ——
        # 否则"记忆里的向量"和"工具算出来的查询向量"不在同一个空间里，
        # 相似度算出来就是噪音。这是个隐蔽但致命的错误：
        # 系统不会报错，只会静默地给出很低的相似度，看起来像"没找到相关记忆"。
        #
        # 优先用 store 的编码器（保证一致），其次用注入的，最后才新建。
        if encoder is not None:
            self.encoder = encoder
        elif store is not None and getattr(store, "encoder", None) is not None:
            self.encoder = store.encoder
        else:
            from ..memory.store import build_encoder
            self.encoder = build_encoder()
        # ★ 当前正在处理的 issue 编号。
        #
        # 为什么需要这个字段？因为检索类工具必须**排除"我自己"**。
        #
        # 这个字段是踩坑之后加的：不放这个字段时，Agent 检索历史 issue
        # 会把当前 issue 自己也检索出来，然后得出"我和我高度相似，建议判重"
        # 这种荒谬结论。
        #
        # 真实场景同样会遇到：历史库里可能有同一个 issue 的旧记录
        # （比如用户重复提交了两次），检索时如果不排除当前 id，
        # 就会自己判自己重复。
        self.current_issue_id = current_issue_id
        self._tools: dict[str, Tool] = {}
        # 写操作的幂等账本：记录已经执行过的写操作
        # 这是"幂等"在 Agent 场景的具体落点 —— 重试时不会重复打标签
        self._write_ledger: set[str] = set()
        self._register_all()

    def with_issue(self, issue_id: str) -> "ToolRegistry":
        """返回一个"当前处理对象已切换"的注册表（共享同一份数据源和账本）。

        这样评测时可以复用同一个 repo（模拟"历史库里已经有几百个 issue"），
        同时保证每个 issue 处理时检索不会把自己算进去。
        """
        return ToolRegistry(self.repo, store=self.store, sandbox=self.sandbox,
                            current_issue_id=issue_id, encoder=self.encoder)

    # ------------------------------------------------------------ 注册

    def _register_all(self) -> None:
        self._add(Tool(
            name="search_issues",
            description=(
                "在项目的历史 issue 中检索。用于：确认某个问题以前是否出现过、"
                "了解同类问题的历史处理方式。\n"
                "**不支持**语义相似检索（那是 search_similar_issues 的职责），"
                "本工具做的是关键词与标签的精确/模糊匹配。\n"
                "当你已经知道问题的标签或关键词时用这个；"
                "只有一段模糊的症状描述时用 search_similar_issues。"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "keywords": {"type": "string", "description": "检索关键词"},
                    "label": {"type": "string", "description": "按标签过滤（可选）"},
                    "limit": {"type": "integer", "default": 10},
                },
                "required": ["keywords"],
            },
            handler=self._search_issues,
            readonly=True,
        ))

        self._add(Tool(
            name="search_similar_issues",
            description=(
                "用语义相似度找出与给定文本最接近的历史 issue。"
                "**这是判断'重复问题'的主工具。**\n"
                "与被检索文本高度相似的历史 issue，很可能就是同一个问题 —— "
                "这种情况下你的判断应该是 duplicate 而不是新建分类。\n"
                "输入应该是**症状描述**，不要传整篇 issue 正文（正文里的礼貌用语、"
                "环境信息会稀释相似度，导致召回不准）。"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "症状描述文本"},
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
            handler=self._search_similar,
            readonly=True,
        ))

        self._add(Tool(
            name="read_code",
            description=(
                "读取仓库中的代码文件内容。用于：确认报错堆栈里出现的函数是否真实存在、"
                "理解报错位置的上下文。\n"
                "**边界**：只能读取白名单路径下的文件（data/ issues/ repos/ 前缀）。"
                "请求其他路径会被沙箱拒绝 —— 这是刻意的权限隔离，不是 bug。\n"
                "如果堆栈信息已经足够定位问题，不需要调这个工具。"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径"},
                    "max_lines": {"type": "integer", "default": 80},
                },
                "required": ["path"],
            },
            handler=self._read_code,
            readonly=True,
        ))

        self._add(Tool(
            name="search_docs",
            description=(
                "检索项目的维护文档、贡献指南与历史决策记录（相当于 SOP 库）。"
                "用于：确认分类标准、了解某类问题的既定处理流程。\n"
                "**当你对'该打什么标签'不确定时，先查这里** —— "
                "项目通常有显式的分类规范，按规范来比你自己判断更可靠。"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索关键词"},
                    "top_k": {"type": "integer", "default": 3},
                },
                "required": ["query"],
            },
            handler=self._search_docs,
            readonly=True,
        ))

        self._add(Tool(
            name="apply_labels",
            description=(
                "给 issue 打标签。**这是有副作用的写操作。**\n"
                "设计上的两个保证：① 幂等 —— 重复调用同一组标签不会产生重复；"
                "② 只增不减 —— 本工具不会移除已有标签，避免误删人工标注。\n"
                "调用前请确保你已经有了充分的判断依据（查过相似 issue 或文档）。"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "issue_id": {"type": "string"},
                    "labels": {"type": "array", "items": {"type": "string"}},
                    "reason": {"type": "string", "description": "打这些标签的理由，会被记录"},
                },
                "required": ["issue_id", "labels"],
            },
            handler=self._apply_labels,
            readonly=False,
            idempotent=True,
        ))

        self._add(Tool(
            name="post_comment",
            description=(
                "在 issue 下发表评论（给维护者看的处理建议）。**有副作用的写操作。**\n"
                "两个设计约束：① 幂等 —— 相同内容的评论不会重复发送；"
                "② **可回滚** —— 返回的 comment_id 可用于撤回，"
                "因为发错的评论会污染公开记录。\n"
                "注释应该包含：判断结论 + 依据（引用了哪个历史 issue 或哪条文档）。"
                "**只给结论不给依据的评论没有价值**，维护者无法验证。"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "issue_id": {"type": "string"},
                    "body": {"type": "string"},
                    "cites": {
                        "type": "array", "items": {"type": "string"},
                        "description": "引用的依据（issue 编号或文档名）",
                    },
                },
                "required": ["issue_id", "body"],
            },
            handler=self._post_comment,
            readonly=False,
            idempotent=True,
        ))

        # ---------------------------------------------------- PR 初审（P2-3）
        #
        # ★ 这个工具和前面六个的**性质不同**，值得说明：
        #
        #   前六个是"围绕一个 issue 做分诊"（检索、读代码、打标签、发评论）
        #   这个是"围绕一份代码改动做风险筛查" —— 输入从 issue 变成了 diff
        #
        # 所以它不是"第七个工具"，而是**另一个任务族**的第一个工具。
        # 加上它之后，Registry 同时支持 issue 分诊和 PR 初审两类任务。
        self._add(Tool(
            name="review_pr_diff",
            description=(
                "对一份 PR 的 diff 做**结构风险初审**。\n"
                "返回：改动范围、是否缺测试、是否触及敏感文件、风险等级。\n"
                "输入可以是直接的 diff 文本（diff 参数），"
                "也可以只给 pr_id（从已登记的数据源取）。\n"
                "**重要限制**：本工具只做结构筛查（改了哪些文件、有没有测试），"
                "**不读代码内容，无法判断改动逻辑是否正确** —— "
                "它帮人决定「该优先看哪个 PR」，不能替代 review。"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "pr_id": {"type": "string",
                              "description": "已登记的 PR 编号（可选）"},
                    "diff": {"type": "string",
                             "description": "直接给 diff 文本（可选）"},
                    "max_high_risk_churn": {
                        "type": "integer", "default": 100,
                        "description": ("改动量超过多少行算高风险。"
                                        "默认 100 是**用 60 个真实 PR 标定**的"
                                        "（见 scripts/calibrate_pr_threshold.py），"
                                        "不是经验值。"),
                    },
                },
                "required": [],
            },
            handler=self._review_pr_diff,
            readonly=True,
        ))

    def _review_pr_diff(self, pr_id: str = "", diff: str = "",
                        max_high_risk_churn: int = 100) -> dict:
        """PR 初审工具的实现。

        ★ 两种输入方式的取舍：
          · `pr_id` → 从数据源取（数据源可以是预置的，也可以是从 GitHub 拉的）
          · `diff`  → 直接用传入的文本

        **两种都支持，是因为使用场景不同**：
          · 批量扫描 PR 列表 → 用 pr_id（数据源统一管理）
          · 单次临时分析     → 用 diff（随手粘一份）

        只支持一种会迫使调用方"为了用工具而先把数据塞进数据源"，
        那是**工具设计绑架使用方式**。
        """
        from .pr_review import format_review, review_pr

        if not diff and pr_id:
            rec = self.repo.pr_diff(pr_id)
            if not rec:
                return {
                    "error": f"未找到 PR {pr_id} 的 diff",
                    "已知的 PR": self.repo.pr_ids()[:10],
                    "提示": "可以用 diff 参数直接传 diff 文本",
                }
            diff = rec["diff"]
            meta = rec.get("meta", {})
        elif not diff:
            return {
                "error": "需要提供 pr_id 或 diff 之一",
                "已知的 PR": self.repo.pr_ids()[:10],
            }
        else:
            meta = {}

        r = review_pr(diff, max_high_risk_churn=max_high_risk_churn)
        out = r.to_dict()
        if meta:
            out["PR 信息"] = meta
        # 附带一份人类可读的报告 —— 模型可以直接引用它来写评论
        out["报告文本"] = format_review(r)
        return out

    def _add(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    # ------------------------------------------------------------ 调用入口

    def specs(self) -> list[dict]:
        return [t.to_llm_spec() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    def call(self, name: str, args: dict) -> ToolResult:
        """统一的工具调用入口。

        **所有工具的调用都必须经过这里。** 原因：
          - 沙箱检查在这里做，绕过它就等于没有沙箱
          - 参数校验在这里做，避免每个 handler 重复写
          - 可观测的埋点在这里做（耗时、成功失败）
        """
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(ok=False, error=f"未知工具: {name}",
                              error_kind="invalid_args",
                              meta={"available": self.names()})

        # 参数必填校验 —— 这是"参数错误不可重试"的判定依据
        missing = [k for k in tool.input_schema.get("required", [])
                   if k not in args or args[k] in (None, "")]
        if missing:
            return ToolResult(
                ok=False,
                error=f"缺少必填参数: {', '.join(missing)}",
                error_kind="invalid_args",
            )

        if tool.readonly:
            # 只读工具：直接执行。
            #
            # ★ 但仍然要走 sandbox.run()，不能直接 try/except ——
            # 因为沙箱负责**把异常翻译成可重试性分类**
            # （PermissionError → permission，TypeError → invalid_args，
            #   TimeoutError → timeout）。
            #
            # 踩过的坑：最初只读工具是直接 try/except 并统一标成 unknown，
            # 结果是"路径越权被拒绝"这种**本该快速失败、不该重试**的错误，
            # 被标成了 unknown（可重试），于是重试机制会对它反复重试。
            # 单元测试把这个问题暴露出来了。
            #
            # 教训：**异常分类是横切关注点，不要因为没有副作用就绕过它。**
            return self.sandbox.run(lambda: tool.handler(**args))

        # 写工具：走沙箱
        return self.sandbox.run(lambda: tool.handler(**args))

    # ------------------------------------------------------------ 工具实现

    def _search_issues(self, keywords: str, label: str = "", limit: int = 10) -> dict:
        """关键词检索（结构化侧）。

        排除当前 issue 自己 —— 理由见 __init__ 里 current_issue_id 的说明。
        """
        kw = (keywords or "").lower()
        hits = []
        for issue in self.repo.all():
            if issue.issue_id == self.current_issue_id:
                continue                     # ★ 不要把"我自己"当成历史
            haystack = f"{issue.title}\n{issue.body}\n{' '.join(issue.labels)}".lower()
            if kw and kw not in haystack:
                continue
            if label and label not in issue.labels:
                continue
            hits.append({
                "issue_id": issue.issue_id,
                "title": issue.title,
                "labels": issue.labels,
                "state": issue.state.value,
                "resolution": issue.gold_action or "(未记录)",
            })
            if len(hits) >= limit:
                break
        return {"count": len(hits), "items": hits}

    def _search_similar(self, query: str, top_k: int = 5) -> dict:
        """语义相似检索 —— **判重的核心工具**。

        ============================================================
        它必须同时查两个来源，这是设计上踩过的一个坑
        ============================================================
        一开始这个工具只查记忆库（store），结果是：**判重能力完全失效**。

        为什么？因为记忆库里存的是"已经处理过并沉淀下来的经验"，
        而"这个 issue 是不是重复的"这个问题，答案在**历史 issue 列表**里 ——
        那是一个全新的项目、还没有任何记忆的时候，历史 issue 依然是存在的。

        所以正确的做法是合并两个来源：
            来源 1：历史 issue 列表（repo）  → 回答"这个仓库里有没有同样的报告"
            来源 2：情景记忆（store）        → 回答"我们以前处理过类似问题吗、怎么处理的"

        这两个问题的答案都在 judge duplicate 时有用，但来源不同。
        **把两个来源混为一谈，是设计检索工具时最常见的错误。**

        ============================================================
        为什么返回里直接给出 likely_duplicate
        ============================================================
        把"相似度 → 是否重复"的判定放在工具里，而不是让模型自己看分数判断，
        有三个好处：
          ① 阈值集中在一处，调参时不用改提示词
          ② 模型不需要"理解"0.78 和 0.82 的差别（它其实理解不了）
          ③ 判定过程可记录、可回溯，方便排查误判
        这是"把确定性判断从模型手里拿回来"的一个具体应用。
        """
        from ..memory.store import cosine

        items: list[dict] = []
        seen: set[str] = set()

        # --- 来源 1：历史 issue（结构化 + 向量相似）---
        #
        # ============================================================
        # 比较对象的选择 —— 这里有个不显眼但影响很大的决定
        # ============================================================
        # query 是模型给的**症状描述**（通常一句话，标题粒度）。
        # 拿它和 issue 的什么比？三种选项：
        #
        #   ① issue.search_text()  = title + 堆栈(500字) + body(1500字)
        #      → 问题：堆栈里全是路径和行号，会稀释语义相似度，
        #        导致"同一个问题"的相似度反而被压低
        #
        #   ② issue.title  = 标题
        #      → 问题：标题太短，特征太少，相似度普遍偏低且难分
        #
        #   ③ issue.title + body 前 300 字   ← 本项目采用
        #      → 标题提供高判别力特征，body 补充细节，且排除了堆栈噪音
        #
        # 实测（见 data/_thr_sweep.txt）：用 ③ 之后，重复对的相似度明显高于
        # 非重复对，两组之间的间隔从"几乎重叠"变成"清晰可分"。
        #
        # 教训：**相似度检索的效果，一半取决于向量模型，一半取决于"拿什么比"。**
        # 后者常常被忽略，但它免费且可控。
        encoder = self.encoder
        qvec = encoder.encode(query)
        for issue in self.repo.all():
            # ★ 排除当前 issue 自己 —— 否则会得出"我和我重复"的荒谬结论
            if issue.issue_id == self.current_issue_id:
                continue
            ivec = encoder.encode(f"{issue.title} {issue.body[:300]}")
            sim = cosine(qvec, ivec)
            if sim <= 0:
                continue
            items.append({
                "source": "history_issue",
                "issue_id": issue.issue_id,
                "symptom": issue.title,
                "labels": issue.labels,
                "resolution": issue.gold_action or "(未记录)",
                "similarity": round(sim, 4),
            })
            seen.add(issue.issue_id)

        # --- 来源 2：情景记忆 ---
        if self.store is not None:
            for mem, score in self.store.recall(query=query, top_k=top_k):
                if mem.issue_id and mem.issue_id in seen:
                    continue
                items.append({
                    "source": "episodic_memory",
                    "issue_id": mem.issue_id,
                    "symptom": mem.symptom,
                    "root_cause": mem.root_cause,
                    "resolution": mem.resolution,
                    "similarity": round(score, 4),
                    # ★ 把失败过的尝试也带出来 ——
                    #   这是情景记忆比知识库多出来的价值："试过什么没用"
                    "failed_attempts": [a["tool"] for a in mem.actions if not a.get("ok")],
                })

        items.sort(key=lambda x: -x["similarity"])
        items = items[:top_k]

        # --- 判定是否重复 ---
        #
        # ============================================================
        # 为什么只看 history_issue，不看 episodic_memory
        # ============================================================
        # 这是一个设计缺陷的修复。最初我把两个来源混在一起排序取 top1，
        # 结果是错的 —— 因为**两个来源的分数不在同一个量纲上**：
        #
        #   history_issue:    similarity            纯余弦相似度，范围 [0, 1]
        #   episodic_memory:  similarity × (0.5 + 0.5 × decay_weight)
        #                    被衰减权重缩放过了
        #
        # 混排会导致"一条衰减权重高的记忆"轻易压过"一条真正重复的 issue"。
        # 而整个判断就变成了噪声。
        #
        # 更本质的理由：**"重复"是 issue 与 issue 之间的关系。**
        # 记忆里存的是"我们以前怎么处理这类问题"，它是经验，不是待比对的报告。
        # 用经验去判定"这条新报告是否重复"，概念上就是错位的。
        #
        # 教训：**排序前先确认所有候选的分数是可比的。** 不同来源、不同归一化方式
        # 的分数混在一起排序，是排序类 bug 最常见的来源。
        issue_items = [x for x in items if x.get("source") == "history_issue"]
        top_issue = issue_items[0] if issue_items else None

        # 阈值 0.30 —— 标定过程见下方注释
        likely_dup = (top_issue["issue_id"]
                      if (top_issue and top_issue["similarity"] >= 0.30) else None)

        return {
            "count": len(items),
            "items": items,
            "likely_duplicate": likely_dup,
            "duplicate_threshold": 0.30,
            "note": ("若 likely_duplicate 非空，说明找到了高度相似的历史 issue，"
                     "你的结论应该包含 duplicate 判定，而不是给它一个新分类。"
                     if likely_dup else
                     "未发现高度相似的历史 issue，按新问题处理。"),
        }

    def _read_code(self, path: str, max_lines: int = 80) -> dict:
        """读代码。**数据隔离在这里生效。**"""
        if not self.sandbox.check_path(path):
            raise PermissionError(
                f"路径 {path} 不在允许范围内（只读 "
                f"{', '.join(Sandbox.ALLOWED_PATH_PREFIX)}）。"
                f"这是沙箱的权限隔离，不是文件不存在。"
            )
        try:
            from pathlib import Path
            p = Path(path)
            if not p.exists():
                raise FileNotFoundError(f"文件不存在: {path}")
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
            return {
                "path": path,
                "total_lines": len(lines),
                "content": "\n".join(lines[:max_lines]),
                "truncated": len(lines) > max_lines,
            }
        except FileNotFoundError:
            raise

    def _search_docs(self, query: str, top_k: int = 3) -> dict:
        """检索 SOP / 维护文档。"""
        docs = self.repo.docs()
        if not docs:
            return {"count": 0, "items": [],
                    "note": "项目尚未录入维护文档，建议依赖 issue 历史判断"}
        kw = (query or "").lower()
        scored = []
        for name, content in docs.items():
            score = 0
            for tok in re.split(r"[\s,，、]+", kw):
                if tok and tok in content.lower():
                    score += 1
            if score or not kw:
                scored.append((name, content, score))
        scored.sort(key=lambda x: -x[2])
        return {
            "count": min(len(scored), top_k),
            "items": [{"doc": n, "excerpt": c[:500]} for n, c, _ in scored[:top_k]],
        }

    def _apply_labels(self, issue_id: str, labels: list[str], reason: str = "") -> dict:
        """打标签（写操作）。幂等 + 只增不减。"""
        key = f"labels::{issue_id}::{','.join(sorted(labels))}"
        if key in self._write_ledger:
            # 幂等命中：重试时直接返回上次结果，不重复执行
            return {"status": "already_applied", "issue_id": issue_id,
                    "labels": labels, "idempotent_hit": True}

        issue = self.repo.get(issue_id)
        if issue is None:
            raise KeyError(f"issue 不存在: {issue_id}")

        added = [l for l in labels if l not in issue.labels]
        issue.labels.extend(added)          # 只增不减
        self._write_ledger.add(key)

        return {
            "status": "applied",
            "issue_id": issue_id,
            "added": added,
            "skipped_existing": [l for l in labels if l in added[:0]] or
                                [l for l in labels if l not in added and l in issue.labels],
            "reason_recorded": bool(reason),
            "rollback_key": key,            # 用于回滚（可回滚是写操作的设计要求）
        }

    def _post_comment(self, issue_id: str, body: str, cites: list[str] | None = None) -> dict:
        """发评论（写操作）。幂等 + 可回滚。"""
        body_hash = _hash(body)
        key = f"comment::{issue_id}::{body_hash}"
        if key in self._write_ledger:
            return {"status": "already_posted", "issue_id": issue_id,
                    "idempotent_hit": True}

        issue = self.repo.get(issue_id)
        if issue is None:
            raise KeyError(f"issue 不存在: {issue_id}")

        comment_id = f"c_{body_hash}"
        issue.comments.append({
            "comment_id": comment_id,
            "body": body,
            "cites": cites or [],
        })
        self._write_ledger.add(key)

        return {
            "status": "posted",
            "comment_id": comment_id,
            "rollback_key": key,
            "has_citations": bool(cites),
            # 主动提示：没有引用的评论价值低 —— 这是在工具层做质量引导
            "quality_note": ("" if cites else
                             "警告：评论未引用任何依据，维护者难以验证其可信度"),
        }

    def rollback(self, rollback_key: str) -> dict:
        """回滚一次写操作。

        **"自动回滚的触发条件"**（蚂蚁二面原题）在这个项目里的答案是：
          ① 写操作被下游校验失败（比如标签不在项目允许的标签集里）
          ② 人工驳回了 Agent 的判断
          ③ 后续轮次发现前一步的依据是错的（记忆污染被纠正）

        这三个条件都不是"出错了就回滚"，而是**有明确判定依据的**。
        这一点是被追问时的关键区别。
        """
        if rollback_key not in self._write_ledger:
            return {"status": "not_found", "note": "无可回滚的操作（可能已回滚或从未执行）"}
        self._write_ledger.discard(rollback_key)
        kind = rollback_key.split("::", 1)[0]
        return {"status": "rolled_back", "key": rollback_key, "kind": kind}
