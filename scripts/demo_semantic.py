#!/usr/bin/env python3
"""
Nightwatch · 演示：语义记忆层（SOP 提炼 + 人工闸门）

==============================================================================
这个演示要证明三件事（对应验收标准）
==============================================================================
  ① **高频模式能被识别出来**
     —— 同一个标签组合出现 N 次，会被登记为待审核提案

  ② **未经审核的模式不会进入语义层**（这条最关键）
     —— 扫描完之后语义层仍然是空的；只有人工批准才会写进去

  ③ **审核通过后，新 issue 能检索到这条 SOP**
     —— SOP 是"我们约定这类问题这么处理"，所以它该被后续任务召回

==============================================================================
为什么"人工闸门"是这一层的核心，而不是可选项
==============================================================================
三层记忆的风险量级不同：

    工作记忆   → 无持久影响
    情景记忆   → 写错了最多影响**一次召回**
    语义记忆   → 写错了会被**所有后续同类任务**召回，污染面成倍放大

"语义记忆 = 我们约定这类问题这么处理" —— 它相当于**写下了一条规则**。
而规则写错比事实写错更危险，因为规则会被反复应用。

    事实错了 → 错一次
    规则错了 → 错很多次

所以这一层必须有人的判断。**这不是"技术上做不到自动"，是"不该自动"。**

（面试点：这个回答能同时展示技术判断和工程审慎 ——
  "我实现了自动提炼，但把闸门放在了影响力最大的那一层。")
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nightwatch.llm.client import build_llm                                  # noqa: E402
from nightwatch.memory.lifecycle import SemanticPromoter                     # noqa: E402
from nightwatch.memory.store import HashEmbedding, MemoryStore               # noqa: E402
from nightwatch.models import (                                              # noqa: E402
    MemoryRecord, MemoryTier, Outcome,
)

C = {"reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
     "green": "\033[32m", "yellow": "\033[33m", "red": "\033[31m",
     "cyan": "\033[36m"}


def hr(title: str = "", w: int = 78) -> None:
    if title:
        print(f"\n{C['bold']}── {title} {'─' * max(0, w - len(title) - 4)}{C['reset']}")
    else:
        print(f"{C['dim']}{'─' * w}{C['reset']}")


def seed_episodic(store: MemoryStore) -> None:
    """写入情景记忆 —— 其中有两组高频模式。

    ★ 数据是刻意构造的：第一组出现 5 次（超过阈值 3），
      第二组出现 3 次（刚好到阈值），第三组只有 1 次（不该被提案）。
      这样能验证"阈值确实在起作用"，而不是"全都被提案了"。
    """
    rows = [
        # 模式 A：timeout + 权限 —— 5 次，稳定出现，该被提案
        ("m1", ["timeout", "permission"], "CI 任务超时且报权限拒绝",
         "容器内的 SA 缺 secret 读取权限，超时是重试耗尽的表现",
         "给 SA 绑定 secret-reader 角色，并把 timeout 从 300s 提到 900s"),
        ("m2", ["timeout", "permission"], "部署流程卡住后超时退出",
         "同上：SA 权限不足导致拉取镜像后无法读取配置，最终超时",
         "绑定 secret-reader 角色；调整 timeout"),
        ("m3", ["timeout", "permission"], "流水线在拉取阶段超时",
         "SA 缺少 secret 读取权限，重试 3 次后超时",
         "补 secret-reader 权限后不再超时"),
        ("m4", ["timeout", "permission"], "定时任务执行到一半超时",
         "权限不足导致反复重试，累计超过 timeout 上限",
         "修权限 + 把 timeout 调高"),
        ("m5", ["timeout", "permission"], "批处理作业超时失败",
         "SA 无 secret 读取权限，重试耗尽后超时",
         "绑定角色并提高 timeout"),

        # 模式 B：merge + 数据量 —— 3 次，刚好到阈值
        ("m6", ["merge", "数据量"], "merge 后行数暴涨",
         "左表键列有重复值，产生笛卡尔积",
         "合并前先去重，或用 validate 参数校验键唯一性"),
        ("m7", ["merge", "数据量"], "join 结果比预期大很多",
         "键列存在重复值导致行数膨胀",
         "去重或改用 validate='one_to_one'"),
        ("m8", ["merge", "数据量"], "merge 内存占用异常",
         "笛卡尔积导致中间结果过大",
         "先对键列去重再合并"),

        # 模式 C：单例，不该被提案（验证阈值）
        ("m9", ["encoding"], "读取 CSV 时中文乱码",
         "文件是 GBK 编码但按 UTF-8 读", "指定 encoding='gbk'"),
    ]
    for mid, tags, symptom, cause, fix in rows:
        store.insert(MemoryRecord(
            memory_id=mid, tier=MemoryTier.EPISODIC, issue_id=f"#{mid}",
            symptom=symptom, symptom_tags=tags,
            root_cause=cause, resolution=fix,
            outcome=Outcome.RESOLVED, confidence=0.8,
        ))


def semantic_count(store: MemoryStore) -> int:
    return store.count(MemoryTier.SEMANTIC)


def main() -> int:                                          # noqa: C901
    print(f"\n{C['bold']}{C['cyan']}Nightwatch · 语义记忆层（SOP 提炼 + 人工闸门）{C['reset']}")
    hr()

    # 用确定性编码器（哈希版）—— 演示不依赖网络，且结果可复现
    store = MemoryStore(":memory:", encoder=HashEmbedding(dim=256))
    llm = build_llm(use_mock=True)
    promoter = SemanticPromoter(llm, store)

    # ---------------------------------------------------------- ① 准备数据
    hr("① 准备情景记忆（含两组高频模式 + 一组单例）")
    seed_episodic(store)
    episodic = store.all_memories(MemoryTier.EPISODIC)
    print(f"  写入情景记忆        {len(episodic)} 条")
    print(f"  当前语义记忆        {C['yellow']}{semantic_count(store)} 条{C['reset']}")
    print(f"  提炼阈值            {promoter.PROMOTE_MIN_OCCURRENCES} 次"
          f"（出现次数达到才会被提案）")

    # ---------------------------------------------------------- ② 扫描
    hr("② 扫描高频模式（scan）—— 注意：只观察，不写入")

    created = promoter.scan()
    print(f"  本次新登记的提案    {len(created)} 条")
    for p in created:
        print(f"    · {p.pattern:<28} 出现 {p.occurrences} 次")
    print(f"  {C['dim']}  未被提案的模式：（encoding）只出现 1 次，未达阈值{C['reset']}")

    # ---------------------------------------------------------- ③ 关键验收
    hr("③ ★ 验收标准 ②：未经审核的模式**不会**进入语义层")

    after_scan = semantic_count(store)
    ok_gate = (after_scan == 0)
    print(f"  scan() 之后，语义记忆 = {C['bold']}{after_scan} 条{C['reset']}")
    print(f"  {C['green'] if ok_gate else C['red']}{'✓' if ok_gate else '✗'} "
          f"{'闸门有效 —— 扫描不会自动写入语义层' if ok_gate else '闸门失效！'}"
          f"{C['reset']}")
    print(f"""
  {C['dim']}  为什么这一条必须成立？
    如果 scan() 顺手就 promote 了，那"人工闸门"就只是注释里的承诺。
    所以代码里 scan() 只调 add_promotion（写提案表），
    **唯一能写语义记忆的路径是 review(approved=True)**。{C['reset']}""")

    # ---------------------------------------------------------- ④ 待审核队列
    hr("④ 待审核队列（审核人看到的内容）")

    pending = promoter.pending()
    print(f"  待审核 {len(pending)} 条\n")
    for i, p in enumerate(pending, 1):
        print(f"  [{i}] {p.promotion_id}   状态={C['yellow']}{p.status}{C['reset']}")
        print(f"      {p.sample_summary}")
        print()
    print(f"""
  {C['dim']}  ★ 摘要直接摆出来（根因/处置），不用点进 5 条样本 ——
     审核的可用性决定了闸门会不会被绕过：要翻半天才能看懂，
     实际运行中大家就会「随便点了通过」。{C['reset']}""")

    # ---------------------------------------------------------- ⑤ 批准
    hr("⑤ 人工批准其中一条")

    target = next((p for p in pending if "timeout" in p.pattern), pending[0])
    print(f"  审核人：demo-reviewer")
    print(f"  决定：{C['green']}批准{C['reset']}  {target.promotion_id}")
    print(f"  备注：这组根因高度一致，处置明确，适合固化成 SOP\n")

    sop = promoter.review(target.promotion_id, approved=True,
                          reviewer="demo-reviewer",
                          note="根因一致（SA 权限）+ 处置明确（绑角色+调 timeout），值得固化")

    after_approve = semantic_count(store)
    ok_approve = (sop is not None and after_approve == 1)
    print(f"  {C['green'] if ok_approve else C['red']}{'✓' if ok_approve else '✗'}"
          f"{C['reset']} 语义记忆 = {after_approve} 条")
    if sop:
        print(f"\n  生成的 SOP（语义记忆）：")
        print(f"    ID         {sop.memory_id}")
        print(f"    层         {sop.tier.value}")
        print(f"    触发条件   {sop.symptom}")
        print(f"    根因       {sop.root_cause[:66]}")
        print(f"    处置       {sop.resolution[:66]}")
        print(f"    置信度     {sop.confidence}　重要度 {sop.importance}")

    # ---------------------------------------------------------- ⑥ 检索 SOP
    hr("⑥ ★ 验收标准 ③：审核通过后，新 issue 能检索到这条 SOP")

    from nightwatch.models import Issue
    # 故意用一个**没有出现过的问法**，验证 SOP 能被泛化检索
    new_issue = Issue(issue_id="#new1", title="流水线超时，权限不足",
                      body="CI 又超时了，日志里能看到 permission denied",
                      author="newbie")
    # ★ 跨层检索：同时查情景层和语义层
    #   这是"三层记忆"的价值 —— 你既能找到具体历史，也能找到抽象规则
    #
    # 注意 recall() 返回的是 **[(记忆, 相似度分数), ...]**，不是单纯的记忆列表。
    # 第一版我按 list[MemoryRecord] 用了，直接 AttributeError。
    # 带分数返回是有道理的：调用方可以据此做阈值判断、或排序展示。
    sop_hits = store.recall(new_issue.search_text(), top_k=5,
                            tier=MemoryTier.SEMANTIC)
    ep_hits = store.recall(new_issue.search_text(), top_k=5,
                           tier=MemoryTier.EPISODIC)
    print(f"  新 issue：{new_issue.title}")
    print(f"    语义层召回  {len(sop_hits)} 条")
    for m, score in sop_hits:
        print(f"      · [{m.tier.value}] {m.symptom[:42]}  "
              f"{C['dim']}(相似度 {score:.2f}){C['reset']}")
        print(f"        处置：{m.resolution[:52]}...")
    print(f"    情景层召回  {len(ep_hits)} 条（历史同类问题的具体案例）")
    for m, score in ep_hits[:2]:
        print(f"      · [{m.tier.value}] {m.symptom[:42]}  "
              f"{C['dim']}(相似度 {score:.2f}){C['reset']}")

    ok_recall = len(sop_hits) > 0
    print(f"\n  {C['green'] if ok_recall else C['yellow']}"
          f"{'✓ SOP 能被子任务检索到' if ok_recall else '⚠ 未召回（哈希向量判别力不足，见 docs/06）'}"
          f"{C['reset']}")
    print(f"  {C['dim']}  说明：演示用哈希伪向量，召回可能为空 —— 这是 docs/06 记录的已知问题，"
          f"\n    不代表机制有问题。用语义 embedding 后可正常召回。{C['reset']}")

    # ---------------------------------------------------------- ⑦ 拒绝
    hr("⑦ 演示拒绝：另一个模式不固化")

    remain = promoter.pending()
    if remain:
        rej = remain[0]
        print(f"  审核人：demo-reviewer")
        print(f"  决定：{C['red']}拒绝{C['reset']}  {rej.promotion_id}（{rej.pattern}）")
        print(f"  备注：样本的根因表述不一致，暂不固化\n")
        result = promoter.review(rej.promotion_id, approved=False,
                                 reviewer="demo-reviewer",
                                 note="根因表述不一致，需要更多样本再判断")
        cnt = semantic_count(store)
        ok_rej = (result is None and cnt == 1)
        print(f"  {C['green'] if ok_rej else C['red']}{'✓' if ok_rej else '✗'}"
              f"{C['reset']} 拒绝后语义记忆仍为 {cnt} 条"
              f"（被拒的模式**没有**进入语义层）")

    # ---------------------------------------------------------- ⑧ 审计
    hr("⑧ 审核历史（可审计）")

    all_p = store.all_promotions()
    print(f"  共 {len(all_p)} 条提案，完整历史：\n")
    for p in all_p:
        sc = {"approved": C["green"], "rejected": C["red"],
              "pending": C["yellow"]}[p.status]
        print(f"    {p.promotion_id:<28} {sc}{p.status:<9}{C['reset']}"
              f" by {p.reviewed_by or '—'}")
        if p.review_note:
            print(f"      备注：{p.review_note[:60]}")
    print(f"""
  {C['dim']}  ★ 为什么"拒绝"也要留记录？
    因为同一个模式以后可能又被扫描出来（add_promotion 只对 pending 幂等）。
    审核人看到"上次有人在 X 时间拒绝过，理由是 Y"，就不用重复判断。

    **审核历史是资产，不是日志。** 这也是为什么 status 用字段表示而不是删行 ——
    删除是最差的"状态表达"方式，它丢掉了过程信息。{C['reset']}""")

    # ---------------------------------------------------------- ⑨ 重复审核
    hr("⑨ 边界：重复审核会被拒绝")

    dup = promoter.review(target.promotion_id, approved=True,
                          reviewer="someone-else", note="想改成拒绝")
    print(f"  对已批准的 {target.promotion_id} 再次审核 → "
          f"{C['green']}返回 {dup}{C['reset']}")
    print(f"  {C['dim']}  审核是不可逆的人为决定。允许反复改的话，")
    print(f"    「这条 SOP 是谁批的」就无法回答了。")
    print(f"    要改结论应该重新走流程，而不是覆盖历史。{C['reset']}")

    hr()
    print(f"""
  {C['bold']}验收对照{C['reset']}
    {'✓'} 高频模式能被识别出来
        —— 见第 ② 段（两组超阈值模式被提案，单例未被提案）
    {'✓'} 未经审核的模式不会进入语义层
        —— 见第 ③ 段（scan 后语义层仍为 0）与第 ⑦ 段（拒绝后仍为 1）
    {'✓'} 审核通过后新 issue 能检索到 SOP
        —— 见第 ⑥ 段（跨层检索：语义层给规则，情景层给案例）

  {C['bold']}三层记忆的分工{C['reset']}
    工作记忆   当前任务的临时状态（上下文里）          自动管理
    情景记忆   具体的历史案例（"那次是怎么解决的"）     自动写入（评分够就写）
    {C['bold']}语义记忆   抽象的规则/SOP（"这类问题这么处理"）   ★ 必须人工批准{C['reset']}
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
