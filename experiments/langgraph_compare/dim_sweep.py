#!/usr/bin/env python3
"""LangGraph 对照实验 · 嵌入维度的单变量对照

    python experiments/langgraph_compare/dim_sweep.py

==============================================================================
这个脚本要回答什么
==============================================================================
本次对照之前，`shared.py` 里写的是 `EMBED_DIM = 1024`，而主项目 `HashEmbedding`
的默认值是 **256**（`src/nightwatch/memory/store.py:199`，
`def __init__(self, dim: int = 256)`）。

于是有个必须回答的问题：**为什么实验不用主项目的默认值？**

`shared.py` 原来的注释写的是"**为什么从默认的 64 提到 1024**" ——
**这句话是错的**：64 不是主项目默认值，是实验脚本自己写死的参数
（`setup_tools()` 早期版本写 `HashEmbedding(dim=64)`）。
把"我自己拍的一个数"说成"默认值"，会让读者以为主项目就是这么窄的 ——
而它其实是 256。**这是"事实陈述错误"，比参数选错更严重。**

跑完本脚本后结论是 **256 分得开（空档 +0.061）**，所以 `shared.py` 已改为
`EMBED_DIM = 256` 对齐主项目默认值。也就是说：**1024 是"改之前"的值，
256 是"改之后"的值**，本脚本保留 1024 那一列是为了让对照可复现。

==============================================================================
方法：只改 dim 一个变量
==============================================================================
两个维度下，**记忆库、样本、编码算法、召回逻辑全部一字不改** ——
唯一变化的就是 `HashEmbedding(dim=...)`。

对照实验的第一原则是"只改一个变量"。这里 `dim` 就是那一个变量。

==============================================================================
要看的三组数字
==============================================================================
① **相关组 TOP1 区间** —— "该命中记忆"的样本（#901~#904）拿到的最高分
② **无关组 TOP1 区间** —— "不该命中"的样本（#906/#907 + 探针）拿到的最高分
③ **空档宽度** —— `min(相关组) - max(无关组)`

**空档 > 0 才叫"分得开"**；空档 ≤ 0 意味着两组的分布**重叠**，
此时**任何阈值都切不开** —— 因为阈值只能切分分布，不能分离重叠的分布。

★ 另外必须同时看一件事：**分数尺度会随 dim 整体变化**。
dim 提高 → 每个桶分到的质量变稀 → 所有余弦分数**普遍下降**
（#901 从 0.473 → 0.366）。所以"区分度出现"里**含有尺度变化的成分**，
不全是碰撞消除带来的。这一点在结论里必须说明，
否则会把"整体下降"误读成"碰撞被消除了"。

==============================================================================
★ 关于"空档宽度"的判读标准
==============================================================================
本脚本按空档宽度分三档：

    gap > 0.10      分得开（余量充足）  —— 改措辞不容易翻
    gap > 0        勉强分开（余量小）  —— **真的悬**，改措辞会翻
    gap ≤ 0        重叠 —— 任何阈值都切不开

`dim=256` 落在**第二档**（+0.061）。这不是"通过"，
而是"刚好够" —— 所以余量守卫（`tests/test_smoke.py` 里的
`worst > 0.02`）是**贴着下限**的，不是形式主义。

==============================================================================
破坏性验证（2026-10-11 第二轮，阈值 0.32 / dim=256）
==============================================================================
**"一条永远通过的断言，比没有断言更糟。"** 所以每条守卫都反向验过 ——
下面的表是实测输出，不是设计意图：

    破坏方向                          期望    实际    说明
    ----------------------------------------------------------------
    阈值 → 0.40（漏召回）              报错    报错    worst 转负 -0.045
    阈值 → 0.20（误召回）              报错    报错    worst 转负 -0.093
    无关样本措辞 → 靠向记忆             报错    报错    两组重叠，0.355 < 0.400
    无关样本正文 → 靠向记忆             报错    报错    两组重叠，0.355 < 0.366
    探针 → 加上 h1 的词                报错    报错    两组重叠，0.355 < 0.358
    阈值 → 0.324（严格中点）            通过    通过    worst +0.031
    阈值保持 0.32                      通过    通过    worst +0.027
    EMBED_DIM → 1024（脱离主项目）      报错    报错    assertEqual 拦住

★ 注意第 3 例与第 4/5 例的区别 —— **它们报错的信息不一样，暴露了两种失效**：

    第一类：#906 加「超时」→ 分数升到 0.300/0.329，
            仍**小于** 0.355 → 两组不重叠，走 `worst > 0.02` 报错。
            （余量被吃掉，但分布还分得开 —— **更隐蔽的那种**）
    第二类：#907 整条改成 h4 措辞 → 升到 **0.400**，
            **超过** 0.355 → 两组重叠，走 `low_hit > high_miss` 报错。
            （边界被彻底击穿）

**守卫要能同时抓到这两类** —— 只断言"分得开"会漏掉第一类，
只断言"有余量"措辞上不精确。所以两条都要留。

⚠️ 一个**差点踩进去**的坑：做破坏验证时如果想"只改一处、快速试"，
   容易把探针或样本的分数**在隔离环境里单独测**。那样量出来的数
   **不可比** —— 因为它没有和完整的记忆库+其余样本一起参与召回，
   top-1 命中的对象都不一样。本次差点把 #907 的 0.380（隔离测）
   当成真实值，而它在完整样本集里其实只有 0.287（**根本没击穿边界**）。
   **破坏验证必须在和生产同一套口径下跑**（这里是完整 store + 完整样本集），
   否则验的是另一个东西。这和"共用要复用实现"是同一条原则的另一面。

实测过的破坏方向（把"不命中"样本的措辞往记忆靠一点点）：

    #906 措辞                         TOP1    后果
    原版「时区转换结果偏移了 8 小时」   0.286   不命中 ✓
    加个「超时」                       ~0.300  越过旧阈值 0.30 →
                                              #906 变成"命中"，
                                              循环覆盖率 6/18 → 3/18
    更像 h1（加"测试阶段超时"）         ~0.329  仍有报错（余量不足）

**注意反方向不会触发问题**：把"该命中"的样本改远一点，
余量反而变大 —— 因为此时"不命中组最高"换成了更低的样本。
**所以这个边界是单侧的**：它只对"无关样本变热"敏感。
加样本时要特别小心的是**新样本的措辞会不会靠近已有记忆**，
而不是"已有样本够不够突出"。

==============================================================================
★ 阈值与 dim 是**绑定**的，不能单独改
==============================================================================
本脚本会打印"当前阈值落在空档里吗"那张表 —— 它就是用来提醒这件事的。
`MIN_RECALL_SIMILARITY` 依赖当前 `EMBED_DIM` 下的空档中点，
**换了 dim 就必须重取中点**（详见 `shared.py` 里该常量的注释）。
换 dim 的正确顺序：改 dim → 跑本脚本看空档 → 取新中点 → 重跑 compare.py。
**顺序反了（先调阈值）就会掉进"阈值怎么调都不对"的坑。**
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

from nightwatch.memory.store import HashEmbedding, MemoryStore   # noqa: E402
from nightwatch.models import MemoryRecord, Outcome              # noqa: E402

from shared import PROBES, _seed_memories, sample_issues          # noqa: E402

C = {"reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
     "green": "\033[32m", "yellow": "\033[33m", "red": "\033[31m",
     "cyan": "\033[36m"}

#: 要对比的维度。
#:  64   = 实验脚本早期写死的值（**不是主项目默认值**）
#:  256  = **主项目 HashEmbedding 的真实默认值**（store.py:199）→ 本次已改用这个
#:  1024 = 改之前实验用的值（保留在对照里，便于复现结论）
DIMS = (64, 256, 1024)

# ★ `PROBES` 已上提到 `shared.py` —— 见那里的说明。
#   原因：原来这份探针定义只在本文件里，测试看不见它，
#   于是测试量出来的余量（只用标定样本）比真实值（含探针）乐观一个数量级。
#   现在两边从**同一个常量**取，口径不可能再分叉。
#   **"共用"要复用实现，不要各写一份** —— 本次是同一个坑的又一遍。


def eval_dim(dim: int, issues: list) -> dict:
    """在给定维度下，测出三组数字。

    ★ 每换一个维度都**重建记忆库** —— 因为记忆的 embedding 是在
      `insert()` 时用当时的 encoder 算出来的。复用旧库会拿到旧维度的向量，
      测出来的东西和维度无关（这个坑很隐蔽：分数会因为"算的是旧向量"而看起来正常）。
    """
    store = MemoryStore(":memory:", encoder=HashEmbedding(dim=dim))
    for m in _seed_memories():
        store.insert(m)

    def top1(text: str) -> float:
        hits = store.recall(text, top_k=1)
        return hits[0][1] if hits else 0.0

    # --- ① 相关组：应该命中记忆的样本 ---
    related = []
    for iss in issues:
        if iss.issue_id in ("#901", "#902", "#903", "#904"):
            related.append((iss.issue_id, top1(iss.search_text())))

    # --- ② 无关组：不该命中的样本 + 未参与标定的探针 ---
    unrelated = []
    for iss in issues:
        if iss.issue_id in ("#906", "#907"):
            unrelated.append((iss.issue_id, top1(iss.search_text())))
    for title, body in PROBES:
        unrelated.append((f"探针:{title[:6]}", top1(f"{title}\n{body}")))

    store.close()

    rel_scores = [s for _, s in related]
    unrel_scores = [s for _, s in unrelated]
    gap = min(rel_scores) - max(unrel_scores)
    return {"dim": dim, "related": related, "unrelated": unrelated,
            "min_rel": min(rel_scores), "max_unrel": max(unrel_scores),
            "gap": gap, "all_scores": rel_scores + unrel_scores}


def main() -> int:
    issues = sample_issues()
    results = [eval_dim(d, issues) for d in DIMS]

    print(f"\n{C['bold']}{C['cyan']}嵌入维度单变量对照 —— "
          f"只改 dim，其余全部共用{C['reset']}")
    print(f"{C['dim']}{'─' * 76}{C['reset']}")
    print(f"""
  变量          HashEmbedding(dim=...)
  固 定  {C['dim']}记忆库(h1~h4) / 样本(#901~#907) / 编码算法 / 召回逻辑
         全部一字不改 —— 只改维度这一个值{C['reset']}
  样本          相关组 4 个（#901~#904，应该命中）
                无关组 {len(results[0]['unrelated'])} 个（#906/#907 + {len(PROBES)} 个未标定探针）
""")

    # ---------------------------------------------------------- 逐维度明细
    for r in results:
        d = r["dim"]
        note = ""
        if d == 256:
            note = f"  {C['green']}← 主项目默认值，也是当前实验值{C['reset']}"
        elif d == 1024:
            note = f"  {C['dim']}← 改之前实验用的值（仅供对照）{C['reset']}"
        else:
            note = f"  {C['red']}← 早期实验写死的值（**不是**默认值）{C['reset']}"
        print(f"\n{C['bold']}dim = {d}{C['reset']}{note}")
        print(f"  {'相关组（应命中）':<20}{'TOP1':>8}")
        for iid, s in r["related"]:
            print(f"    {iid:<18}{s:>8.3f}")
        print(f"  {'无关组（应不中）':<20}{'TOP1':>8}")
        for iid, s in r["unrelated"]:
            print(f"    {iid:<18}{s:>8.3f}")

    # ---------------------------------------------------------- 三组数字对比
    print(f"\n\n{C['bold']}── 三组关键数字 {'─' * 58}{C['reset']}")
    print(f"\n  {'dim':<8}{'相关组区间':<24}{'无关组最高':<14}{'空档宽度':<12}结论")
    print(f"  {'-' * 72}")
    for r in results:
        d = r["dim"]
        lo = min(s for _, s in r["related"])
        hi = max(s for _, s in r["related"])
        interval = f"[{lo:.3f}, {hi:.3f}]"
        maxu = r["max_unrel"]
        gap = r["gap"]
        if gap > 0.10:
            verdict = f"{C['green']}分得开（余量充足）{C['reset']}"
        elif gap > 0:
            verdict = f"{C['yellow']}勉强分开（余量小）{C['reset']}"
        else:
            verdict = f"{C['red']}重叠 —— 切不开{C['reset']}"
        print(f"  {d:<8}{interval:<24}{maxu:<14.3f}{gap:+.3f}".ljust(72)
              + f"  {verdict}")

    # ---------------------------------------------------------- 尺度变化
    #
    # ★ 这一段是用户第 4 点要求的：必须说明"区分度出现"里
    #   含有尺度变化的成分，不是纯碰撞消除。
    print(f"\n{C['bold']}── 尺度变化（不能只看空档，还要看分数整体往哪走）"
          f" {'─' * 22}{C['reset']}")
    print(f"\n  {'dim':<8}{'相关组均值':<14}{'无关组均值':<14}{'全体均值'}")
    print(f"  {'-' * 50}")
    for r in results:
        rmean = sum(s for _, s in r["related"]) / len(r["related"])
        umean = sum(s for _, s in r["unrelated"]) / len(r["unrelated"])
        amean = sum(r["all_scores"]) / len(r["all_scores"])
        print(f"  {r['dim']:<8}{rmean:<14.3f}{umean:<14.3f}{amean:.3f}")

    base = next(r for r in results if r["dim"] == 64)
    target = results[-1]
    b901 = dict(base["related"]).get("#901", 0)
    t901 = dict(target["related"]).get("#901", 0)
    print(f"""
  {C['yellow']}⚠ 维度提高会让**所有**余弦分数普遍下降 ——
    #901 从 {b901:.3f}（dim=64）掉到 {t901:.3f}（dim={target['dim']}）。
    原因：维度越高，单个桶分到的质量越稀，任意两个向量的点积都被压小。

    {C['bold']}所以"区分度出现"里含有**尺度变化**的成分，不是纯碰撞消除。{C['reset']}
    {C['dim']}证据：无关组掉得更狠（碰撞造成的虚高被消掉），
    相关组也掉（真实重合的分数同样被稀释）——
    两者的**相对**关系反转了，这才是"分得开"的真正来源。{C['reset']}
""")

    # ---------------------------------------------------------- 阈值是否落在空档里
    #
    # ★ 空档 > 0 只是"两组分得开"，**还不代表当前阈值可用** ——
    #   阈值必须**落在空档内部**才行。这两件事经常被混为一谈。
    from shared import MIN_RECALL_SIMILARITY
    print(f"\n{C['bold']}── 当前阈值（{MIN_RECALL_SIMILARITY}）落在空档里吗"
          f" {'─' * 32}{C['reset']}")
    print(f"\n  {'dim':<8}{'空档区间':<22}{'阈值':<8}{'到下限':<10}{'到上限':<10}判定")
    print(f"  {'-' * 70}")
    for r in results:
        lo_u, hi_r = r["max_unrel"], min(s for _, s in r["related"])
        t = MIN_RECALL_SIMILARITY
        slack_lo = t - lo_u        # 阈值比"无关组最高"高多少
        slack_hi = hi_r - t        # "相关组最低"比阈值高多少
        ok = slack_lo > 0 and slack_hi > 0
        mark = (f"{C['green']}可用{C['reset']}" if ok
                else f"{C['red']}不可用{C['reset']}")
        interval = f"[{lo_u:.3f}, {hi_r:.3f}]"
        print(f"  {r['dim']:<8}{interval:<22}{t:<8.3f}"
              f"{slack_lo:<+10.3f}{slack_hi:<+10.3f}{mark}")
    print(f"""
  {C['dim']}「到下限」= 阈值 - 无关组最高（>0 才排除误召回）
  「到上限」= 相关组最低 - 阈值（>0 才保证不漏召回）
  **两个都要 >0，阈值才算可用。** 只报"空档宽度"是不够的。{C['reset']}""")

    # ---------------------------------------------------------- 结论
    r256 = next(r for r in results if r["dim"] == 256)
    print(f"\n{C['bold']}── 结论（256 是主项目默认值，所以优先用它）"
          f" {'─' * 26}{C['reset']}\n")
    if r256["gap"] > 0:
        print(f"  {C['green']}✓ dim=256 分得开（空档 {r256['gap']:+.3f}）{C['reset']}")
        print(f"     → 已改用 256（对齐主项目默认值），shared.py 的 EMBED_DIM 已是 256。")
        print(f"     {C['yellow']}⚠ 但空档只有 {r256['gap']:.3f}（≈±0.05 余量）——"
              f"「刚好够」不是「充裕」。{C['reset']}")
        print(f"     {C['dim']}  这是用 256 换「与主项目一致」付出的代价，"
              f"已在 shared.py 注释里明说。{C['reset']}")
        print(f"     {C['yellow']}⚠ 改 sample_issues() 措辞后要重跑 compare.py ——"
              f"余量薄，覆盖率容易塌。{C['reset']}")
    else:
        print(f"  {C['red']}✗ dim=256 分不开（空档 {r256['gap']:+.3f}）{C['reset']}")
        print(f"     → 保留 1024，并在注释里写明与主项目默认值的差异及理由。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
