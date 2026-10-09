"""
Nightwatch · 空记忆存储（Null Object 模式）

================================================================================
用途：组件组合实验里"关闭记忆"这一组的实现
================================================================================
P1-3 要比较"开记忆"和"关记忆"的效果差异。要构造"关记忆"这个条件，
有两种做法：

    做法 A：传 None
    做法 B：传一个"什么也不做"的 store（Null Object）

**做法 A 是错的**，因为调用方写的是 `store.recall(...)` ——
传 None 会直接 `AttributeError`。要支持它，就得在**每个调用点**
加 `if store is not None:` 判断。

而"每个调用点都要记得判断"的东西，一定会在某处被忘记
（这个教训在 `RemoteMemoryStore` 的降级设计里也出现过）。

**做法 B（Null Object）让调用方代码完全不用改。**
这就是"多态"的价值：用一个行为符合接口的空实现，让上层无感。

================================================================================
★ 为什么它还要统计调用次数
================================================================================
如果 NullMemoryStore 什么都不做，你怎么知道"记忆确实关掉了"？

    如果你不小心在代码某处仍然用了真实的 store，
    那一组的"关闭记忆"就是假的 —— 而实验会继续跑，
    只是结论悄悄错了。

所以它记录被调用的次数。实验报告里会打印这些计数：
**如果"关闭记忆"组的 recall 调用次数是 0，说明配置根本没生效**，
实验结论不可信。

**这属于"实验有效性检查"** —— 和 `docs/03` 里长程压测那个
"验证实验真的测到了东西"是同一个思路。
"""

from __future__ import annotations

from typing import Any, Optional

from ..models import MemoryRecord, MemoryTier, PendingConflict


class NullMemoryStore:
    """不存不取的记忆存储 —— 接口与 MemoryStore 一致。

    所有读取返回空、所有写入返回 False，但**记录调用次数**以便验证。
    """

    def __init__(self, name: str = "null"):
        self.name = name
        self.calls: dict[str, int] = {}
        # ★ 必须提供 encoder —— 这个属性最初漏了，导致"关记忆"组全部
        #   AttributeError。调用方（TriageAgent / ToolRegistry）会用它。
        #
        #   这不是"多此一举"：空实现要**完整**替代真实实现，
        #   缺任何一个属性都会让调用方崩 —— 而崩溃发生在"关记忆"这一组，
        #   会被误读成"关掉记忆导致 Agent 不可用"，
        #   而不是"我们的空实现不完整"。
        #
        #   教训：**空实现的正确性标准和真实实现一样高。**
        from .store import HashEmbedding
        self.encoder = HashEmbedding(dim=256)
        # 语义层相关的接口（P1-2 新增）—— 同样要补齐
        self.db_path = ":null:"

    def _count(self, method: str) -> None:
        self.calls[method] = self.calls.get(method, 0) + 1

    # ---------------------------------------------------------------- 写入

    def insert(self, mem: MemoryRecord) -> bool:
        self._count("insert")
        return False        # 没写入 —— 这是刻意的

    def update(self, mem: MemoryRecord) -> None:
        self._count("update")

    def delete(self, memory_id: str) -> None:
        self._count("delete")

    # ---------------------------------------------------------------- 读取

    def get(self, memory_id: str) -> Optional[MemoryRecord]:
        self._count("get")
        return None

    def get_by_idem(self, idem_key: str) -> Optional[MemoryRecord]:
        self._count("get_by_idem")
        return None

    def count(self, tier: MemoryTier | None = None) -> int:
        self._count("count")
        return 0

    def recall(self, query: str, author: str | None = None,
               tier: MemoryTier | None = MemoryTier.EPISODIC,
               top_k: int = 5, candidate_pool: int = 50,
               include_archived: bool = False) -> list[tuple[MemoryRecord, float]]:
        """返回空列表 —— 接口形状必须和真实 store 一致。

        ★ 注意返回类型是 `list[tuple[MemoryRecord, float]]`（记忆 + 分数），
        不是 `list[MemoryRecord]`。

        这个细节在 P1-2 时踩过坑：远程客户端的返回类型和本地不一致，
        导致"单机能跑、切分布式就崩"。所以**空实现也必须严格对齐接口** ——
        它平时不执行循环体，一旦有人改了返回值形状，
        这个类会是最不容易暴露问题的地方。
        """
        self._count("recall")
        return []

    def find_similar(self, mem: MemoryRecord,
                     threshold: float) -> Optional[MemoryRecord]:
        self._count("find_similar")
        return None         # "没找到相似" —— 保守取值

    def touch_access(self, memory_id: str, used: bool) -> None:
        self._count("touch_access")

    def all_memories(self, tier: MemoryTier | None = None) -> list[MemoryRecord]:
        self._count("all_memories")
        return []

    # ---------------------------------------------------------------- 其他

    def add_conflict(self, c: PendingConflict) -> None:
        self._count("add_conflict")

    def pending_conflicts(self) -> list[PendingConflict]:
        self._count("pending_conflicts")
        return []

    def save_trace(self, trace) -> None:
        self._count("save_trace")

    def list_traces(self, outcome: str | None = None) -> list[dict]:
        self._count("list_traces")
        return []

    def close(self) -> None:
        pass

    @staticmethod
    def _embed_text(mem: MemoryRecord) -> str:
        """拼接用于向量化的文本。

        ★ 为什么空实现也要有这个**私有**方法？

        因为它虽然叫 `_embed_text`，但实际上被**外部调用**了
        （比如 lifecycle 里构造 embedding 时用它）。

        **"下划线开头"在 Python 里只是约定，不是访问控制。**
        一旦有外部调用点，它就成了事实上的公开接口 ——
        空实现要完整替代真实实现，就必须提供它。

        这个坑是实测踩到的：NullStore 少了它，导致"关记忆"组
        全部 AttributeError，而那个错误很容易被误读成
        "关掉记忆导致 Agent 不可用"（实际是"我们的空实现不完整"）。
        """
        return f"{mem.symptom} {mem.root_cause} {mem.resolution}"

    # ------------------------------------------------- 语义层（P1-2 新增接口）
    # ★ 这些方法是 P1-2 加到 MemoryStore 上的，NullStore 当时并不存在。
    #
    #   写 NullStore 时（P1-3）才发现漏了这一整组 ——
    #   这就是**接口漂移**：多个实现之间没有强制的契约检查，
    #   加方法时很容易只改一个。
    #
    #   所以 tests/test_smoke.py 里有一条测试专门比对
    #   NullMemoryStore 和 MemoryStore 的公开方法是否一致 ——
    #   **让"忘记同步"变成测试失败，而不是运行时的 AttributeError。**

    def add_promotion(self, p) -> bool:
        self._count("add_promotion")
        return False

    def pending_promotions(self) -> list:
        self._count("pending_promotions")
        return []

    def all_promotions(self, status: str | None = None) -> list:
        self._count("all_promotions")
        return []

    def get_promotion(self, promotion_id: str):
        self._count("get_promotion")
        return None

    def review_promotion(self, promotion_id: str, approved: bool,
                         reviewer: str, note: str = "",
                         promoted_memory_id: str = "") -> bool:
        self._count("review_promotion")
        return False

    # ------------------------------------------------- 向量索引（本次新增的接口）
    # ★ 这就是 `TestStoreInterfaceConsistency` 抓到的 ——
    #   给 MemoryStore 加索引方法时，忘了同步空实现。
    #
    #   **测试在第一时间拦住了它**，而不是等到某次实验跑出
    #   "关记忆组 AttributeError" 才去排查。
    #   这正是那条测试存在的意义：让"忘记同步"变成测试失败。

    def build_index(self, nlist: int = 16, tier=None):
        self._count("build_index")
        return None       # 空实现没有向量可索引

    def attach_index(self, index, tier=None) -> None:
        self._count("attach_index")

    def index_stats(self):
        self._count("index_stats")
        return None

    # ---------------------------------------------------------------- 报告

    def total_calls(self) -> int:
        return sum(self.calls.values())

    def __repr__(self) -> str:
        return f"<NullMemoryStore calls={self.total_calls()}>"
