# 12 · 把每日任务挂到 GitHub Actions

> 让"每日分诊 + 对账"**自己跑起来**，而不是每次都手动执行。
>
> **这一步做完，项目就从"能跑"变成"在跑"。**

---

## 一、这一步到底在做什么

```
现在：  你手动跑 python scripts/daily_triage.py → 有结果
        （不跑 → 什么都没有）

挂上后：每天北京时间 10:17 自动跑 → 状态库一天天长大
        → 两周后有一份**真实的持续运行记录**
```

**Actions 不是"部署"** —— 它不托管你的服务、不需要服务器，只是**闹钟 + 一次性容器**：

```
10:17 触发 → 开容器 → 装依赖 → 跑脚本 → 存状态 → 容器销毁
（第二天重复，从 cache 恢复状态）
```

**代价是"状态要自己存"** —— 这就是 workflow 里用 Actions cache 的原因。

---

## 二、三步操作（10 分钟）

### 第 1 步：在 GitHub 建**私有**仓库

仓库名建议 `nightwatch`。**设为 Private**（不公开代码）。

**Actions 在私有仓库同样可用**：免费额度 2000 分钟/月，
本任务每天约 5~25 分钟 → 够用。

### 第 2 步：推代码

在 `D:/agent program` 下执行（把 `<你的用户名>` 换掉）：

```bash
git remote add origin git@github.com:<你的用户名>/nightwatch.git
git push -u origin main
```

**代码已经初始化好了** —— 本地仓库已建、已提交（分支 `main`）、`.gitignore` 已配好。

**已确认不会提交的东西**（`.gitignore` 41 条规则）：
```
.env              ← API key（**最重要的一条**）
data/*.db         ← 运行时状态库
data/_*           ← 临时文件
__pycache__/
```

**验证方法**（推送前可以再自查一次）：
```bash
git ls-files --cached | findstr /I ".env"    # 应该只看到 .env.example
```

### 第 3 步：配 3 个 Secret

仓库页面 → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**：

| Name | Value | 说明 |
|---|---|---|
| `NW_GITHUB_TOKEN` | 你的 GitHub PAT | **只需 `public_repo` 读权限**（别用全权限的） |
| `NW_LLM_API_KEY` | 模型服务 key | 和本地 `.env` 里的一样 |
| `NW_LLM_BASE_URL` | 模型服务地址 | 同上 |

**模型名不用配** —— 已经写在 workflow 里（`qwen3.8-flash`），不是密钥。

---

## 三、验证跑通了

**手动触发一次**：仓库 → **Actions** → 左侧 `daily-triage` → **Run workflow**。

看这几处：

| 检查点 | 期望 |
|---|---|
| 任务变绿 | 无报错 |
| 日志里有"已获取 N 条" | GitHub API 通了 |
| 日志里有预测结果 | LLM 通了 |
| 末尾的"状态"段有非零数字 | 状态库在累积 |
| Artifacts 里有 `triage-status-N` | 报告存下来了 |

**如果失败**，最常见的三个原因：

| 现象 | 原因 | 修法 |
|---|---|---|
| `401 Unauthorized` | token 没配或权限不对 | 检查 Secret 名字拼写 + token 权限 |
| `403 rate limit` | token 无效（在匿名访问） | 同上 |
| 模型调用失败 | `NW_LLM_BASE_URL` 格式不对 | 本地 `.env` 里的值直接复制 |

---

## 四、之后会发生什么

```
每天 10:17  抓最近 7 天新开的 issue → 跑预测 → 存状态库
            ↓
            顺带对账"3 天前预测的那些"（查维护者真实标签）
            ↓
            预测错的 → 记录误诊 → 牵连用到的记忆
            ↓
            误诊率高的记忆 → 进复核队列（等你裁决）
```

**两周后你会得到**：
- 一份 `predictions` 表（每天新增的预测 + 对账结果）
- 一份 `misuse_events` 表（哪些记忆在误导）
- 一个真实的命中率数字（**前瞻验证**，不是离线 gold 上的 F1）

**这些正是 `docs/09` 说的"跑两周的不是脚本，是状态库在积累"。**

---

## 五、几个必须知道的事

### ① Actions 的 cron 不准时

延迟几分钟到几小时都正常（高峰期更慢）。**这个任务不依赖精确时间**，可以接受。

### ② 状态库用 cache 保存，有 7 天过期

workflow 用的是 GitHub Actions cache（不污染仓库）。cache 7 天不用会过期 ——
但**每天都跑，所以不会触发过期**。

（如果停跑超过 7 天，状态会丢 —— 这是这个方案的已知代价。
换成"把 db commit 回仓库"能避免，但会让仓库历史里塞满二进制文件。）

### ③ 手工触发比等定时更实用

调试阶段用 `workflow_dispatch`（页面上的 "Run workflow" 按钮）——
不用等到第二天。

### ④ 想暂停？

仓库 → Actions → 选中 workflow → 右上角 **Disable workflow**。

---

## 六、不想推 GitHub 的替代方案

| 方案 | 代价 |
|---|---|
| **Windows 任务计划程序** | 本机要一直开机；没有免费容器；但零门槛 |
| **自己的服务器 + cron** | 要有服务器；好处是完全可控 |
| **GitHub Actions（推荐）** | 要 push（可私有）；**但这是业界真实做法**，面试时能说"挂在 CI 上每天跑" |

**推荐 Actions 的理由不只是"能用"** ——
而是"**把 Agent 挂在 CI 上定时跑**"这句话本身就是工程成熟度的信号。
