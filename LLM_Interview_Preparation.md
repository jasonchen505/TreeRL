# TreeRL 面试准备手册 - LLM 算法实习深挖指南

> 基于 ACL'25 论文 TreeRL: LLM Reinforcement Learning with On-Policy Tree Search
> 
> 目标：帮助 MS 在读生系统掌握此框架，并准备 LLM & Agent 方向的算法实习面试

---

## 目录

1. [项目概述与技术定位](#1-项目概述与技术定位)
2. [核心算法深度解析](#2-核心算法深度解析)
3. [代码架构与关键实现](#3-代码架构与关键实现)
4. [LLM & Agent 面试高频考点](#4-llm--agent-面试高频考点)
5. [细节深挖与追问准备](#5-细节深挖与追问准备)
6. [自我介绍与项目介绍模板](#6-自我介绍与项目介绍模板)
7. [附录：关键术语表](#7-附录关键术语表)

---

## 1. 项目概述与技术定位

### 1.1 一句话定位

TreeRL 是一个**将 On-Policy 树搜索与强化学习结合**的 LLM 推理增强框架，通过熵引导的树搜索（EPTree）生成多样化推理路径，并利用过程监督信号替代传统的结果监督，显著提升了数学和代码推理任务的性能。

### 1.2 解决的核心问题

| 传统 ChainRL 的痛点 | TreeRL 的解决方案 |
|---|---|
| 独立采样探索空间有限 | 树搜索在相同推理成本下生成 **2x 多样化响应** |
| 只有最终结果的稀疏奖励 | 树结构提供**细粒度过程监督**（step-level reward） |
| 需要单独训练 PRM（Process Reward Model） | **直接从树结构推导过程奖励**，避免 reward hacking |
| MCTS 效率低（逐步生成） | EPTree 仅需 **2 次迭代** 即可构建多样化树 |

### 1.3 技术背景图谱

```
                    LLM 推理增强
                         │
        ┌────────────────┼────────────────┐
        │                │                │
   Prompting          Training        Search
        │                │                │
   CoT/ToT          SFT/RLHF        MCTS/EPTree
   Self-Consistency   DPO/KTO         Beam Search
                         │
              ┌──────────┼──────────┐
              │          │          │
          Outcome    Process    Combined
          Supervision Supervision
```

### 1.4 实验结果速览

在 Qwen-2.5-14B 上的关键结果：
- **MATH500**: SFT 76.6 → ChainRL 81.6 → **TreeRL 81.7** (+5.1)
- **AIME2024**: SFT 48.0 → ChainRL 53.9 → **TreeRL 55.9** (+7.9)
- **平均提升**: TreeRL 比 ChainRL 高 **2.9 个百分点**

---

## 2. 核心算法深度解析

### 2.1 EPTree：熵引导的树搜索

#### 核心思想

**从高不确定性的中间步骤分叉**，而非随机或逐步分叉。

#### 算法参数

`(M, N, L, T)` 树搜索的四个关键参数：
- `M`: 并行树的数量（初始化时生成的独立链数）
- `N`: 每次迭代选择的分叉点数量
- `L`: 迭代扩展的轮数
- `T`: 每个分叉点的分支因子

#### 算法流程详解

```
Algorithm: EPTree (M, N, L, T)-tree

Input: Prompt x, Policy π_θ, 参数 M, N, L, T
Output: 树结构集合 {T_i}

Step 1: 初始化
    For i = 1 to M:
        Y^(i) ~ π_θ(· | x)          # 生成 M 个初始响应
        T_i = {Y^(i)}               # 构建初始树

Step 2: 迭代扩展（循环 L 轮）
    For l = 1 to L:
        For 每棵树 T_i:
            # 计算所有 token 的熵
            H(y_t) = -log π_θ(y_t | x, y_{<t})  ∀t ∈ T_i
            
            # 选择 Top-N 高熵 token 作为分叉点
            B_{i,l} = TopN_H { (t, H(y_t | x, y_{<t})) | t ∈ T_i }
            
            # 对每个分叉点生成 T 个新分支
            For 每个分叉点 (t, ·) ∈ B_{i,l}:
                Y_new^(i,l) ~ {π_θ(· | x, y_{<t})}^T
                T_i ← T_i ∪ Y_new^(i,l)

Step 3: 评估
    For 每个节点 s_n ∈ T_i:
        # 蒙特卡洛估计节点价值
        V(s_n) = (1/|L(s_n)|) * Σ_{l∈L(s_n)} 1(l is correct)
        
        # 计算过程奖励
        R(s_n) = |L(s_n)|^(-1/2) * [G_A(s_n) + L_A(s_n)]
        
        # 其中：
        # G_A(s_n) = V(s_n) - V(root)     # 全局优势
        # L_A(s_n) = V(s_n) - V(p(s_n))   # 局部优势

Step 4: RL 训练
    使用树中提取的序列进行策略梯度更新
```

#### 熵的计算与意义

```python
# 代码实现 (tree_node.py:192-199)
def get_max_entropy_tokens(self, top_n: int = 1) -> List[int]:
    entropies = []
    for i, log_prob in enumerate(self.log_prob_list):
        if not self.mask[i]:  # 只考虑未被 mask 的 token
            entropy = -log_prob  # 负对数概率作为熵
            entropies.append((entropy, i))
    
    sorted_indices = sorted(entropies, key=lambda x: x[0], reverse=True)
    return [idx for _, idx in sorted_indices[:top_n]]
```

**为什么用熵？**
- 高熵 = 模型对该位置的预测不确定性高
- 这些位置是推理的关键决策点
- 在这些点分叉能探索更多可能的推理路径

#### Mask 机制

```python
# tree_node.py:92-116
self.mask: List[bool] = [False] * len(self.token_str_list)

# 旁支的第一个 token 不再被选中（避免重复分叉）
if len(self.aggregate_token_ids) > 0 and len(self.token_str_list) > 0:
    self.mask[0] = True

# 超过最大长度的 token 被 mask
if total_length > max_length:
    for i in range(max(0, len(self.mask) - tokens_to_mask), len(self.mask)):
        self.mask[i] = True
    self.is_end = True

# 特殊 token（如 "conclusion", "answer"）后的 token 被 mask
for i, token_str in enumerate(self.token_str_list):
    if "conclusion" in token_str.lower() or "answer" in token_str.lower():
        for j in range(i + 1, len(self.mask)):
            self.mask[j] = True
        self.is_end = True
        break
```

### 2.2 过程监督信号

#### 全局优势 (Global Advantage)

```python
# 论文公式
G_A(s_n) = V(s_n) - V(root)

# 直觉：这个节点比平均水平好多少？
# V(root) 是所有生成响应的平均正确率
```

#### 局部优势 (Local Advantage)

```python
# 论文公式
L_A(s_n) = V(s_n) - V(p(s_n))

# 直觉：这个步骤比父节点好多少？
# 如果 L_A > 0，说明这一步是正确的推理方向
```

#### 最终奖励

```python
# 论文公式
R(s_n) = |L(s_n)|^(-1/2) * [G_A(s_n) + L_A(s_n)]

# 1/√|L(s_n)| 是重加权因子
# 防止共享节点（非叶子节点出现在多条路径中）被过度训练
```

#### 与 GAE 的关系

```python
# TreeRL 的过程监督可以看作 GAE 的特例
R_GAE(s_n) = Σ_{j∈P(s_n)} λ_j * [V(s_n) - V(s_j)]

# 当只考虑 root 和 parent 时：
# λ_j = 1  if j ∈ {root, parent}
# λ_j = 0  otherwise
# 这就退化为 TreeRL 的公式
```

### 2.3 优势函数对比

| 方法 | 优势函数定义 | 特点 |
|---|---|---|
| GRPO | A(y_i) = (r(y_i) - mean) / std | 组内标准化 |
| RLOO | A(y_i) = r(y_i) - (1/(K-1))Σ_{j≠i} r(y_j) | 留一法 baseline |
| **TreeRL** | A(s_n) = G_A(s_n) + L_A(s_n) | 全局+局部优势 |

---

## 3. 代码架构与关键实现

### 3.1 整体架构

```
TreeRL/
├── openrlhf/
│   ├── models/
│   │   ├── actor.py              # Actor 模型（策略模型）
│   │   ├── loss.py               # 损失函数（PolicyLoss, ValueLoss）
│   │   └── utils.py              # 工具函数
│   ├── trainer/
│   │   ├── ppo_trainer.py        # PPO 训练器
│   │   ├── reinforce_trainer.py  # REINFORCE 训练器
│   │   └── ppo_utils/
│   │       ├── tree_node.py      # 树节点定义
│   │       ├── entropy_chain_local_manager.py  # EPTree 管理器
│   │       ├── entropy_guided_tree_search.py   # 熵引导树搜索
│   │       ├── parallel_mcts.py  # MCTS 实现（对比基线）
│   │       ├── experience_maker.py  # 经验生成器
│   │       └── experience_maker_ppo.py  # PPO 经验生成
│   └── utils/                    # 工具模块
├── train_ppo.py                  # PPO 训练脚本
├── train_reinforce.py            # REINFORCE 训练脚本
├── train_reinforce_ray.py        # Ray 分布式训练脚本
└── scripts/
    └── treerl-qw14b.sh           # Qwen-14B 训练配置
```

### 3.2 关键类解析

#### TreeNode 类

```python
class TreeNode:
    def __init__(self, ...):
        # --- 分组信息 ---
        self.tree_idx: int          # 树的索引
        self.node_idx: int          # 节点的索引
        
        # --- 文本信息 ---
        self.token_id_list: List[int]      # token IDs
        self.token_str_list: List[str]     # token 字符串
        self.log_prob_list: List[float]    # log 概率
        
        # --- 父子关系 ---
        self.parent_node = None            # 父节点
        self.child_nodes: List[TreeNode]   # 子节点列表
        self.child_split_indices: List[int]  # 分叉位置
        
        # --- 聚合信息（避免重复遍历）---
        self.aggregate_str: str = ""       # 从根到当前的完整字符串
        self.total_str: str = ""           # 当前节点的完整字符串
        self.aggregate_token_ids: List[int] = []
        
        # --- Mask 信息 ---
        self.mask: List[bool]  # 标记哪些位置不能再分叉
        
        # --- 分数 ---
        self.binary_score: Optional[float] = None  # 二元正确性
        self.score: Optional[float] = None          # 最终分数
```

#### EntropyGuidedChainLocalManager 类

```python
class EntropyGuidedChainLocalManager:
    def entropy_guided_chain(self, problem_str, answer_str, args):
        """核心：熵引导的树搜索"""
        
        # Step 1: 初始化 M 棵树
        initial_results = query_local_vllm_ids_with_logprobs(
            initial_prompt_ids, llm=self.llm, ...
        )
        for idx, (...) in enumerate(zip(*initial_results)):
            root_node = TreeNode(...)
            self.tree_lists.append([root_node])
        
        # Step 2: 迭代扩展
        for iteration in range(L):
            expansion_tasks = []
            
            # 收集所有高熵 token
            for tree_idx, tree_list in enumerate(self.tree_lists):
                for node_idx, node in enumerate(tree_list):
                    if not all(node.mask):
                        entropy_tokens = node.get_max_entropy_tokens(top_n=N)
                        for token_idx in entropy_tokens:
                            entropy_value = -node.log_prob_list[token_idx]
                            tree_entropy_tokens.append(
                                (entropy_value, tree_idx, node_idx, node, token_idx)
                            )
            
            # 排序选择 Top-N
            tree_entropy_tokens.sort(reverse=True)
            expansion_tasks.extend(tree_entropy_tokens[:N])
            
            # 批量推理生成新分支
            inference_results = query_local_vllm_ids_with_logprobs(
                m_tree_top_n_prompt_ids, llm=self.llm, ...
            )
            
            # 更新树结构
            for i, (...) in enumerate(zip(*inference_results)):
                new_node = TreeNode(...)
                parent_node.add_child(new_node, split_idx)
                self.tree_lists[tree_idx].append(new_node)
        
        # Step 3: 评估和构建训练数据
        paths['pass_k_result'] = self.evaluate_trees(problem_str, answer_str, args)
        root, selected_terminals = build_into_tree_format(self.tree_lists, ...)
        paths = gather_paths(root=root, selected_terminals=selected_terminals, ...)
        
        return paths
```

### 3.3 训练流程

```python
# train_reinforce_ray.py 的主要流程

# 1. 初始化分布式环境
strategy = get_strategy(args)

# 2. 加载模型
actor = Actor(args.pretrain, ...)           # 策略模型
initial_model = Actor(args.pretrain, ...)    # 参考模型
reward_model = get_llm_for_sequence_regression(...)  # 奖励模型（可选）

# 3. 创建 Experience Maker
experience_maker = RemoteExperienceMakerPPO(
    actor, critic, reward_model, remote_rm_url, initial_model, tokenizer, ...
)

# 4. 创建 Trainer
trainer = REINFORCETrainer(
    strategy, actor, reward_model, initial_model, actor_optim, ...
)

# 5. 训练循环
for episode in range(num_episodes):
    for rand_prompts in prompts_dataloader:
        # 5.1 生成经验（包含树搜索）
        experience = experience_maker.make_experience(rand_prompts, ...)
        
        # 5.2 存入 Replay Buffer
        replay_buffer.append(experience)
        
        # 5.3 策略更新
        if global_step % update_timesteps == 0:
            status = reinforce_train(global_step)
            replay_buffer.clear()
```

### 3.4 分布式架构（Ray）

```
┌─────────────────────────────────────────────────────────────┐
│                      Ray Cluster                            │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Actor Node  │  │  Actor Node  │  │  Actor Node  │        │
│  │  (GPU 0-7)  │  │  (GPU 0-7)  │  │  (GPU 0-7)  │        │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│         │                │                │                 │
│         └────────────────┼────────────────┘                 │
│                          │                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              vLLM Engines (16个)                      │   │
│  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐   │   │
│  │  │ TP1 │ │ TP1 │ │ TP1 │ │ TP1 │ │ TP1 │ │ TP1 │   │   │
│  │  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘   │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Remote Services                         │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │   │
│  │  │ Reward Model │  │  Extractor  │  │   Checker   │  │   │
│  │  │  (Qwen-72B) │  │  (答案提取)  │  │  (答案验证)  │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. LLM & Agent 面试高频考点

### 4.1 基础概念

#### Q1: 什么是 RLHF？为什么需要 RL？

**标准回答**：
RLHF（Reinforcement Learning from Human Feedback）是通过人类反馈来训练语言模型的方法。SFT 只能模仿人类标注的数据，但 RLHF 可以：
1. **探索更优解**：模型可以生成比训练数据更好的回答
2. **对齐价值观**：通过奖励模型引导模型生成更符合人类偏好的内容
3. **避免 Distribution Shift**：On-policy 训练确保模型从自己的分布中学习

**加分点**：
- 提到 SFT 的局限性：只能学到训练数据的分布，无法超越
- 提到 Reward Hacking：单独训练的奖励模型容易被 hack
- 提到 On-policy vs Off-policy 的区别

#### Q2: 解释 PPO 和 REINFORCE 的区别

| 特性 | PPO | REINFORCE |
|---|---|---|
| 是否需要 Critic | 需要（Value Network） | 不需要 |
| 优势估计 | GAE (Generalized Advantage Estimation) | 简单 baseline |
| 方差 | 较低 | 较高 |
| 计算成本 | 较高 | 较低 |
| 稳定性 | 更稳定 | 较不稳定 |

**TreeRL 的选择**：TreeRL 使用 REINFORCE 风格的训练，因为：
1. 树搜索本身提供了低方差的优势估计
2. 避免了训练额外的 Value Network
3. 实现更简单，更容易扩展

#### Q3: 什么是 On-policy 和 Off-policy？

**On-policy**：
- 策略模型生成的数据用于训练同一个策略模型
- 数据分布与当前策略一致
- 例如：PPO, REINFORCE

**Off-policy**：
- 使用其他策略生成的数据训练当前策略
- 数据分布可能与当前策略不一致
- 例如：DPO (使用 SFT 模型生成的 chosen/rejected 对)

**TreeRL 的优势**：TreeRL 是 On-policy 的，因为树搜索是用当前策略模型生成的。

### 4.2 进阶概念

#### Q4: 什么是 Process Supervision？为什么比 Outcome Supervision 好？

**Outcome Supervision**：
```
Prompt → Response → Reward (只看最终结果)
```

**Process Supervision**：
```
Prompt → Step1 → Step2 → ... → StepN → Reward
              ↑        ↑              ↑
           每一步都有监督信号
```

**TreeRL 的过程监督**：
```python
# 从树结构推导过程奖励
R(s_n) = G_A(s_n) + L_A(s_n)

# G_A: 全局优势 - 这个节点比平均水平好多少？
# L_A: 局部优势 - 这个步骤比父节点好多少？
```

**优势**：
1. **细粒度反馈**：每一步都有明确的改进方向
2. **避免 Reward Hacking**：不需要单独训练 PRM
3. **更高效的探索**：知道哪些步骤是关键的

#### Q5: 什么是 Advantage Function？为什么需要 Baseline？

**优势函数**：
```
A(s, a) = Q(s, a) - V(s)
```
表示在状态 s 下采取动作 a 比平均水平好多少。

**为什么需要 Baseline**：
1. **降低方差**：减去 baseline 后，优势的方差更小
2. **不影响梯度**：baseline 不改变梯度的期望

**常见 Baseline**：
- GRPO: `mean(reward)`
- RLOO: `(1/(K-1)) * Σ_{j≠i} reward_j`
- TreeRL: `V(root)` (全局) 和 `V(parent)` (局部)

#### Q6: 什么是 KL 散度约束？为什么需要？

**KL 散度**：
```
KL(π_θ || π_ref) = Σ π_θ(a|s) * log(π_θ(a|s) / π_ref(a|s))
```

**作用**：
1. **防止策略偏移**：确保新策略不会偏离参考策略太远
2. **保持语言质量**：避免生成不自然的文本
3. **稳定训练**：防止策略崩溃

**TreeRL 的做法**：
```python
# 在经验中计算 KL
base_action_log_probs = initial_model(sequences, num_actions, attention_mask)
kl = action_log_probs - base_action_log_probs

# 计算 reward 时加入 KL 惩罚
reward, kl = compute_reward(
    r, self.kl_ctl.value, action_log_probs, base_action_log_probs, action_mask=action_mask
)
```

### 4.3 树搜索与推理

#### Q7: MCTS 和 TreeRL 的 EPTree 有什么区别？

| 特性 | MCTS | EPTree |
|---|---|---|
| 分叉策略 | UCT (探索-利用权衡) | 熵引导 |
| 生成方式 | 逐步生成 | 批量生成 |
| 迭代次数 | 多次 | 仅 2 次 |
| 并行性 | 有限 | 高度并行 |
| 探索效率 | 一般 | 更高 |

**EPTree 的优势**：
1. **更高效**：在相同推理成本下生成 2x 多样化响应
2. **更易并行**：批量生成适合 vLLM 等推理引擎
3. **更聚焦**：在高不确定性点分叉，探索更有效

#### Q8: 什么是 PassRate？TreeRL 如何提高 PassRate？

**PassRate 定义**：
```
PassRate = (1/|D|) * Σ_{d∈D} max_{l∈L_d} c(l)
```
其中 c(l) ∈ {0, 1} 表示响应 l 是否正确。

**直觉**：在给定的推理预算内，至少生成一个正确答案的概率。

**TreeRL 如何提高 PassRate**：
1. **更多多样化响应**：相同 token 预算下生成 2x 响应
2. **更智能的探索**：在高熵点分叉，探索更有潜力的路径
3. **过程监督**：知道哪些步骤是正确的，鼓励这些步骤

#### Q9: 什么是 Entropy？为什么用它来指导树搜索？

**信息熵**：
```
H(X) = -Σ p(x) * log p(x)
```

**在 LLM 中**：
```
H(y_t) = -log π_θ(y_t | x, y_{<t})
```

**高熵意味着**：
1. 模型对该位置的预测不确定性高
2. 这是推理的关键决策点
3. 在这些点分叉能探索更多可能的推理路径

**实验验证**（论文 Figure 10）：
高频分叉 token 包括：
- 数学运算符：`\(`, `=`
- 逻辑连接词：`since`, `but`
- 反思词：`wait`（o1-like 模型常用）

#### Q10: 如何评估树搜索的效果？

**PassRate**：
```
PassRate = (1/|D|) * Σ_{d∈D} max_{l∈L_d} c(l)
```
在给定推理预算内，至少生成一个正确答案的概率。

**Pass@K**：
```
Pass@K = 1 - C(n-c, k) / C(n, k)
```
在 K 次采样中至少有一次正确的概率。

**TreeRL 的评估方法**：
```python
# 在 experience_maker.py 中
pass_rate = ((judge_rwd > 0).sum(1) > 0).float()
```

---

## 5. 细节深挖与追问准备

### 5.1 训练细节

#### Q1: 为什么 TreeRL 使用 KL=0？

**代码配置** (`scripts/treerl-qw14b.sh`):
```bash
KL=0.0
```

**原因**：
1. 树搜索已经提供了足够的探索
2. 避免 KL 惩罚过于严格，限制探索
3. 在 On-policy 设置下，KL 约束可能不是必需的

**追问**：如果 KL > 0 会怎样？
- 可能会限制树搜索的探索空间
- 但可能提高训练稳定性
- 需要实验验证最佳值

#### Q2: 为什么使用 RLOO 而不是 GRPO？

**代码** (`experience_maker.py`):
```python
if self.strategy.args.normalize_reward_from_multi_traces_with_rloo:
    r = normalize_reward_from_multi_traces_rloo(r, ...)
```

**原因**：
1. RLOO 的 baseline 是无偏估计
2. 在树搜索中，不同路径的长度不同，RLOO 更适合
3. 实验发现 RLOO 效果更好

**追问**：RLOO 和 GRPO 的区别？
- GRPO: `A(y_i) = (r(y_i) - mean) / std`
- RLOO: `A(y_i) = r(y_i) - (1/(K-1)) * Σ_{j≠i} r(y_j)`
- RLOO 的 baseline 是留一法估计，更无偏

#### Q3: 为什么使用 1/√|L(s_n)| 作为重加权因子？

**论文公式**：
```
R(s_n) = |L(s_n)|^(-1/2) * [G_A(s_n) + L_A(s_n)]
```

**原因**：
1. **防止过拟合**：共享节点（非叶子节点）出现在多条路径中
2. **平衡贡献**：避免高频节点主导训练
3. **实验验证**：√ 比线性或平方效果更好

**代码实现** (`tree_node.py:558`):
```python
if reweighted_value_style == "sqrt":
    node.accumulated_value = node.accumulated_value / math.sqrt(node.selected_terminal_in_subtree)
```

#### Q4: 如何处理重复模式？

**代码** (`parallel_mcts.py:184-199`):
```python
def find_repeated_patterns(s, pattern_length=50, threshold=20):
    ngrams = [s[i:i + pattern_length] for i in range(len(s) - pattern_length + 1)]
    ngram_counts = Counter(ngrams)
    repeated_patterns = {gram: count for gram, count in ngram_counts.items() if count > threshold}
    return repeated_patterns
```

**作用**：
1. 检测模型是否陷入重复循环
2. 如果检测到重复，将该节点标记为重复
3. 重复节点的奖励设为 0

### 5.2 工程细节

#### Q1: 为什么使用 vLLM 进行推理？

**优势**：
1. **高吞吐量**：支持批量推理
2. **PagedAttention**：高效管理 KV Cache
3. **连续批处理**：动态调整 batch size
4. **Tensor Parallelism**：支持多 GPU 推理

**代码** (`entropy_guided_tree_search.py:97-108`):
```python
llm = LLM(
    model=tokenizer_path,
    tensor_parallel_size=1,
    trust_remote_code=True,
    gpu_memory_utilization=0.85,
    seed=3407
)
```

#### Q2: 如何处理不同长度的树路径？

**方法**：Left Padding + Action Mask

```python
# experience_maker.py:388-396
def zero_pad_batch(sequences, side="right", pad_token_id=0):
    max_len = max(seq.size(1) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(1)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding, value=pad_token_id))
    return torch.cat(padded_sequences, dim=0)
```

#### Q3: 如何处理不同模型的 Tokenization？

**代码** (`experience_maker.py:672-679`):
```python
def tokenize_fn(self, texts, max_length, device, system_prompt=None):
    if "glm" in self.current_model.lower():
        return tokenize_fn_chatglm(self.tokenizer, texts, max_length, device)
    elif "qwen" or "llama" in self.current_model.lower():
        return tokenize_fn_llama(self.tokenizer, texts, max_length, device, system_prompt=system_prompt)
    else:
        raise ValueError(f"Unsupported model: {self.current_model}")
```

**ChatGLM 特殊处理**：
```python
def _tokenize_fn_chatglm(tokenizer, prompt, history, max_length):
    gmask = tokenizer.encode("[gMASK]")[0]
    sop = tokenizer.encode("<sop>")[0]
    input_ids = [gmask, sop]
    # ... 添加对话历史
    sample_input_ids = tokenizer.encode("<|user|>\n" + prompt, add_special_tokens=False)
    sample_input_ids = sample_input_ids[-max_length:] + tokenizer.encode("<|assistant|>\n", add_special_tokens=False)
    return sample_input_ids
```

### 5.3 扩展性问题

#### Q1: 如何将 TreeRL 扩展到其他任务（如代码生成）？

**挑战**：
1. 代码任务的正确性更难验证
2. 需要更复杂的奖励函数
3. 可能需要多轮交互

**可能的解决方案**：
1. 使用单元测试作为奖励信号
2. 结合静态分析工具
3. 引入代码执行反馈

#### Q2: TreeRL 的计算成本如何？

**训练成本**：
- 推理：16 个 vLLM engines，每个 GPU 一个
- 训练：2 个节点，每个节点 8 个 GPU
- 总 GPU 数：16 + 16 = 32 个 GPU

**内存优化**：
```bash
--zero_stage 3           # ZeRO Stage 3 分片
--adam_offload           # Adam 优化器 offload 到 CPU
--gradient_checkpointing # 梯度检查点
--bf16                   # 混合精度训练
```

#### Q3: 如何进一步优化 TreeRL？

**可能的优化方向**：
1. **更智能的分叉策略**：使用 Value Network 预测哪些点值得分叉
2. **动态调整 M, N, L, T**：根据问题难度自适应调整
3. **更高效的过程监督**：使用更细粒度的 step reward
4. **结合其他 RL 算法**：如 GRPO, RLOO 等

---

## 6. 自我介绍与项目介绍模板

### 6.1 一分钟自我介绍

> 您好，我是 [姓名]，[学校] [专业] 的硕士生，主要研究方向是 [具体方向]。我对 LLM 的推理增强和强化学习训练非常感兴趣，最近深入研究了 TreeRL 这篇 ACL'25 的论文，并复现了其中的核心算法。在这个过程中，我对 RLHF、PPO、过程监督等技术有了深入的理解，也积累了分布式训练和推理优化的实践经验。

### 6.2 三分钟项目介绍

> 我来介绍一下我最近研究的 TreeRL 框架。
>
> **背景与问题**：
> 传统的 LLM 强化学习（如 ChainRL）使用独立的多链采样，存在三个主要问题：
> 1. 探索空间有限，在相同推理成本下生成的多样化响应较少
> 2. 只有最终结果的稀疏奖励，缺乏细粒度的反馈
> 3. 如果使用过程监督，需要单独训练 PRM，容易出现 reward hacking
>
> **核心创新**：
> TreeRL 提出了 EPTree（熵引导的树搜索）来解决这些问题：
> 1. **分叉策略**：在模型预测不确定性最高的位置（高熵 token）分叉，而非随机分叉
> 2. **过程监督**：直接从树结构推导过程奖励，包括全局优势（比平均水平好多少）和局部优势（比父节点好多少）
> 3. **效率提升**：仅需 2 次迭代即可构建多样化的树，在相同推理成本下生成 2x 响应
>
> **实验结果**：
> 在 Qwen-2.5-14B 上，TreeRL 在多个数学推理基准测试上显著优于 ChainRL：
> - AIME2024: 48.0 → 55.9 (+7.9)
> - 平均提升: +2.9 个百分点
>
> **我的理解**：
> 通过深入研究这个项目，我理解了：
> 1. RLHF 的完整流程：SFT → Reward Model → PPO/REINFORCE
> 2. 过程监督 vs 结果监督的优劣
> 3. 树搜索在推理增强中的应用
> 4. 分布式训练和推理优化的实践经验

### 6.3 面试官可能的追问与回答

#### 追问1: 为什么选择在高熵位置分叉？

> 高熵意味着模型对该位置的预测不确定性高。这些位置通常是推理的关键决策点，比如：
> - 选择使用哪个数学公式
> - 决定推理的方向
> - 判断前一步的结论是否正确
>
> 在这些点分叉，能更有效地探索不同的推理路径，而不是在确定性高的位置（如函数名、变量名）浪费计算资源。

#### 追问2: TreeRL 的过程监督和单独训练 PRM 有什么区别？

> 主要区别在于：
> 1. **避免分布偏移**：PRM 是在离线数据上训练的，而 TreeRL 的过程监督是在线生成的，更符合当前策略的分布
> 2. **避免 Reward Hacking**：单独训练的 PRM 容易被 hack，而 TreeRL 直接从正确性推导奖励
> 3. **更简单**：不需要额外训练和维护 PRM
>
> 论文的消融实验也验证了这一点：直接使用 TreeRL 的过程监督比使用 PRM 效果更好。

#### 追问3: 如果让你改进 TreeRL，你会怎么做？

> 我有几个想法：
> 1. **更智能的分叉策略**：训练一个小型的 Value Network 来预测哪些位置值得分叉，而不是仅依赖熵
> 2. **动态调整参数**：根据问题难度自适应调整 M, N, L, T，简单问题用更少的计算资源
> 3. **多模态扩展**：将树搜索扩展到多模态推理，如视觉问答
> 4. **结合其他 RL 算法**：尝试 GRPO 或 RLOO 等更先进的优势函数

#### 追问4: 你在研究过程中遇到了什么困难？

> 有三个主要困难：
> 1. **代码理解**：OpenRLHF 的代码量很大，理解树搜索的完整流程需要时间
> 2. **分布式调试**：Ray + vLLM + DeepSpeed 的分布式环境调试比较复杂
> 3. **超参数调优**：M, N, L, T 的选择对性能影响很大，需要仔细调优
>
> 通过阅读论文、调试代码和分析实验结果，我逐步解决了这些问题。

---

## 7. 附录：关键术语表

| 术语 | 定义 | 相关论文/代码 |
|---|---|---|
| **RLHF** | Reinforcement Learning from Human Feedback | InstructGPT |
| **PPO** | Proximal Policy Optimization | Schulman et al., 2017 |
| **REINFORCE** | 一种策略梯度算法 | Williams, 1992 |
| **GRPO** | Group Relative Policy Optimization | DeepSeekMath |
| **RLOO** | REINFORCE Leave-One-Out | Ahmadian et al., 2024 |
| **DPO** | Direct Preference Optimization | Rafailov et al., 2023 |
| **KTO** | Kahneman-Tversky Optimization | Ethayarajh et al., 2024 |
| **MCTS** | Monte Carlo Tree Search | Browne et al., 2012 |
| **EPTree** | Entropy-guided Parallel Tree (TreeRL) | TreeRL |
| **PRM** | Process Reward Model | Lightman et al., 2023 |
| **ORM** | Outcome Reward Model | - |
| **Advantage** | 优势函数，表示动作比平均水平好多少 | - |
| **Baseline** | 用于降低方差的参考值 | - |
| **KL Divergence** | KL 散度，衡量两个分布的距离 | - |
| **On-policy** | 使用当前策略生成的数据训练 | - |
| **Off-policy** | 使用其他策略生成的数据训练 | - |
| **PassRate** | 在给定预算内至少生成一个正确答案的概率 | TreeRL |
| **Pass@K** | K 次采样中至少有一次正确的概率 | - |
| **vLLM** | 高效的 LLM 推理引擎 | - |
| **DeepSpeed** | 微软的分布式训练框架 | - |
| **Ray** | 分布式计算框架 | - |
| **ZeRO** | Zero Redundancy Optimizer | - |
| **Gradient Checkpointing** | 梯度检查点，减少内存使用 | - |
| **LoRA** | Low-Rank Adaptation，参数高效微调 | - |

---

## 参考资源

1. **论文**：TreeRL: LLM Reinforcement Learning with On-Policy Tree Search (ACL'25)
2. **代码**：https://github.com/THUDM/TreeRL
3. **基础论文**：
   - Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
   - Ouyang et al., "Training language models to follow instructions with human feedback" (2022)
   - Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models" (2024)
   - Ahmadian et al., "Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs" (2024)

---

*Last Updated: 2026-07-03*
*Author: TreeRL Study Group*
