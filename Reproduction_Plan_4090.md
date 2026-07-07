# TreeRL 8卡4090复现完整计划

> 基于原始项目分析，针对8卡RTX 4090 (24GB) 环境的适配方案

---

## 目录

1. [硬件资源评估](#1-硬件资源评估)
2. [原始项目配置分析](#2-原始项目配置分析)
3. [适配方案设计](#3-适配方案设计)
4. [分阶段复现计划](#4-分阶段复现计划)
5. [详细执行步骤](#5-详细执行步骤)
6. [常见问题与解决方案](#6-常见问题与解决方案)
7. [时间规划与里程碑](#7-时间规划与里程碑)

---

## 1. 硬件资源评估

### 1.1 8卡RTX 4090 规格

| 参数 | RTX 4090 | A100 80GB (原始) | 差异 |
|---|---|---|---|
| **显存** | 24 GB | 80 GB | 4090 仅 30% |
| **FP32** | 82.6 TFLOPS | 19.5 TFLOPS | 4090 更高 |
| **BF16** | 165.2 TFLOPS | 312 TFLOPS | A100 更高 |
| **互联** | PCIe 4.0 | NVLink 3.0 | 4090 带宽低 |
| **多卡并行** | 需要 PCIe 交换 | 原生 NVLink | 4090 效率低 |

### 1.2 显存需求估算

```
Qwen-2.5-14B-Instruct 模型大小：
- FP16: ~28 GB
- BF16: ~28 GB
- INT8: ~14 GB
- INT4: ~7 GB

训练时显存需求（14B模型）：
- 模型参数: 28 GB (BF16)
- 优化器状态: ~56 GB (Adam)
- 梯度: ~28 GB
- 激活值: ~10-20 GB (取决于batch size)
- 总计: ~120+ GB (需要多卡并行)

推理时显存需求（14B模型）：
- 模型参数: ~14 GB (INT8)
- KV Cache: ~4-8 GB (取决于序列长度)
- 总计: ~18-22 GB
```

### 1.3 可行性结论

**直接复现14B模型：❌ 不可行**
- 8x24GB = 192GB 总显存
- 14B模型训练需要 120+GB
- 推理需要 18-22GB/引擎
- 无法同时运行多个vLLM引擎

**推荐方案：✅ 使用7B模型**
- 7B模型 BF16: ~14 GB
- 7B模型 INT8: ~7 GB
- 8x4090 可以支持训练和推理

---

## 2. 原始项目配置分析

### 2.1 原始训练配置

```bash
# scripts/treerl-qw14b.sh 核心参数
--pretrain qwen-14b-2.5-instruct          # 14B模型
--actor_num_nodes 2                        # 2个节点
--actor_num_gpus_per_node 8                # 每节点8卡
--vllm_num_engines 16                      # 16个vLLM引擎
--vllm_tensor_parallel_size 1              # 每个引擎1卡
--zero_stage 3                             # ZeRO Stage 3
--micro_train_batch_size 1                 # 微批大小
--train_batch_size 256                     # 训练批大小 (16*16)
--generate_max_len 8192                    # 最大生成长度
--num_trace_per_sample 16                  # 每样本16条轨迹
```

### 2.2 资源分配分析

```
原始配置（16卡A100）：
┌─────────────────────────────────────────────────────────────┐
│  Actor 训练: 16 GPU (ZeRO-3)                                │
│  vLLM 推理: 16 GPU (每个引擎1卡)                             │
│  Reference: 与Actor共享权重，不需要额外GPU                    │
│  Reward: 远程服务，不需要本地GPU                             │
└─────────────────────────────────────────────────────────────┘

问题：Actor训练和vLLM推理需要同时占用GPU
     但ZeRO-3允许模型分片，可以部分共享
```

### 2.3 关键瓶颈

1. **显存瓶颈**：14B模型在24GB卡上无法运行
2. **带宽瓶颈**：PCIe互联比NVLink慢4-8倍
3. **引擎数量**：16个引擎需要16张卡，4090无法支持

---

## 3. 适配方案设计

### 3.1 模型选择

**推荐：Qwen-2.5-7B-Instruct**

| 对比项 | Qwen-2.5-7B | Qwen-2.5-14B |
|---|---|---|
| 参数量 | 7B | 14B |
| FP16大小 | 14 GB | 28 GB |
| INT8大小 | 7 GB | 14 GB |
| INT4大小 | 3.5 GB | 7 GB |
| 4090可训练 | ✅ 是 | ❌ 困难 |
| 预期性能 | 基准 | +5-10% |

### 3.2 适配后的配置

```bash
# 适配后的配置
--pretrain qwen-2.5-7b-instruct           # 7B模型 (替代14B)
--actor_num_nodes 1                        # 1个节点 (替代2)
--actor_num_gpus_per_node 8                # 每节点8卡
--vllm_num_engines 4                       # 4个vLLM引擎 (替代16)
--vllm_tensor_parallel_size 2              # 每引擎2卡 (替代1)
--zero_stage 2                             # ZeRO Stage 2 (替代3)
--micro_train_batch_size 2                 # 增加微批大小
--train_batch_size 64                      # 64 = 4*16
--generate_max_len 4096                    # 减少最大长度
--num_trace_per_sample 8                   # 减少轨迹数
```

### 3.3 GPU资源分配

```
适配后配置（8卡4090）：
┌─────────────────────────────────────────────────────────────┐
│  GPU 0-3: Actor 训练 (ZeRO-2, 4卡)                          │
│  GPU 4-5: vLLM 引擎 1 (Tensor Parallel 2)                  │
│  GPU 6-7: vLLM 引擎 2 (Tensor Parallel 2)                  │
│                                                             │
│  或者：                                                     │
│  GPU 0-5: Actor 训练 (ZeRO-2, 6卡)                          │
│  GPU 6-7: vLLM 引擎 (Tensor Parallel 2)                    │
│                                                             │
│  最优方案：                                                  │
│  GPU 0-7: Actor 训练 (ZeRO-2, 8卡)                          │
│  推理时切换到 vLLM 模式                                      │
└─────────────────────────────────────────────────────────────┘
```

### 3.4 内存优化策略

```python
# 1. 混合精度训练
--bf16                    # 使用BF16
--fp16                    # 或使用FP16

# 2. 梯度检查点
--gradient_checkpointing
--gradient_checkpointing_use_reentrant

# 3. 优化器offload
--adam_offload

# 4. 模型量化（可选）
--load_in_4bit           # 4bit量化
--load_in_8bit           # 8bit量化

# 5. LoRA微调（可选）
--lora_rank 16
--lora_alpha 32
```

---

## 4. 分阶段复现计划

### Phase 1: 环境搭建与基础验证 (Week 1)

```
目标：搭建完整的开发环境，验证基础功能

任务清单：
├── [ ] 1.1 环境准备
│   ├── [ ] 安装 CUDA 12.1+ 和 cuDNN
│   ├── [ ] 安装 Python 3.10+
│   ├── [ ] 创建 conda 环境
│   └── [ ] 安装依赖包
│
├── [ ] 1.2 代码克隆与配置
│   ├── [ ] 克隆 TreeRL 仓库
│   ├── [ ] 安装项目依赖
│   ├── [ ] 配置环境变量
│   └── [ ] 验证代码可运行
│
├── [ ] 1.3 数据准备
│   ├── [ ] 下载训练数据 (train_30k.jsonl)
│   ├── [ ] 下载评估数据
│   ├── [ ] 验证数据格式
│   └── [ ] 数据预处理
│
└── [ ] 1.4 单卡验证
    ├── [ ] 加载7B模型
    ├── [ ] 测试推理功能
    ├── [ ] 测试训练一步
    └── [ ] 验证显存占用
```

### Phase 2: 模型适配与调试 (Week 2)

```
目标：适配代码支持7B模型和8卡4090环境

任务清单：
├── [ ] 2.1 模型下载与转换
│   ├── [ ] 下载 Qwen-2.5-7B-Instruct
│   ├── [ ] 测试模型加载
│   ├── [ ] 验证tokenizer
│   └── [ ] 测试生成质量
│
├── [ ] 2.2 训练脚本适配
│   ├── [ ] 修改 train_reinforce_ray.py
│   ├── [ ] 调整显存相关参数
│   ├── [ ] 适配ZeRO-2配置
│   └── [ ] 测试单卡训练
│
├── [ ] 2.3 推理引擎适配
│   ├── [ ] 配置vLLM引擎
│   ├── [ ] 调整Tensor Parallel
│   ├── [ ] 测试推理性能
│   └── [ ] 验证生成质量
│
└── [ ] 2.4 集成测试
    ├── [ ] 测试完整训练循环
    ├── [ ] 验证树搜索功能
    ├── [ ] 检查显存使用
    └── [ ] 性能基准测试
```

### Phase 3: 完整训练运行 (Week 3-4)

```
目标：在8卡4090上完成完整训练

任务清单：
├── [ ] 3.1 小规模测试
│   ├── [ ] 使用100条数据测试
│   ├── [ ] 验证训练流程
│   ├── [ ] 监控显存使用
│   └── [ ] 调整超参数
│
├── [ ] 3.2 中规模训练
│   ├── [ ] 使用1000条数据训练
│   ├── [ ] 验证收敛性
│   ├── [ ] 监控训练指标
│   └── [ ] 调整学习率
│
├── [ ] 3.3 完整训练
│   ├── [ ] 使用30k数据训练
│   ├── [ ] 监控WandB指标
│   ├── [ ] 定期保存checkpoint
│   └── [ ] 记录训练日志
│
└── [ ] 3.4 模型评估
    ├── [ ] 在MATH500上评估
    ├── [ ] 在Omni-MATH上评估
    ├── [ ] 对比基线性能
    └── [ ] 分析结果
```

### Phase 4: 优化与总结 (Week 5)

```
目标：优化性能，总结经验

任务清单：
├── [ ] 4.1 性能优化
│   ├── [ ] 优化训练速度
│   ├── [ ] 优化推理速度
│   ├── [ ] 减少显存占用
│   └── [ ] 记录优化过程
│
├── [ ] 4.2 结果分析
│   ├── [ ] 分析训练曲线
│   ├── [ ] 分析评估结果
│   ├── [ ] 对比论文结果
│   └── [ ] 总结改进点
│
├── [ ] 4.3 文档编写
│   ├── [ ] 编写复现报告
│   ├── [ ] 记录遇到的问题
│   ├── [ ] 总结解决方案
│   └── [ ] 提供改进建议
│
└── [ ] 4.4 开源贡献
    ├── [ ] 整理代码和配置
    ├── [ ] 编写README
    ├── [ ] 准备GitHub仓库
    └── [ ] 分享复现经验
```

---

## 5. 详细执行步骤

### 5.1 环境搭建

```bash
# 1. 创建conda环境
conda create -n treerl python=3.10
conda activate treerl

# 2. 安装PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. 安装项目依赖
cd /data/home/yizhou/TreeRL
pip install -r requirements.txt

# 4. 安装vLLM
pip install vllm

# 5. 安装DeepSpeed
pip install deepspeed

# 6. 安装Ray
pip install "ray[default]"

# 7. 验证安装
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```

### 5.2 数据准备

```bash
# 1. 检查数据文件
ls -la datasets/train/train_30k.jsonl
wc -l datasets/train/train_30k.jsonl

# 2. 验证数据格式
head -5 datasets/train/train_30k.jsonl | python -m json.tool

# 3. 创建数据索引（可选）
python -c "
import json
data = []
with open('datasets/train/train_30k.jsonl', 'r') as f:
    for line in f:
        data.append(json.loads(line))
print(f'Total samples: {len(data)}')
print(f'Sample keys: {data[0].keys()}')
"
```

### 5.3 模型下载

```bash
# 1. 使用huggingface-cli下载
pip install huggingface_hub
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ./models/qwen2.5-7b-instruct

# 2. 或者使用modelscope下载（国内更快）
pip install modelscope
modelscope download --model qwen/Qwen2.5-7B-Instruct --local_dir ./models/qwen2.5-7b-instruct

# 3. 验证模型
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('./models/qwen2.5-7b-instruct')
tokenizer = AutoTokenizer.from_pretrained('./models/qwen2.5-7b-instruct')
print('Model loaded successfully!')
print(f'Model parameters: {model.num_parameters()/1e9:.1f}B')
"
```

### 5.4 单卡测试

```python
# test_single_gpu.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_inference():
    """测试单卡推理"""
    model_path = "./models/qwen2.5-7b-instruct"
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 测试生成
    prompt = "Solve the equation: 2x + 3 = 7"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    
    # 检查显存使用
    print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"GPU Memory Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")

if __name__ == "__main__":
    test_inference()
```

### 5.5 适配训练脚本

```bash
# 创建适配后的训练脚本
cat > scripts/treerl-qwen7b-4090.sh << 'EOF'
set -x

# 适配8卡4090的配置
NUM_TRACE=8
KL=0.0

TAG=qwen7b-4090-${NUM_TRACE}
SAVE_DIR=/data/home/yizhou/TreeRL/checkpoints/$TAG
mkdir -p $SAVE_DIR

DATASETS="
    datasets/train/train_30k.jsonl,1
"

# 启动Ray集群
ray start --head --num-gpus=8

# 提交训练任务
ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json='{
        "working_dir": ".",
        "pip": "requirements.txt",
        "env_vars": {
            "NCCL_TIMEOUT_MS": "3600000"
        }
    }' \
    -- python train_reinforce_ray.py \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 4 \
    --reward_num_nodes 0 \
    --reward_num_gpus_per_node 0 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 4 \
    --vllm_num_engines 2 \
    --vllm_tensor_parallel_size 2 \
    --pretrain ./models/qwen2.5-7b-instruct \
    --reward_pretrain ./models/qwen2.5-7b-instruct \
    --save_path $SAVE_DIR \
    --ckpt_path $SAVE_DIR \
    --micro_train_batch_size 2 \
    --train_batch_size $((NUM_TRACE * 4)) \
    --micro_rollout_batch_size 1 \
    --generation_batch_size 8 \
    --inference_batch_size 1 \
    --rollout_batch_size 4 \
    --num_episodes 2 \
    --prompt_max_len 1024 \
    --generate_max_len 4096 \
    --zero_stage 2 \
    --adam_offload \
    --bf16 \
    --actor_learning_rate 1.5e-6 \
    --lr_scheduler_type cosine \
    --init_kl_coef $KL \
    --prompt_data $DATASETS \
    --max_samples 30000 \
    --normalize_reward \
    --top_p 0.95 \
    --temperature 1.2 \
    --actor_init_on_gpu \
    --num_trace_per_sample $NUM_TRACE \
    --gradient_checkpointing \
    --input_key text \
    --save_steps 50 \
    --wandb_run_name $TAG \
    --task_type qwen-math-reinforce \
    --label_key label \
    --source_key data_type \
    --normalize_reward_from_multi_traces_with_rloo \
    --wandb_project treerl-4090 \
    --use_entropy_tree \
    --m 4 \
    --n 2 \
    --l 1 \
    --t 2 \
    --use_pure_binary \
    --use_weighted_value \
    --weighted_value_style sqrt \
    --gradient_checkpointing_use_reentrant

# 停止Ray集群
ray stop
EOF

chmod +x scripts/treerl-qwen7b-4090.sh
```

---

## 6. 常见问题与解决方案

### 6.1 显存不足 (OOM)

**问题**：`CUDA out of memory`

**解决方案**：
```bash
# 1. 减小batch size
--micro_train_batch_size 1
--micro_rollout_batch_size 1

# 2. 减少序列长度
--generate_max_len 2048
--prompt_max_len 512

# 3. 使用更激进的量化
--load_in_4bit

# 4. 增加offload
--adam_offload
--gradient_checkpointing

# 5. 减少vLLM引擎数量
--vllm_num_engines 1
```

### 6.2 训练速度慢

**问题**：训练速度远低于预期

**解决方案**：
```bash
# 1. 使用Flash Attention
pip install flash-attn

# 2. 优化数据加载
--dataloader_num_workers 4

# 3. 使用更高效的优化器
--optim adamw_torch

# 4. 减少日志频率
--logging_steps 10

# 5. 监控GPU使用率
nvidia-smi -l 1
```

### 6.3 vLLM启动失败

**问题**：vLLM引擎无法启动

**解决方案**：
```python
# 1. 检查端口占用
netstat -tuln | grep 8000

# 2. 检查GPU可用性
python -c "import torch; print(torch.cuda.mem_get_info())"

# 3. 减少vLLM内存占用
--vllm_gpu_memory_utilization 0.7

# 4. 使用单引擎模式
--vllm_num_engines 1
--vllm_tensor_parallel_size 1
```

### 6.4 远程Reward模型不可用

**问题**：无法连接远程Reward模型

**解决方案**：
```python
# 1. 使用本地Reward模型（可选）
# 修改代码使用本地模型

# 2. 使用规则奖励
--use_pure_binary

# 3. 跳过Reward模型
--use_state_value_reward

# 4. 自定义奖励函数
def custom_reward_fn(prompt, response, label):
    # 实现简单的规则奖励
    if label in response:
        return 1.0
    return 0.0
```

### 6.5 Ray集群问题

**问题**：Ray集群无法启动或连接

**解决方案**：
```bash
# 1. 清理Ray状态
ray stop
ray start --head

# 2. 检查Ray状态
ray status

# 3. 查看Ray日志
ls /tmp/ray/session_latest/logs/

# 4. 重置Ray
ray stop
rm -rf /tmp/ray
ray start --head --num-gpus=8
```

---

## 7. 时间规划与里程碑

### 7.1 总体时间线

```
Week 1: 环境搭建与基础验证
├── Day 1-2: 环境安装与配置
├── Day 3-4: 数据准备与验证
└── Day 5-7: 单卡测试与验证

Week 2: 模型适配与调试
├── Day 8-9: 模型下载与测试
├── Day 10-11: 训练脚本适配
├── Day 12-13: 推理引擎配置
└── Day 14: 集成测试

Week 3-4: 完整训练运行
├── Day 15-17: 小规模测试
├── Day 18-21: 中规模训练
├── Day 22-25: 完整训练
└── Day 26-28: 模型评估

Week 5: 优化与总结
├── Day 29-30: 性能优化
├── Day 31-32: 结果分析
├── Day 33-34: 文档编写
└── Day 35: 开源贡献
```

### 7.2 关键里程碑

| 里程碑 | 时间 | 验证标准 |
|---|---|---|
| **M1: 环境就绪** | Week 1 结束 | 能加载模型并生成文本 |
| **M2: 单卡训练** | Week 2 中期 | 能完成1个epoch训练 |
| **M3: 多卡训练** | Week 2 结束 | 8卡训练无OOM |
| **M4: 小规模验证** | Week 3 中期 | 100条数据训练收敛 |
| **M5: 完整训练** | Week 4 结束 | 30k数据训练完成 |
| **M6: 性能达标** | Week 5 结束 | 评估结果合理 |

### 7.3 成功标准

```
必须达成：
✅ 完成完整训练流程
✅ 模型在评估集上有效果
✅ 训练过程稳定，无崩溃
✅ 记录完整的训练日志

期望达成：
🎯 性能接近论文结果的80%
🎯 训练速度合理（<2小时/epoch）
🎯 代码可复现
🎯 文档完整

加分项：
⭐ 优化训练速度
⭐ 改进模型性能
⭐ 开源贡献
⭐ 技术博客
```

### 7.4 风险评估与应对

| 风险 | 概率 | 影响 | 应对措施 |
|---|---|---|---|
| 显存不足 | 高 | 高 | 使用更小模型，增加量化 |
| 训练不收敛 | 中 | 高 | 调整学习率，检查数据 |
| 速度太慢 | 中 | 中 | 优化代码，减少batch |
| 代码bug | 中 | 中 | 详细调试，参考原始代码 |
| 硬件故障 | 低 | 高 | 定期备份，使用云服务 |

---

## 附录：参考资源

### A. 官方资源
- TreeRL GitHub: https://github.com/THUDM/TreeRL
- Qwen-2.5-7B: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
- OpenRLHF: https://github.com/OpenLLMAI/OpenRLHF
- vLLM: https://github.com/vllm-project/vllm

### B. 学习资源
- ZeRO 论文: ZeRO: Memory Optimizations Toward Training Trillion Parameter Models
- PPO 论文: Proximal Policy Optimization Algorithms
- RLHF 论文: Training language models to follow instructions with human feedback

### C. 工具资源
- WandB: https://wandb.ai/
- TensorBoard: https://www.tensorflow.org/tensorboard
- NSight Systems: NVIDIA GPU profiling tool

---

*Last Updated: 2026-07-04*
*Target Hardware: 8x RTX 4090 (24GB)*
*Target Model: Qwen-2.5-7B-Instruct*
