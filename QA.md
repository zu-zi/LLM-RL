### QA
#### 在 LLM 问题中，举出具体的 s, a, r 定义的例子（最好多个），以及一个 episode 的长度是多少，episode 终止的标准是什么
**在 LLM 问题中，s（状态）、a（动作）、r（奖励）:**奖励可以是人工设计匹配，人工打分，也可以是reward_model打分
+ 例1：LLM对话生成:
     + s：当前对话历史（包括用户所有输入和模型之前所有回复），用文本或其编码表示。
     + a：模型生成的下一个回复文本（通常是一个完整的回答，或者是生成下一个token）。
     + r：根据人类反馈（比如对回复的满意度评分），或自动评价指标（比如回答的相关度、流畅度、礼貌度）计算的分数或reward_model给出的分数。
+ 例2：代码生成任务中的RL优化
     + s:当前给定的代码上下文，包括函数定义、注释和之前生成的代码片段
     + a:生成下一个代码token或者下一行代码
     + r：代码是否能通过单元测试（通过得高分，未通过得低分或负分），或代码简洁性、执行效率等指标
+ 例3：基于LLM的游戏AI
     + s：当前游戏环境的文本描述及历史动作
     + a：生成游戏命令，如“go north”, “pick up key”
     + 奖励 r：游戏内积分变化、任务完成状态，或者探索新区域的奖励
**episode**：一次完整流程
+ 长度：通常是动作序列的长度，也就是模型生成的文本长度或完成一个完整任务的步骤数
+ 终止标准：达到预定最大生成长度；任务成功完成；任务失败或异常；模型生成特定的终止符
#### advantage 部分的计算，是否传递梯度去更新模型参数，为什么
否。
+ 用 reward、value function 计算 advantage 时，会对 value function 的参数求梯度，用来训练 critic（让它拟合正确的价值）；计算 policy loss 时，advantage 是一个**常数**（stop gradient），不反传梯度到 critic，也不更新 policy 之外的部分
+ 因为它是**评估信号**，不是可微的奖励函数
+ 根据公式A作为权重，它的来源和计算**不依赖于当前 policy 的梯度链路**，而是来自 采样的轨迹。如果让 advantage 参与梯度回传，会破坏无偏估计的性质
+ 如果 advantage 对 policy 参数可微，policy 可能会通过修改 value function 来“**作弊**”，使 advantage 偏大，从而得到更高的 policy 更新信号，而不是学习真正好的策略
+ PPO 的 clip loss 本来就是为了抑制训练不稳定，如果 advantage 参与梯度，会额外引入**不稳定因素**
+ 补充：即使critic，回传 value 预测的梯度，但 advantage 本身不反传
#### loss 中 pi_new 比 pi_old 的计算，具体实现中为什么常在对数域（log）进行计算
##### 如果直接用概率计算：是多个 token 概率相乘得到的非常小的数：下溢出，变成0，没梯度，梯度消失

##### 具体实现中，常在对数域进行计算：
+ 先log减法，再exp
+ 防止下溢/上溢:映射且不丢精度
+ 避免多 token 概率连乘导致的精度损失：乘法变加法，既节省计算又数值稳定：乘法链路容易梯度消失，加法链路在梯度传播上更稳定

##### PPO for LLM：
```
log_probs = torch.log_softmax(logits, dim=-1) # 用 log_softmax 得到每个 token 的 log 概率
log_pi_new = torch.sum(token_log_probs_new, dim=-1)
log_pi_old = torch.sum(token_log_probs_old, dim=-1)
log_ratio = log_pi_new - log_pi_old
ratio = torch.exp(log_ratio)
```
#### PPO 的 clip 为什么是单边的，即实现中对应一个 min 操作
**min() 不是“单边剪裁”，而是 通过 advantage 的正负实现了上下两边的约束**
+ A>0：只限制 ratio 的上界 → 否则会过度提高动作概率。
+ A<0：只限制 ratio 的下界 → 否则会过度降低动作概率。
##### 情况 1：A>0（动作比平均好）
+ 希望增加该动作的概率（ratio > 1）
+ 但为了防止更新过大，设置上限 1+ϵ
+ min() 让超过 1+ϵ 后的目标值不再增加
##### 情况 2：A<0（动作比平均差）
+ 希望减少该动作的概率（ratio < 1）
+ 为了防止更新过大，设置下限 1-ϵ
+ min() 让超过 1-ϵ 后的目标值不再进一步减少
##### PPO for LLM：
```
ratio = torch.exp(logp_new - logp_old)        # r_t
clip_ratio = torch.clamp(ratio, 1 - eps, 1 + eps)
policy_loss = -torch.mean(torch.min(ratio * adv, clip_ratio * adv))
```
#### PPO 中一般没有 Q function，那 value loss 具体是如何计算的，为什么不用 Q 呢
PPO:不显式学习 Q-function，而是直接学习 V(s)，然后用它算 advantage 来更新策略

R=A+V(old)=returns

value_loss = 0.5 * ((V_pred - returns) ** 2).mean()

**为什么不用 Q ?：**不直接学 Q-function 是因为 Q 在大动作空间下难估且 on-policy 样本效率低，而 V(s) 足够用来算 advantage
+ 避免额外动作维度:Q(s,a) 需要对所有可能动作求值，但 LLM 或连续动作空间里，动作空间巨大甚至连续，估计 Q 会非常难。而V(s) 不依赖具体动作，估计更稳定、更高效
+ 直接配合 advantage 计算：A(s,a)=Q(s,a)−V(s)；用 V(s) + 采样轨迹的 reward 就能通过 GAE 得到近似 advantage，不需要显式计算 Q
+ on-policy 的数据效率：每次更新只能用最新策略的数据，如果还要估计 Q(s,a)，需要足够多覆盖动作的样本，样本效率更低
#### KL 是一种不对称运算，那 KL reward 中，为什么我们选 pi_new 和 pi_ref 之间的 KL，而不反过来
KL(πnew∥πref)=E[logπnew-logπref]：PPO中出现在reward中，对变化过大的策略分布惩罚
##### 优化视角：谁是“变量”，谁是“常数”
+ 在训练中，pi_new 的参数是可更新的变量，pi_ref 是固定的常数
+ 期望是对 pi_new 的分布取的，这样梯度能自然地传到 pi_new。反过来不匹配
##### 采样匹配：PPO 数据来自 pi_new
+ KL 的第一个分布是采样来源
+ PPO 是 on-policy 算法，所有采样都来自 pi_new
##### 惩罚的含义不同
+ pi_new 和 pi_ref：惩罚 pi_new 在 pi_ref 低概率区域上分配了过高的概率（更符合“不要乱跑到参考模型不认可的地方”）
+ 反过来：惩罚 pi_new 在 pi_ref 高概率区域没有给足概率，这会强迫 pi_new 必须覆盖 pi_ref 所有高概率输出，降低探索性
##### 数值稳定性和梯度
+ 不会出现这种梯度爆炸，因为 pi_new 是采样来源，不会真的给 0 概率。
+ 反过来：如果 pi_new 在某些 pi_ref 高概率区域给了非常低的概率，会出现 log0 → 无穷大 → 数值不稳定
#### GRPO 的 KL loss 为什么使用 k3 形式来计算
补充背景：
+ k₁ 是最直接的近似形式：log(q/p)，常用于 PPO 中作为 KL 惩罚项的一部分。这个估计是无偏的，但方差较高，尤其在 p 与 q 差距较大时不稳定
+ k₂ 是另一种形式：0.5 * (log(p/q))²，这是一个有偏估计，但在分布接近时方差较低，更稳定
+ k₃，即 (p/q - 1) - log(p/q)，结合了优势和稳定性，并且仍然是无偏的。它实际上等价于 加上一个控制变量 (p/q−1)，这个控制变量的期望值是 0，因此不会引入偏差，却能显著降低方差
**为什么 GRPO 采用 k₃ 表达式**
+ 方差更低，更稳定
+ 数学上无偏

#### GRPO 中，不同 response 的 token 长度差异很大，按现在的方式做平均会有什么问题
sample-level  ->  token-level(全局平均)

+ sample-level (直接平均)	    长样本权重大	   长文本更重要
+ token-level (先平摊再平均)	每条样本贡献平摊	短文本反而相对更重要

#### LLM + RL 中 KL loss 和 KL reward 的区别
KL Reward = “你做的动作偏离参考模型有代价”，通过奖励信号影响策略更新。

KL Loss = “你输出的分布要和参考模型一致”，直接作为监督信号优化模型。

把 KL reward 想象成 RL 的“惩罚项”，KL loss 想象成“训练目标函数的一部分”
##### KL Reward
+ KL reward 用来约束新策略（pi_new）不要偏离参考模型（pi_ref）太远
+ 方向固定
+ 采样自 pi_new：因为 PPO 是 on-policy，采样的数据来自新策略
+ 和策略梯度Loss直接相关：KL reward 会影响 advantage，从而影响 policy gradient 更新
+ 动态权重：可以通过 β 调整 KL 惩罚强度，以平衡探索与保守
+ 采样：pi_new
+ 目标：控制策略偏离，稳定探索与保守平衡
##### KL Loss
+ 显式对 pi_new 与 pi_ref 做监督约束，用于训练模型直接逼近参考策略。不是约束奖励，而是直接控制Loss
+ 更像监督学习：直接最小化分布差异，不依赖 reward
+ 梯度直接反传到模型参数：不是通过 reward，而是直接更新 policy
+ 采样：可以是数据集或任意状态
+ 目标：强制 pi_new 拟合 pi_ref
