# 精读笔记

open-loop execution：直接执行一整个action chunk的动作，不在执行过程中进行observation和judgement。
closed-loop execution：在执行过程中一步一步地不断进行observation和judgement，根据环境反馈调整动作。

自回归（autoregressive）模型：在生成文本时，模型会根据之前生成的结果作为依据来生成接下来的内容。比如在生成一段文本时，模型会根据已经生成的部分来预测接下来的词语。
正则化：对模型进行约束，防止过拟合，提高模型的泛化能力，而不是为了达到效果就随机生成。

on-policy RL：在训练过程中，模型使用当前的策略来生成动作，并根据这些动作的结果来更新策略。
off-policy RL：在训练过程中，模型使用一个独立的策略来生成动作，而使用另一个策略来更新模型。

online RL：模型在训练过程中不断与环境交互，实时更新策略。
offline RL：模型在训练过程中使用预先收集的数据进行训练，而不与环境进行实时交互。

actor-critic方法：一种强化学习算法，其中actor负责生成动作，critic负责评估动作的价值。actor根据critic的反馈来调整自己的策略，以提高整体性能。常见的一种训练Embodied agent的方法。

replay buffer：在强化学习中，replay buffer是一种数据结构，用于存储智能体在环境中经历的状态、动作、奖励和下一个状态等信息。智能体在训练过程中会从replay buffer中随机抽取样本进行学习，这有助于打破数据之间的相关性，提高训练的稳定性和效率。

## 相关的领域内容

VLA领域中常见现在已经使用DiT-based模型来作为action chunk的视觉理解以及generation的model，但是在精细任务上面还是不理想，通常需要去fine-tune模型来适应特定的任务。也就是说泛化性不是很足。

VLA 模型的 RL 微调：更新哪些部分，以及如何将 RL 信号引入模型训练中。

## Related works list：

RECAP [3] 通过基于优势函数条件的策略提取方法，使用离线 RL 对整个 π*0.6 模型进行端到端训练。

## Goal&&method：在不承担完整模型 RL 训练成本的情况下，改进一个预训练 VLA 。

第一，RLT 引入了一个 RL token。它是一个紧凑的读出表征，通过训练来压缩 VLA 的内部嵌入，并作为轻量级 actor-critic 的状态观测使用。这样既保留了 VLA 预训练得到的感知结构，又支持高效的在线学习。

第二，RLT 在与 VLA 原生动作接口对齐的动作块上进行操作。这样可以在高控制频率、稀疏奖励的条件下，缩短时间差分学习中的有效决策时域。相比之下，单步方法会面对更长的 credit assignment，也就是信用分配问题。

第三，RLT 的 actor 并不是预测残差或者潜在噪声，而是直接以 VLA 采样得到的参考动作块作为条件，并通过正则化让输出动作靠近这个参考动作。这样，在线 RL 就变成了对一个较好的 VLA 先验行为策略进行局部精修，而不是无约束搜索，或者对扩散过程进行隐式调制。

## 自回归式重建RL-Token

这篇文章通过用两个transformer作为encoder和decoder来进行压缩表征-RL-Token，进行自回归式的重建。Encoder将image embedding压缩为RL-token，然后根据RL-Token以及前面相关的token进行下一个token的重建。

## Related works list

1. Behavioral Cloning
2. Action Chunking
3. Diffusion Policy / diffusion action generation
4. Autoregressive action generation
5. VLA / OpenVLA / π0 / RT-2 基本范式
6. Off-policy actor-critic
7. Replay buffer
8. TD learning / Q function
9. PPO 为什么是 on-policy
10. Residual policy
11. Human-in-the-loop RL
12. DAgger
13. Diffusion noise space / latent action modulation