# Transformer
## 注意力机制
注意力机制（Attention Mechanism）是一种模仿人类视觉注意力的机制，允许模型在处理输入数据时动态地关注不同部分的信息。它通过计算输入序列中各个位置之间的相关性，生成一个权重分布，从而决定哪些部分的信息更重要。
$$
   Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$
其中，\(Q\)（Query）、\(K\)（Key）和\(V\)（Value）分别表示查询、键和值的矩阵，\(d_k\)是键的维度。通过这种机制，模型能够有效地捕捉输入数据中的长距离依赖关系，提高了在自然语言处理等任务中的表现。
## 自注意力机制
### 词嵌入 （word embedding）
词嵌入是将离散的词汇映射到连续的向量空间中，以捕捉词与词之间的语义关系。在Transformer模型中，输入的词汇通过词嵌入层转换为固定维度的向量表示。这些向量不仅包含了词的语义信息，还能够通过训练学习到词之间的相对位置关系。
可以通过vectorization技术将词汇转换为向量表示，例如使用Word2Vec、GloVe或通过Transformer模型自身的嵌入层进行训练得到的词向量。

自注意力机制（Self-Attention Mechanism）是注意力机制的一种特殊形式，主要用于处理序列数据。它允许序列中的每个位置与序列中的其他位置进行交互，从而捕捉全局的依赖关系。自注意力机制通过计算输入序列中每个位置的查询（Query）、键（Key）和值（Value）之间的相关性，生成一个加权和，作为该位置的输出表示。
output表示为：
$$
   SelfAttention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

自注意力机制在Transformer模型中发挥了关键作用，使其能够高效地处理长序列数据，并在机器翻译、文本生成等任务中取得了显著的效果。

## 多头注意力机制
多头注意力机制（Multi-Head Attention Mechanism）是自注意力机制的扩展，通过并行地计算多个自注意力头（Attention Heads），使模型能够从不同的子空间中捕捉信息。每个注意力头独立地计算查询、键和值之间的相关性，然后将各个头的输出进行拼接和线性变换，得到最终的输出表示。多头注意力机制增强了模型的表达能力，使其能够更好地捕捉复杂的模式和关系。
$$
   MultiHead(Q, K, V) = Concat(head_1, \ldots, head_h)W^O
$$
$$
    head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
$$
其中，\(h\)表示注意力头的数量，\(W_i^Q\ 、W_i^K\)、\(W_i^V\)和\(W^O\)是可学习的权重矩阵。通过多头注意力机制，Transformer模型能够更全面地理解输入数据的结构和语义，提高了在各种自然语言处理任务中的性能。
## Transformer架构
Transformer是一种基于注意力机制的深度学习模型架构，广泛应用于自然语言处理任务。它由编码器（Encoder）和解码器（Decoder）两部分组成，每部分都包含多个相同的层（Layer）。每个层包括多头注意力机制和前馈神经网络（Feed-Forward Neural Network），并使用残差连接（Residual Connection）和层归一化（Layer Normalization）来稳定训练过程。Transformer通过并行处理序列数据，显著提高了训练效率，并在各种任务中取得了优异的性能。
## 位置编码
由于Transformer模型不具备处理序列顺序信息的能力，因此引入了位置编码（Positional Encoding）来为输入序列中的每个位置添加位置信息。位置编码通常采用正弦和余弦函数生成，能够为模型提供关于单词在序列中相对和绝对位置的信息。通过将位置编码与输入嵌入（Input Embedding）相加，Transformer模型能够更好地理解序列数据的结构和顺序。
