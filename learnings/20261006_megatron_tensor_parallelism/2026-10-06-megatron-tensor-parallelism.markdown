---
layout: post
title:  "Parallelism Basics - Megatron-style tensor parallelism"
date:   2026-10-06 00:00:00 +0530
categories: deep-learning parallelism LLM
mathjax: true
---

Lately, I have been reading about how Megatron-LM splits a transformer layer across GPUs, and this note is about tensor parallelism.

Tensor parallelism means splitting a single matrix multiplication across workers (GPUs), so that each worker holds only a part of the weight. There are two ways to split a matmul: by columns and by rows. Megatron-style tensor parallelism pairs the two so that the workers talk to each other only once.

Before diving in, let's look at a plain matrix multiplication. Let's say `A` is 2X4 and `B` is 4X2. Each element of the output `C` is a dot product of a row of `A` and a column of `B`:

$$
c_{00} = a_{00}b_{00} + a_{01}b_{10} + a_{02}b_{20} + a_{03}b_{30}
$$

![Matrix multiplication setup](https://raw.githubusercontent.com/mrtpk/kaizen/master/learnings/20261006_megatron_tensor_parallelism/resources/tp-page1.png)

## Column parallel
In column parallel, `A` is copied to each worker and `B` is split by columns. Worker 1 computes `C1 = A @ B1` and worker 2 computes `C2 = A @ B2`. Each of them is a complete column of `C`, so we get the result by concatenating them: `C = [C1 | C2]`.

![Column parallelism](https://raw.githubusercontent.com/mrtpk/kaizen/master/learnings/20261006_megatron_tensor_parallelism/resources/tp-page2.png)

## Row parallel
In row parallel, `B` is split by rows. Since the rows of `B` are the contracting dimension, we have to split the columns of `A` too, into matching blocks. Worker 1 computes `A1 @ B1` and worker 2 computes `A2 @ B2`. This time each result is a partial sum (2X2, the same shape as `C`), so we get the result by adding them: `C = C1 + C2`. Across GPUs, this sum is an all-reduce.

![Row parallelism](https://raw.githubusercontent.com/mrtpk/kaizen/master/learnings/20261006_megatron_tensor_parallelism/resources/tp-page3.png)

## Megatron-style tensor parallelism
Now that we know both, let's look at two matmuls in a row, like the MLP block in a transformer:

$$
Y = GeLU(X W_1) W_2
$$

The first matmul is column parallel. Its output stays sharded across the workers and is not gathered. The second matmul is row parallel, and each worker multiplies its shard with the matching rows of `W2`. The partial outputs are summed with one all-reduce at the end.

![Megatron-style tensor parallelism](https://raw.githubusercontent.com/mrtpk/kaizen/master/learnings/20261006_megatron_tensor_parallelism/resources/tp-page4.png)

Why column first? GeLU is applied to each element on its own, so a worker can apply it to its own columns without knowing anything about the other columns. If we split the first matmul by rows instead, each worker holds a partial sum, and $GeLU(a + b) \neq GeLU(a) + GeLU(b)$. We would have to all-reduce before the activation. It means that the order matters: column parallel then row parallel needs one all-reduce, and the other order needs an extra one in the middle.

The same pattern is used in the attention layer. The Q, K, V projections are column parallel (each worker gets its own attention heads), and the attention output projection is row parallel.

![The whole idea in one picture](https://raw.githubusercontent.com/mrtpk/kaizen/master/learnings/20261006_megatron_tensor_parallelism/resources/tp-page5.png)

Let's look at the code. Each split is checked against the single-worker matmul.

{% highlight python %}
import numpy as np

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

# Megatron-style MLP - Y = gelu(X @ W1) @ W2
X = np.random.rand(2, 4) # 2 tokens, hidden size 4
W1 = np.random.rand(4, 8) # first projection, 4 -> 8
W2 = np.random.rand(8, 4) # second projection, 8 -> 4
ref = gelu(X @ W1) @ W2

W1_1, W1_2 = W1[:, :4], W1[:, 4:] # column parallel
W2_1, W2_2 = W2[:4, :], W2[4:, :] # row parallel
Z1 = gelu(X @ W1_1) # worker 1, still sharded
Z2 = gelu(X @ W1_2) # worker 2, still sharded
Y = Z1 @ W2_1 + Z2 @ W2_2 # the only communication, all-reduce
print("Megatron MLP. Is pass?", np.allclose(ref, Y)) # True

# Row parallel first - activation applied on partial sums
X1, X2 = X[:, :2], X[:, 2:]
W1_r1, W1_r2 = W1[:2, :], W1[2:, :]
Z = gelu(X1 @ W1_r1) + gelu(X2 @ W1_r2)
print("Row parallel first. Is pass?", np.allclose(gelu(X @ W1), Z)) # False
{% endhighlight %}

The full script, with column and row parallel checked separately, is available [here](https://github.com/mrtpk/kaizen/tree/master/learnings/20261006_megatron_tensor_parallelism).

So, in short: column parallel keeps the output sharded, row parallel sums the partial outputs, and pairing them means we communicate only when needed.

**Reference:**
+ [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism][megatron-lm]
+ [Dive into Tensor Parallelism: Building ColumnParallelLinear and RowParallelLinear from Scratch][dive-into-tp]

[megatron-lm]: https://arxiv.org/abs/1909.08053
[dive-into-tp]: https://medium.com/@zdj0712/dive-into-tensor-parallelism-building-columnparallellinear-and-rowparallellinear-from-scratch-cf68ce7332d8
