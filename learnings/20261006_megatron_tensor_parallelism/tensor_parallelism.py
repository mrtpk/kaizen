import numpy as np

np.random.seed(0)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

# A matmul on a single worker is the reference
A = np.random.rand(2, 4)
B = np.random.rand(4, 2)
ref = A @ B

# Column parallel - A is copied to each worker, B is split by columns
B1, B2 = B[:, :1], B[:, 1:]
C1 = A @ B1 # worker 1, 2X1
C2 = A @ B2 # worker 2, 2X1
C = np.concatenate([C1, C2], axis=1) # gather
print("Column parallel. Is pass?", np.allclose(ref, C))

# Row parallel - A is split by columns, B is split by rows (the contracting dimension)
A1, A2 = A[:, :2], A[:, 2:]
B1, B2 = B[:2, :], B[2:, :]
C1 = A1 @ B1 # worker 1, partial output 2X2
C2 = A2 @ B2 # worker 2, partial output 2X2
C = C1 + C2 # sum, i.e. all-reduce
print("Row parallel. Is pass?", np.allclose(ref, C))

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
print("Megatron MLP. Is pass?", np.allclose(ref, Y))

# Why column parallel first? Row parallel gives partial sums and gelu(a + b) != gelu(a) + gelu(b)
X1, X2 = X[:, :2], X[:, 2:]
W1_r1, W1_r2 = W1[:2, :], W1[2:, :]
Z = gelu(X1 @ W1_r1) + gelu(X2 @ W1_r2) # activation applied before the sum
print("Row parallel first. Is pass?", np.allclose(gelu(X @ W1), Z))
