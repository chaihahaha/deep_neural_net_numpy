from activate_functions import *
import matplotlib.pyplot as plt
train_samples = np.loadtxt("TrainSamples.csv", delimiter=",")
train_label = np.loadtxt("TrainLabels.csv", delimiter=",")
m, n0 = train_samples.shape

# 设置网络结构
alpha0 = 0.5  # 学习率
decay_rate = 0.001
lambda_ = 0  # 正则化参数
beta = 0.1
L = 5        # 网络层数
n = [n0, 10, 15, 14, 12, 10]   # 各层神经元个数
W, dW, b, db, Z, dZ, A, dA, dW_old, db_old = [[0]*(L + 1) for i in range(10)]
g = [linear, leakyReLU, leakyReLU, leakyReLU, leakyReLU, sigmoid]   # 各层激活函数
g_prime = [linear, d_leakyReLU, d_leakyReLU, d_leakyReLU, d_leakyReLU, d_sigmoid]  # 导数

# 编码标签
labelset = np.unique(train_label)
y = np.zeros((len(labelset), len(train_label)))
for i in range(len(labelset)):
    y[i, train_label == labelset[i]] = 1

# 初始化网络参数
for l in range(1, L + 1):
    W[l] = np.random.randn(n[l], n[l - 1]) * 0.1
    dW[l] = np.zeros((n[l], n[l - 1]))
    dW_old[l] = np.zeros((n[l], n[l - 1]))
    b[l] = np.random.randn(n[l], 1) * 0.1
    db[l] = np.zeros((n[l], 1))
    db_old[l] = np.zeros((n[l], 1))
    Z[l] = np.zeros((n[l], m))
    A[l] = np.zeros((n[l], m))
    dZ[l] = np.zeros((n[l], m))
    dA[l] = np.zeros((n[l], m))

losses = []

A[0] = train_samples.T

# 规格化 error
mu = np.mean(A[0], axis=1, keepdims=True)         # shared
A[0] -= mu
sigma_square = np.sum(A[0]**2, axis=0) / m    # shared
A[0] /= sigma_square

for epoch in range(1000):
    # 前向传播
    for l in range(1, L + 1):
        Z[l] = np.dot(W[l], A[l - 1]) + b[l]
        A[l] = g[l](Z[l])

    losses.append(loss(y, A[L]))
    print("epoch: ", epoch, " loss: ", losses[-1])

    # 反向传播
    dA[L] = d_loss(y, A[L])
    for l in range(L, 0, -1):
        alpha = 1 / (1 + decay_rate * epoch) * alpha0
        dZ[l] = dA[l] * g_prime[l](Z[l])
        dW[l] = beta * dW_old[l] + (1 - beta) * np.dot(dZ[l], A[l-1].T)/m
        db[l] = beta * db_old[l] + (1 - beta) * np.sum(dZ[l], axis=1, keepdims=True)/m
        dA[l-1] = np.dot(W[l].T, dZ[l])
        Z[l] -= alpha * dZ[l]
        W[l] -= alpha * (dW[l] + lambda_ * W[l] / m)
        b[l] -= alpha * db[l]
        dW_old[l] = dW[l].copy()
        db_old[l] = db[l].copy()

# 解码标签
output_label = labelset[np.argmax(A[L], axis=0)]

print("正确率: ", np.sum(output_label == train_label)/len(train_label))
plt.plot(losses)
plt.show()
