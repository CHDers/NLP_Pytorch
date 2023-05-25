import torch
import matplotlib.pyplot as plt
import numpy as np
from rich import print

# loss = w * 2

# 随机初始化
w = torch.tensor([99.995], requires_grad=True)

# 训练
loop = []

for i in range(10000):
    # 前向传播
    loss = torch.pow(w, 2)
    loop.append(loss.cpu().detach().numpy())
    print(f"eopch={i}, w={w}, loss={loss}")
    # 反向传播
    loss.backward()
    # 梯度下降
    w.requires_grad_(False).add_(-0.001*w.grad).requires_grad_(True)
    w.grad.zero_()

plt.plot(loop)
plt.show()