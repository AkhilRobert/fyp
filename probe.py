import numpy as np
import torch

x = np.random.rand(128, 128)
print(x.shape)
x = x.reshape(-1, 128, 128)
print(x.shape)
print(x)

s = []

m = np.random.rand(1, 2)
n = np.random.rand(1, 2)

s.append(m)
s.append(n)

s = np.array(s)

print(s.shape)
