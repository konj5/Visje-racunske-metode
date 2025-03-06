import numpy as np

A = np.random.random((2,4))
B = np.random.random((4,3))

print(A.shape)
print(B.shape)
print(np.einsum("ij,jk->kj", A, B).shape)