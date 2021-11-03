import numpy as np

arr = np.array([
    [np.random.normal() for i in range(5)] for j in range(5)
])
arr[arr > 0.09] = 42
arr[arr != 42] **= 2
np.set_printoptions(precision=2, suppress=True)
print(arr)
print(arr[:, 3])
