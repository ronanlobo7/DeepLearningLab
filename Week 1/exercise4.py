import numpy as np
import torch

np_array = np.random.randn(10)

print('----- <<< Numpy and Tensor Operations >>> -----')
print('Numpy Array:\n', np_array)

tensor = torch.from_numpy(np_array)
print('Tensor from Numpy Array:\n', tensor)

np_array_from_tensor = tensor.numpy()
print('Numpy Array from Tensor:\n', np_array_from_tensor)
