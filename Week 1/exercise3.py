import torch

tensor = torch.rand(3, 3, 3)

print('----- <<< Tensor Indexing Operations >>> -----')
print(f'Initial tensor of shape {tensor.shape}:\n', tensor)

indexed1 = tensor[0]
print(f'Indexed tensor of shape {indexed1.shape}:\n', indexed1)

indexed2 = tensor[:, 0]
print(f'Indexed tensor of shape {indexed2.shape}:\n', indexed2)

indexed3 = tensor[:, :, 0]
print(f'Indexed tensor of shape {indexed3.shape}:\n', indexed3)
