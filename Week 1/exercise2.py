import torch

tensor = torch.rand(1, 2, 3, 4)

print('----- <<< Torch Permute Operations >>> -----')
print(f'Initial tensor of shape {tensor.shape}:\n', tensor)
permute1 = torch.permute(tensor, (1, 2, 3, 0))
print(f'Permuted tensor of shape {permute1.shape}:\n', permute1)
permute2 = torch.permute(tensor, (2, 0, 3, 1))
print(f'Permuted tensor of shape {permute2.shape}:\n', permute2)
