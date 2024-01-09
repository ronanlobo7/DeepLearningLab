import torch

torch.manual_seed(7)

tensor = torch.rand(1, 1, 1, 10)

print('\n----- <<< Dimension Removal >>> -----')
print(f'Initial Tensor of shape {tensor.shape}:\n', tensor)

squeezed = torch.squeeze(tensor)
print(f'Tensor after removing all the 1 dimensions, shape {squeezed.shape}:\n', squeezed)
