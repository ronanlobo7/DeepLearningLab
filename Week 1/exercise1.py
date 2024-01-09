import torch

# Reshaping Operations
tensor = torch.arange(0, 8)

print('----- <<< Reshaping Operations >>> -----')
print('Initial Tensor:\n', tensor)
print('Tensor Reshaped to (2, 8):\n', torch.reshape(tensor, (2, 4)))
print('Tensor Reshaped to (2, 2, 4):\n', torch.reshape(tensor, (2, 2, 2)))
print('Initial Tensor is Unchanged:\n', tensor)


# Viewing Operations
tensor = torch.rand(2, 2)

print('\n----- <<< Viewing Operations >>> -----')
print('Initial Tensor:\n', tensor)
tensor_4 = tensor.view(4)
print('Tensor Viewed as (4):\n', tensor_4)
tensor_4[2] = 0.055
print('Element at index 2 changed to 0.55. Tensor being viewed as (4):\n', tensor_4)
print('Initial Tensor:\n', tensor)


# Stacking Operations
tensor1 = torch.rand(2, 2)
tensor2 = torch.rand(2, 2)

print('\n----- <<< Stacking Operations >>> -----')
print('Tensor 1:\n', tensor1)
print('Tensor 2:\n', tensor2)
print('Tensor after stacking Tensors 1 and 2:\n', torch.stack((tensor1, tensor2)))


# Squeeze Operations
tensor = torch.zeros(2, 1, 2, 1, 2)

print('\n----- <<< Squeezing and Unsqueezing Operations >>> -----')
print(f'Initial Tensor of shape {tensor.shape}:\n', tensor)
squeezed = torch.squeeze(tensor)
print(f'Squeezed tensor of shape {squeezed.shape}:\n', squeezed)
unsqeezed = torch.unsqueeze(squeezed, 1)
print(f'Unsqueezed tensor of shape {unsqeezed.shape}:\n', unsqeezed)
