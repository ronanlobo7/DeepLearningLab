import torch

tensor1 = torch.rand(2, 3)
tensor2 = torch.rand(2, 3)

print('\n----- <<< Maximum and Minimum Values of Tensors >>> -----')
print(f'Tensor 1 of shape {tensor1.shape}:\n', tensor1)
print('Maximum value in Tensor 1: ', torch.max(tensor1))
print('Minimum value in Tensor 1: ', torch.min(tensor1))

print(f'\nTensor 2 of shape {tensor2.shape}:\n', tensor2)
print('Maximum value in Tensor 2: ', torch.max(tensor2))
print('Minimum value in Tensor 2: ', torch.min(tensor2))
