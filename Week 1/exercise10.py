import torch

tensor1 = torch.rand(2, 3)
tensor2 = torch.rand(2, 3)

print('\n----- <<< Maximum and Minimum Index Values of Tensors >>> -----')
print(f'Tensor 1 of shape {tensor1.shape}:\n', tensor1)
print('Maximum value index in Tensor 1: ', torch.argmax(tensor1))
print('Minimum value index in Tensor 1: ', torch.argmin(tensor1))

print(f'\nTensor 2 of shape {tensor2.shape}:\n', tensor2)
print('Maximum value index in Tensor 2: ', torch.argmax(tensor2))
print('Minimum value index in Tensor 2: ', torch.argmin(tensor2))
