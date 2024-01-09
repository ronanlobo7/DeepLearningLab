import torch

tensor1 = torch.rand(2, 3)
tensor2 = torch.rand(2, 3)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('\n----- <<< Tensors on GPU >>> -----')
print(f'Initial Tensors on device {tensor1.device}')
print(f'Tensor 1 of shape {tensor1.shape}:\n', tensor1)
print(f'Tensor 2 of shape {tensor2.shape}:\n', tensor2)

tensor1_gpu = tensor1.to(device)
tensor2_gpu = tensor2.to(device)

print(f'\nTensors now on device {tensor1_gpu.device}')
print(f'Tensor 1 of shape {tensor1_gpu.shape}:\n', tensor1_gpu)
print(f'Tensor 2 of shape {tensor2_gpu.shape}:\n', tensor2_gpu)
