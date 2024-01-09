import torch

tensor1 = torch.rand(1, 7)
tensor2 = torch.rand(1, 7)

print('----- <<< Tensor Multiplication >>> -----')
print(f'Tensor 1 of shape {tensor1.shape}:\n', tensor1)
print(f'Tensor 2 of shape {tensor2.shape}:\n', tensor2)

transposed2 = tensor2.T
print(f'Transposed Tensor 2 of shape {transposed2.shape}:\n', transposed2)

output_tensor = torch.matmul(tensor1, transposed2)
print(f'Output Tensor after Multiplication of shape {output_tensor.shape}:\n', output_tensor)
