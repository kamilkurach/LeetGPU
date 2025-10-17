import torch

# A, B, C are tensors on the GPU
def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, M: int, N: int, K: int):
    a = A.detach().clone()
    b = B.detach().clone()
    torch.matmul(a, b, out=C)
