import torch

# A, B, C are tensors on the GPU
def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N: int):
    a = A.detach().clone()
    b = B.detach().clone()
    C.copy_(a.add(b))
