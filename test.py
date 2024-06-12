import torch

# Tensor ban đầu có 3 chiều
tensor_3d = torch.tensor([[[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9]]])

# Chuyển đổi sang tensor 2 chiều bằng phương thức reshape
tensor_2d = torch.reshape(tensor_3d, (tensor_3d.size(0), -1))
print("Tensor 2 chiều (reshape):", tensor_2d)
