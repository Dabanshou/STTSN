import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

'''您可以将代码中实现 patching 的部分封装成一个新的类，让代码结构更加清晰和模块化。以下是一个可能的实现方式
   Creatded by GPT
'''


class PatchingLayer(nn.Module):
    def __init__(self, padding_patch=None, patch_len=None, stride=None, pad=None):
        super(PatchingLayer, self).__init__()
        self.padding_patch = padding_patch
        self.patch_len = patch_len
        self.stride = stride

        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, pad))

    def forward(self, z):
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # z = z.permute(0, 1, 3, 2)
        return z


class newPatchingLayer(nn.Module):
    def __init__(self, padding_patch=None, patch_len=None, stride=None, pad=None):
        super(newPatchingLayer, self).__init__()
        self.padding_patch = padding_patch
        self.patch_len = patch_len
        self.stride = stride

        if padding_patch == 'end':
            # Instead of using ReplicationPad1d, we'll handle padding manually
            self.pad = pad

    def forward(self, z):
        if self.padding_patch == 'end':
            # Manually apply replication padding
            z = self.replicate_padding(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # z = z.permute(0, 1, 3, 2)
        return z

    def replicate_padding(self, z):
        # Replicate the last value along the time dimension
        batch_size, channels, seq_len = z.size()
        last_frame = z[:, :, -1:]  # Extract the last frame
        padding_frames = last_frame.repeat(1, 1, self.pad)  # Repeat the last frame 'pad' times
        z_padded = torch.cat([z, padding_frames], dim=-1)
        return z_padded


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()

        self.individual = individual  # individual: 一个布尔值，指示是否对每个时间序列的变量（即 n_vars）使用不同的线性层。如果为True，则对每个变量都使用不同的线性层，否则使用共享的线性层
        self.n_vars = n_vars
        self.nf = nf
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            # self.linear2 = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])  # z: [bs x d_model * patch_num]
                z = self.linears[i](z)  # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)  # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            # x = self.linear2(self.linear(x).relu())
            x = self.linear(x)
            x = self.dropout(x)
        return x

class Flatten_Head_strategy(nn.Module):
    def __init__(self, nf, hf, target_window, head_dropout=0):
        super().__init__()
        self.nf = nf

        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, hf)
        self.linear2 = nn.Linear(hf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]

        x = self.flatten(x)
        x = self.linear2(self.linear(x).relu())
        # x = self.linear(x)
        x = self.dropout(x)
        return x

# class PatchTST_backbone(nn.Module):
#     def __init__(self, c_in, context_window, target_window, patch_len, stride, ...):
#         super(PatchTST_backbone, self).__init__()
#
#         # RevIn
#         ...
#
#         # Patching
#         self.patching_layer = PatchingLayer(padding_patch, patch_len, stride)
#
#         # Backbone
#         self.backbone = TSTiEncoder(c_in, patch_num=..., patch_len=..., ...)
#
#         # Head
#         ...
#
#     def forward(self, z):
#         # norm
#         ...
#
#         # do patching
#         z = self.patching_layer(z)
#
#         # model
#         z = self.backbone(z)
#         z = self.head(z)
#
#         # denorm
#         ...
#
#         return z
#
#     def create_pretrain_head(self, head_nf, vars, dropout):
#         ...
class PatchingDifined(nn.Module):
    def __init__(self):
        super(PatchingDifined, self).__init__()
        self.dim1 = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17, 18, 19, 20, 21, 22, 23])
        self.dim2 = np.array([42, 43, 44, 45, 46, 47, 48, 49, 50, 57, 58, 59, 60, 61, 62, 63, 64, 65])
        self.dim3 = np.array([0, 1, 2, 12, 13, 14, 36, 37, 38, 39, 40, 41, 51, 52, 53, 54, 55, 56])

        self.dim4 = np.array([24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 51, 52, 53])
        self.dim5 = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 42, 43, 44, 45, 46, 47, 48, 49, 50])
        self.dim6 = np.array([15, 16, 17, 18, 19, 20, 21, 22, 23, 57, 58, 59, 60, 61, 62, 63, 64, 65])

        self.dim7 = np.array(
            [3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17, 18, 19, 20, 21, 22, 23, 42, 43, 44, 45, 46, 47, 48, 49, 50, 57,
             58, 59, 60, 61, 62, 63, 64, 65])  # 36
        self.dim8 = np.array(
            [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 36, 37, 38, 39, 40, 41, 51, 52, 53, 54, 55, 56])  # 24
        self.dim9 = np.array([24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 51, 52, 53])  # 18

        # self.dim10 = np.array([24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35])  # 12
        # self.dim11 = np.array(
        #     [0, 1, 2, 12, 13, 14, 36, 37, 38, 39, 40, 41, 51, 52, 53, 54, 55, 56])   #18
        dim = 50
        self.fc7 = nn.Linear(36, dim)
        self.fc8 = nn.Linear(24, dim)
        self.fc9 = nn.Linear(18, dim)

    def forward(self, z):
        part1 = z[:, :, self.dim7]
        part2 = z[:, :, self.dim8]
        part3 = z[:, :, self.dim9]

        part1 = self.fc7(part1)
        part2 = self.fc8(part2)
        part3 = self.fc9(part3)

        output = torch.stack([part1, part2, part3], dim=2)
        return output


class Flatten_Head1(nn.Module):
    def __init__(self, head_dropout=0):
        super().__init__()
        # self.nf = nf
        self.flatten = nn.Flatten(start_dim=-2)
        # self.linear = nn.Linear(nf, target_window)
        # self.linear2 = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)
        dim = 50
        self.fc7 = nn.Linear(dim, 36)
        self.fc8 = nn.Linear(dim, 24)
        self.fc9 = nn.Linear(dim, 18)
        self.dim = dim
        self.fc = nn.Linear(150, 66)
        self.dim7 = np.array(
            [3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17, 18, 19, 20, 21, 22, 23, 42, 43, 44, 45, 46, 47, 48, 49, 50, 57,
             58, 59, 60, 61, 62, 63, 64, 65])  # 36
        self.dim8 = np.array(
            [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 36, 37, 38, 39, 40, 41, 51, 52, 53, 54, 55, 56])  # 24
        self.dim9 = np.array([24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 51, 52, 53])  # 18
        self.linear = nn.Linear(dim * 4, dim * 4)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        # part1 = x[:, :, 0:self.dim]
        # part2 = x[:, :, self.dim:self.dim*2]
        # part3 = x[:, :, self.dim*2:self.dim*3]
        part4 = x[:, :, self.dim * 3:self.dim * 4]
        part5 = x[:, :, 0:self.dim * 3]

        # y = torch.zeros_like(part1)
        #
        # part1 = self.fc7(part1)
        # part2 = self.fc8(part2)
        # part3 = self.fc9(part3)
        y = self.fc(part5)

        # F_allT = part4.transpose(-1, -2)
        # A1 = torch.softmax(torch.matmul(F_allT, part1), dim=-1)
        # A2 = torch.softmax(torch.matmul(F_allT, part2), dim=-1)
        # A3 = torch.softmax(torch.matmul(F_allT, part3), dim=-1)

        # y[:, :, self.dim7] = part1
        # y[:, :, self.dim8] = part2
        # y[:, :, self.dim9] = part3
        # y = part4 + torch.matmul(part1, A1) + torch.matmul(part2, A2) + torch.matmul(part3, A3)
        # p1 = part1 + torch.matmul(part4, A1)
        # p2 = part2 + torch.matmul(part4, A2)
        # p3 = part3 + torch.matmul(part4, A3)

        # y[:, :, self.dim7] = p1
        # y[:, :, self.dim8] = p2
        # y[:, :, self.dim9] = p3
        # y = torch.cat((y, part4),dim=-1)
        # y = y + part4

        y = self.dropout(y)
        return y


class MLPblock(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, target, dropout=0.):
        super(MLPblock, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, target)
        self.dropout = nn.Dropout(dropout)
        # self.gelu = nn.gule()
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        x_ = self.ln(x)
        # x_ = self.dropout(self.w_1(x).relu())
        x_ = self.w_2(self.dropout(self.w_1(x_).relu()))
        return x_


# if __name__ == '__main__':
#     a = torch.ones(16, 10, 96)
#     abc = PatchingDifined()
#     print(abc(a).shape)
#     dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
#                          26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
#                          46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
#                          75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])
#     print(dim_used.reshape(22,3))

class PatchingDifined1(nn.Module):
    def __init__(self):
        super(PatchingDifined1, self).__init__()
        self.dim1 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])  # 右脚 12
        self.dim2 = np.array([12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])  # 左脚 12
        self.dim3 = np.array([24, 25, 26, 27, 28, 29])  # 躯干 6
        self.dim4 = np.array([30, 31, 32, 33, 34, 35])  # naodai
        self.dim5 = np.array([36, 37, 38, 39, 40, 41, 42, 43, 44])  # 右臂

        self.dim6 = np.array([42, 43, 44, 45, 46, 47, 48, 49, 50])  # 右手
        self.dim7 = np.array(
            [51, 52, 53, 54, 55, 56, 57, 58, 59])  # 左臂
        self.dim8 = np.array([57, 58, 59, 60, 61, 62, 63, 64, 65])  # 左手

        dim = 20
        self.fc1 = nn.Linear(12, dim)
        self.fc2 = nn.Linear(12, dim)
        self.fc3 = nn.Linear(6, dim)
        self.fc4 = nn.Linear(6, dim)
        self.fc5 = nn.Linear(9, dim)
        self.fc6 = nn.Linear(9, dim)
        self.fc7 = nn.Linear(9, dim)
        self.fc8 = nn.Linear(9, dim)

    def forward(self, z):
        part1 = z[:, :, self.dim1]
        part2 = z[:, :, self.dim2]
        part3 = z[:, :, self.dim3]
        part4 = z[:, :, self.dim4]
        part5 = z[:, :, self.dim5]
        part6 = z[:, :, self.dim6]
        part7 = z[:, :, self.dim7]
        part8 = z[:, :, self.dim8]

        part1 = self.fc1(part1)
        part2 = self.fc2(part2)
        part3 = self.fc3(part3)
        part4 = self.fc4(part4)
        part5 = self.fc5(part5)
        part6 = self.fc6(part6)
        part7 = self.fc7(part7)
        part8 = self.fc8(part8)

        output = torch.stack([part1, part2, part3, part4, part5, part6, part7, part8], dim=2)
        return output


class PatchingDifined2(nn.Module):
    def __init__(self):
        super(PatchingDifined2, self).__init__()
        self.dim1 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])  # 右脚 12
        self.dim2 = np.array([12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])  # 左脚 12
        self.dim3 = np.array([24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35])  # 躯干 12
        # self.dim4 = np.array([30, 31, 32, 33, 34, 35])  # naodai
        self.dim5 = np.array([36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50])  # 右臂 15

        # self.dim6 = np.array([42, 43, 44, 45, 46, 47, 48, 49, 50])  # 右手
        self.dim6 = np.array(
            [51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65])  # 左臂
        # self.dim8 = np.array([57, 58, 59, 60, 61, 62, 63, 64, 65])  # 左手

        dim = 50
        self.fc1 = nn.Linear(12, dim)
        self.fc2 = nn.Linear(12, dim)
        self.fc3 = nn.Linear(12, dim)
        # self.fc4 = nn.Linear(6, dim)
        self.fc5 = nn.Linear(15, dim)
        self.fc6 = nn.Linear(15, dim)
        # self.fc7 = nn.Linear(9, dim)
        # self.fc8 = nn.Linear(9, dim)
        # self.fc9 = nn.Linear(9, dim)

    def forward(self, z):
        part1 = z[:, :, self.dim1]
        part2 = z[:, :, self.dim2]
        part3 = z[:, :, self.dim3]
        # part4 = z[:, :, self.dim4]
        part5 = z[:, :, self.dim5]
        part6 = z[:, :, self.dim6]
        # part7 = z[:, :, self.dim7]
        # part8 = z[:, :, self.dim8]

        part1 = self.fc1(part1)
        part2 = self.fc2(part2)
        part3 = self.fc3(part3)
        # part4 = self.fc4(part4)
        part5 = self.fc5(part5)
        part6 = self.fc6(part6)
        # part7 = self.fc7(part7)
        # part8 = self.fc8(part8)

        output = torch.stack([part1, part2, part3, part5, part6], dim=2)
        return output


class PatchingDifined4(nn.Module):
    def __init__(self):
        super(PatchingDifined4, self).__init__()
        self.dim1 = np.array([6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23,45,46,47,48,49,50,60,61,62,63,64,65])   #24    #
        self.dim2 = np.array([3, 4, 5, 15, 16, 17, 42, 43, 44, 57, 58, 59])       # 12
        self.dim3 = np.array([0, 1, 2, 12, 13, 14, 39, 40, 41, 54, 55, 56])          # 12

        self.dim4 = np.array([24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 51, 52, 53])
        self.dim5 = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 42, 43, 44, 45, 46, 47, 48, 49, 50])
        self.dim6 = np.array([15, 16, 17, 18, 19, 20, 21, 22, 23, 57, 58, 59, 60, 61, 62, 63, 64, 65])

        self.dim7 = np.array(
            [3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17, 18, 19, 20, 21, 22, 23, 42, 43, 44, 45, 46, 47, 48, 49, 50, 57,
             58, 59, 60, 61, 62, 63, 64, 65])  # 36
        self.dim8 = np.array(
            [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 36, 37, 38, 39, 40, 41, 51, 52, 53, 54, 55, 56])  # 24
        self.dim9 = np.array([24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 51, 52, 53])  # 18 qugan

        # self.dim10 = np.array([24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35])  # 12
        # self.dim11 = np.array(
        #     [0, 1, 2, 12, 13, 14, 36, 37, 38, 39, 40, 41, 51, 52, 53, 54, 55, 56])   #18
        dim = 50
        self.fc1 = nn.Linear(24, dim)
        self.fc2 = nn.Linear(12, dim)
        self.fc3 = nn.Linear(12, dim)
        self.fc9 = nn.Linear(18, dim)

    def forward(self, z):
        part1 = z[:, :, self.dim1]
        part2 = z[:, :, self.dim2]
        part3 = z[:, :, self.dim3]
        part9 = z[:, :, self.dim9]

        part1 = self.fc1(part1)
        part2 = self.fc2(part2)
        part3 = self.fc3(part3)
        part9 = self.fc9(part9)

        output = torch.stack([part1, part2, part3, part9], dim=2)
        return output

class PatchingDifined5(nn.Module):
    def __init__(self):
        super(PatchingDifined5, self).__init__()
        self.dim1 = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17, 18, 19, 20, 21, 22, 23])
        self.dim2 = np.array([42, 43, 44, 45, 46, 47, 48, 49, 50, 57, 58, 59, 60, 61, 62, 63, 64, 65])
        self.dim3 = np.array([0, 1, 2, 12, 13, 14, 36, 37, 38, 39, 40, 41, 51, 52, 53, 54, 55, 56])

        self.dim4 = np.array([24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 51, 52, 53])
        self.dim5 = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 42, 43, 44, 45, 46, 47, 48, 49, 50])
        self.dim6 = np.array([15, 16, 17, 18, 19, 20, 21, 22, 23, 57, 58, 59, 60, 61, 62, 63, 64, 65])

        self.dim7 = np.array(
            [3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17, 18, 19, 20, 21, 22, 23, 42, 43, 44, 45, 46, 47, 48, 49, 50, 57,
             58, 59, 60, 61, 62, 63, 64, 65])  # 36
        self.dim8 = np.array(
            [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 36, 37, 38, 39, 40, 41, 51, 52, 53, 54, 55, 56])  # 24
        self.dim9 = np.array([24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 51, 52, 53])  # 18

        # self.dim10 = np.array([24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35])  # 12
        # self.dim11 = np.array(
        #     [0, 1, 2, 12, 13, 14, 36, 37, 38, 39, 40, 41, 51, 52, 53, 54, 55, 56])   #18
        dim = 50
        self.fc7 = nn.Linear(36, dim)
        self.fc8 = nn.Linear(24, dim)
        self.fc9 = nn.Linear(18, dim)

    def forward(self, z):
        part1 = z[:, :, self.dim7]
        part2 = z[:, :, self.dim8]
        part3 = z[:, :, self.dim9]

        part1 = self.fc7(part1)
        part2 = self.fc8(part2)
        part3 = self.fc9(part3)

        output = torch.stack([part1, part2, part3], dim=2)
        return output
