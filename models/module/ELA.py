import torch
import torch.nn as nn
 
 
class ELA(nn.Module):
    def __init__(self, in_channels, phi):
        super(ELA, self).__init__()
        '''
        ELA-T 和 ELA-B 设计为轻量级，非常适合网络层数较少或轻量级网络的 CNN 架构
        ELA-B 和 ELA-S 在具有更深结构的网络上表现最佳
        ELA-L 特别适合大型网络。
        '''
        Kernel_size = {'T': 5, 'B': 7, 'S': 5, 'L': 7}[phi]
        groups = {'T': in_channels, 'B': in_channels, 'S': in_channels//8, 'L': in_channels//8}[phi]
        num_groups = {'T': 32, 'B': 16, 'S': 16, 'L': 16}[phi]
        pad = Kernel_size//2
        self.con1 = nn.Conv1d(in_channels, in_channels, kernel_size=Kernel_size, padding=pad, groups=groups, bias=False)
        self.GN = nn.GroupNorm(num_groups, in_channels)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, input):
        b, c, h, w = input.size()
        x_h = torch.mean(input, dim=3, keepdim=True).view(b,c,h)
        x_w = torch.mean(input, dim=2, keepdim=True).view(b,c,w)
        x_h = self.con1(x_h)    # [b,c,h]
        x_w = self.con1(x_w)    # [b,c,w]
        x_h = self.sigmoid(self.GN(x_h)).view(b, c, h, 1)   # [b, c, h, 1]
        x_w = self.sigmoid(self.GN(x_w)).view(b, c, 1, w)   # [b, c, 1, w]
        return x_h * x_w * input

class ELA_3D(nn.Module):
    def __init__(self, in_channels, phi='B'):
        super(ELA_3D, self).__init__()
        '''
        ELA-T 和 ELA-B 设计为轻量级，非常适合网络层数较少或轻量级网络的 CNN 架构
        ELA-B 和 ELA-S 在具有更深结构的网络上表现最佳
        ELA-L 特别适合大型网络。
        '''
        Kernel_size = {'T': 5, 'B': 7, 'S': 5, 'L': 7}[phi]
        groups = {'T': in_channels, 'B': in_channels, 'S': in_channels//8, 'L': in_channels//8}[phi]
        num_groups = {'T': 32, 'B': 8, 'S': 16, 'L': 16}[phi]#'B': 16 'S': 16, 'L': 16
        pad = Kernel_size // 2

        # 1D卷积改为针对3D的1D卷积，卷积在通道维度上运算
        self.con1 = nn.Conv1d(in_channels, in_channels, kernel_size=Kernel_size, padding=pad, groups=groups, bias=False)
        self.GN = nn.GroupNorm(num_groups, in_channels)  # 归一化层保持不变
        self.sigmoid = nn.Sigmoid()  # Sigmoid 激活函数保持不变

    def forward(self, input):
        # input 的形状是 [b, c, d, h, w]
        b, c, d, h, w = input.size()

        # 沿宽度（w）计算平均值，生成 [b, c, d, h] 张量
        x_h = torch.mean(input, dim=4, keepdim=True)  # [b, c, d, h, 1]
        x_h = x_h.view(b, c, d * h)  # 调整形状为 [b, c, d * h]

        # 沿高度（h）计算平均值，生成 [b, c, d, w] 张量
        x_w = torch.mean(input, dim=3, keepdim=True)  # [b, c, d, 1, w]
        x_w = x_w.view(b, c, d * w)  # 调整形状为 [b, c, d * w]

        # 沿深度（d）计算平均值，生成 [b, c, h, w] 张量
        x_d = torch.mean(input, dim=2, keepdim=True)  # [b, c, 1, h, w]
        x_d = x_d.view(b, c, h * w)  # 调整形状为 [b, c, h * w]

        # 分别对深度、宽度和高度方向的特征进行 1D 卷积
        x_h = self.con1(x_h)    # [b, c, d * h]，沿高度方向
        x_w = self.con1(x_w)    # [b, c, d * w]，沿宽度方向
        x_d = self.con1(x_d)    # [b, c, h * w]，沿深度方向

        # 归一化并通过 Sigmoid 函数生成注意力权重
        x_h = self.sigmoid(self.GN(x_h)).view(b, c, d, h, 1)   # [b, c, d, h, 1]
        x_w = self.sigmoid(self.GN(x_w)).view(b, c, d, 1, w)   # [b, c, d, 1, w]
        x_d = self.sigmoid(self.GN(x_d)).view(b, c, 1, h, w)   # [b, c, 1, h, w]

        # 将深度、宽度和高度方向的注意力权重分别应用到原始输入特征图上
        return x_h * x_w * x_d * input  # 最终注意力应用到原始输入特征图


 
if __name__ == "__main__":
 
    # 创建一个形状为 [batch_size, channels, height, width] 的虚拟输入张量
    input = torch.randn(2, 256, 40, 40, 50)
    ela = ELA_3D(in_channels=256, phi='T')
    output = ela(input)
    print(output.size())