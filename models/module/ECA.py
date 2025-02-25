import torch
from torch import nn
from torch.nn.parameter import Parameter

class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()
 
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
 
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
 
        # Multi-scale information fusion
        y = self.sigmoid(y)
 
        return x * y.expand_as(x)
    
class eca_layer_3d(nn.Module):
    """Constructs a 3D ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer_3d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)  # 对 3D 特征图进行全局平均池化
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, d, h, w]
        b, c, d, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)  # 对输入特征图进行全局空间平均池化, 形状变为 [b, c, 1, 1, 1]
        # y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Two different branches of ECA module
        y = y.view(b, c, -1).transpose(-1, -2)  # 将 [b, c, 1, 1, 1] reshape 为 [b, c, 1]->[b, 1, c]
        y = self.conv(y).view(b, c, 1, 1, 1)  # 在通道维度上进行 1D 卷积，再恢复形状

        # Multi-scale information fusion
        y = self.sigmoid(y)  # 使用 Sigmoid 激活

        return x * y.expand_as(x)  # 将通道权重应用到输入特征图上

if __name__ == "__main__":

    # 创建一个形状为 [batch_size, channels, height, width] 的虚拟输入张量
    input = torch.randn(2, 256, 40, 40, 50)
    ela = eca_layer_3d(channel=256)
    output = ela(input)
    print(output.size())