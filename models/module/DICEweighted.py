import torch
import numpy as np
from scipy.ndimage import convolve

def dice_val_local_weighted(y_pred, y_true, VOI_lbls=[2], window_size=5):
    """
    基于局部窗口计算每个像素的DICE，并生成权重矩阵。
    
    y_pred: 预测的分割图像，尺寸为 (1, 1, D, H, W)
    y_true: 真实标签图像，尺寸为 (1, 1, D, H, W)
    VOI_lbls: 感兴趣的标签列表
    window_size: 局部窗口大小，必须为奇数
    """
    # 确保窗口大小为奇数
    assert window_size % 2 == 1, "Window size must be odd"
    
    # 获取预测和真实标签的Numpy数组表示
    pred = y_pred.detach().cpu().numpy()[0, 0, ...]
    true = y_true.detach().cpu().numpy()[0, 0, ...]
    
    # 初始化权重矩阵
    weight_matrix = np.zeros_like(pred, dtype=np.float32)
    
    # 定义卷积核，用于计算局部求和
    kernel = np.ones((window_size, window_size, window_size))
    
    for label in VOI_lbls:
        # 获取当前标签的二值图像
        pred_label = (pred == label).astype(np.float32)
        true_label = (true == label).astype(np.float32)
        
        # 计算局部区域内的交集和并集
        intersection = convolve(pred_label * true_label, kernel, mode='constant', cval=0.0)
        sum_pred = convolve(pred_label, kernel, mode='constant', cval=0.0)
        sum_true = convolve(true_label, kernel, mode='constant', cval=0.0)
        union = sum_pred + sum_true
        
        # 计算局部DICE系数
        local_dice = (2 * intersection) / (union + 1e-5)
        
        # 将局部DICE系数赋予对应位置的权重
        weight_matrix += local_dice  # 如果多个标签重叠，可以累加或取最大值
    
    # 归一化权重矩阵到 [0, 1]
    weight_matrix = weight_matrix / len(VOI_lbls)
    
    # 计算加权的DICE loss
    intersection = np.sum(weight_matrix * (pred == true))
    union = np.sum(weight_matrix * ((pred != 0) + (true != 0)))
    weighted_dice_loss = 1 - (2. * intersection / (union + 1e-5))
    
    return weighted_dice_loss


if __name__ == "__main__":
    # 测试
    y_pred = torch.randint(0, 3, (1, 1, 64, 64, 64))  # 假设有3个类别的预测图像
    y_true = torch.randint(0, 3, (1, 1, 64, 64, 64))  # 假设有3个类别的真实标签
    weighted_dice_loss = dice_val_local_weighted(y_pred, y_true, VOI_lbls=[1, 2], window_size=5)
    print("Weighted DICE Loss:", weighted_dice_loss)


# import torch
# import numpy as np
# from scipy.ndimage import binary_erosion

# def dice_val_pixel_weighted(y_pred, y_true, VOI_lbls=[2], window_size=3):
#     """
#     计算每个像素的局部区域DICE并基于此生成权重矩阵。
    
#     y_pred: 预测的分割图像，尺寸为 (1, 1, 224, 224, 192)
#     y_true: 真实标签图像，尺寸为 (1, 1, 224, 224, 192)
#     VOI_lbls: 感兴趣的标签列表
#     window_size: 用于局部DICE计算的小窗口大小
#     """
#     # 获取预测和真实标签的Numpy数组表示
#     pred = y_pred.detach().cpu().numpy()[0, 0, ...]
#     true = y_true.detach().cpu().numpy()[0, 0, ...]

#     # 创建一个与输入相同大小的权重矩阵
#     weight_matrix = np.ones_like(pred, dtype=np.float32)
    
#     half_window = window_size // 2

#     # 遍历每个VOI标签
#     for i in VOI_lbls:
#         pred_i = pred == i  # 预测为标签i的区域
#         true_i = true == i  # 真实标签为i的区域

#         # 对每个像素的局部区域计算DICE系数
#         for z in range(half_window, pred.shape[2] - half_window):
#             for y in range(half_window, pred.shape[1] - half_window):
#                 for x in range(half_window, pred.shape[0] - half_window):
#                     # 提取局部窗口区域
#                     pred_window = pred_i[x-half_window:x+half_window+1, 
#                                          y-half_window:y+half_window+1, 
#                                          z-half_window:z+half_window+1]
#                     true_window = true_i[x-half_window:x+half_window+1, 
#                                          y-half_window:y+half_window+1, 
#                                          z-half_window:z+half_window+1]
                    
#                     # 计算局部区域的DICE系数
#                     intersection = np.sum(pred_window * true_window)
#                     union = np.sum(pred_window) + np.sum(true_window)
#                     pixel_dice = (2. * intersection) / (union + 1e-5)

#                     # 将局部的DICE值赋给中心像素的权重
#                     weight_matrix[x, y, z] = pixel_dice

#     # 计算整体DICE系数
#     intersection_total = np.sum(weight_matrix * (pred == true))  # 逐像素乘以权重
#     union_total = np.sum(weight_matrix * ((pred != 0) + (true != 0)))  # 并集加权

#     # 返回加权的DICE loss
#     weighted_dice_loss = 1 - (2. * intersection_total / (union_total + 1e-5))
    
#     return weighted_dice_loss

# # 测试
# y_pred = torch.rand(1, 1, 224, 224, 192)  # 假设为随机生成的预测图像
# y_true = torch.rand(1, 1, 224, 224, 192)  # 假设为随机生成的真实标签
# y_pred = y_pred > 0.5  # 二值化
# y_true = y_true > 0.5  # 二值化
# y_pred = y_pred.long()
# y_true = y_true.long()

# weighted_dice_loss = dice_val_pixel_weighted(y_pred, y_true)
# print("Weighted DICE Loss:", weighted_dice_loss)
