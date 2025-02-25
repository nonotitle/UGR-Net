import os
import nibabel as nib
import numpy as np

# 定义文件夹路径
root_dir = '/mnt/data1/wangnaying/dataset/CBCT_NII_06/train/'
tooth_dir = os.path.join(root_dir, 'expandtooth20')
allseg_dir = os.path.join(root_dir, 'segnotooth20_visualization')

# 确保allseg文件夹存在
os.makedirs(allseg_dir, exist_ok=True)

# 遍历文件夹，查找包含"_segout_"的nii.gz文件
for root, dirs, files in os.walk(root_dir):
    for file in files:
        if '_segout_' in file and file.endswith('.nii.gz'):
            seg_file_path = os.path.join(root, file)

            # 生成对应的牙齿文件名
            base_name = file.split('_segout_')[0]
            tooth_file_name = f'{base_name}_expandtooth20_{file.split("_segout_")[1]}'
            tooth_file_path = os.path.join(tooth_dir, tooth_file_name)

            # 检查牙齿文件是否存在
            if os.path.exists(tooth_file_path):
                # 读取牙齿文件和分割文件
                teeth_img = nib.load(tooth_file_path)
                seg_img = nib.load(seg_file_path)

                teeth_data = teeth_img.get_fdata()
                seg_data = seg_img.get_fdata()

                # 将牙齿分割文件非0的区域设为6
                teeth_data[teeth_data != 0] = 6

                # 将分割文件的同样区域设为6
                seg_data[(teeth_data == 6) & (seg_data == 1)] = 0
                seg_data[(teeth_data == 6) & (seg_data == 4)] = 0

                seg_data = seg_data.round()

                # 使用原始分割文件的affine和头文件
                affine = seg_img.affine
                header = seg_img.header

                # 创建新的Nifti1Image对象
                new_seg_img = nib.Nifti1Image(seg_data, affine)
                # 不使用头文件，否则在itksnap中标签显示错误，在slicer中可以看到是产生了小数

                # 重命名并保存文件
                new_file_name = f'{base_name}_segnotooth20_{file.split("_segout_")[1]}'
                new_file_path = os.path.join(allseg_dir, new_file_name)
                nib.save(new_seg_img, new_file_path)

                print(f'处理完成: {new_file_path}')
            else:
                print(f'对应的牙齿文件不存在: {tooth_file_path}')

