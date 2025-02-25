from torch.utils.tensorboard import SummaryWriter
import os, utils, glob, losses, random, math
import sys
from torch.utils.data import DataLoader
from data import datasets_cbct, trans
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn.functional as F
from natsort import natsorted
# from models.TransMorph_affine import CONFIGS as CONFIGS_TM
import models.NICEnet_affine14deform_weightnox as NICEnet
# import models.NICEnetdeform1aff4 as NICEnet
from losses import NCC, GCC, GlobalNCC, MaskedGCC, LGCC_tooth, Grad3d
import torch.nn as nn
import json
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir + "logfile.log", "a+")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass




def load_patient_data(data_dir):
    # 创建一个空字典来存储每个病人的数据
    patient_data = {}
    for file_path in natsorted(glob.glob(data_dir+'/**', recursive=True)):
        if file_path.endswith('.nii.gz'):
            # 获取文件名（包括路径）
            file_name = os.path.basename(file_path)
            # 提取病人 ID
            patient_id = file_name.split('_')[0].split('_')[0]
            # 如果该病人 ID 已经在字典中，将数据添加到该病人的列表中
            if patient_id in patient_data:
                pass
            # 如果该病人 ID 不在字典中，创建一个新的字典，并将数据添加到字典中
            else:
                patient_data[patient_id] = {'image':[],'segout':[],'segnotooth':[],'tooth':[],}

            if len(file_name.split('_'))<=2:
                patient_data[patient_id]['image'].append(file_path)
                # patient_data[patient_id][file_name.split('_')[1]].append(file_path)
            else:
                if '_allseg_' in file_name:
                    patient_data[patient_id]['segnotooth'].append(file_path)
                elif '_expandtooth_' in file_name:
                    patient_data[patient_id]['tooth'].append(file_path)
                elif '_segout_' in file_name:
                    patient_data[patient_id]['segout'].append(file_path)
    return patient_data
def load_patient_data_test(data_dir):
    # 创建一个空字典来存储每个病人的数据
    patient_data = {}
    for file_path in natsorted(glob.glob(data_dir+'/**', recursive=True)):
        if file_path.endswith('.nii.gz'):
            # 获取文件名（包括路径）
            file_name = os.path.basename(file_path)
            # 提取病人 ID
            patient_id = file_name.split('_')[0].split('_')[0]
            # 如果该病人 ID 已经在字典中，将数据添加到该病人的列表中
            if patient_id in patient_data:
                pass
            # 如果该病人 ID 不在字典中，创建一个新的字典，并将数据添加到字典中
            else:
                patient_data[patient_id] = {'image':[],'segout':[],'segnotooth':[],'tooth':[],}

            if len(file_name.split('_'))<=2:
                patient_data[patient_id]['image'].append(file_path)
                # patient_data[patient_id][file_name.split('_')[1]].append(file_path)
            else:
                if '_segnotooth_' in file_name:
                    patient_data[patient_id]['segnotooth'].append(file_path)
                elif '_expandtooth_' in file_name:
                    patient_data[patient_id]['tooth'].append(file_path)
                elif '_segout_' in file_name:
                    patient_data[patient_id]['segout'].append(file_path)
    return patient_data


def mask_images_by_type(x_moved, y_fixed, x_seg_moved, y_seg_fixed, target_label=[2], type = 'separate'):
    # 创建一个与 x_moved, y_fixed 相同大小的零张量作为新的 x_, y_
    x_ = torch.zeros_like(x_moved)
    y_ = torch.zeros_like(y_fixed)

    if type == 'or':
    # 获取目标标签的位置索引
        target_mask = (x_seg_moved == target_label) | (y_seg_fixed == target_label)
        # 将 x_moved, y_fixed 中目标标签对应位置的像素值复制到 x_, y_
        x_[target_mask] = x_moved[target_mask]
        y_[target_mask] = y_fixed[target_mask]
    elif type == 'and':
        target_mask = (x_seg_moved == target_label) & (y_seg_fixed == target_label)
        x_[target_mask] = x_moved[target_mask]
        y_[target_mask] = y_fixed[target_mask]
    elif type == 'separate':
        # 将列表转换为张量
        target_label_tensor = torch.tensor(target_label).cuda()

        # 对 x 的操作
        target_mask = torch.isin(x_seg_moved, target_label_tensor).cuda()
        x_[target_mask] = x_moved[target_mask]

        # 对 y 的操作
        target_mask = torch.isin(y_seg_fixed, target_label_tensor).cuda()
        y_[target_mask] = y_fixed[target_mask]
    return x_, y_

def crop_image_by_label(seg_image, label, threshold):
    # 获取分割图像的形状
    _, _, _, y, _ = seg_image.shape
    
    # 找到指定label在y轴上的最小值和最大值
    label_y_min = None
    label_y_max = None
    for i in range(y):
        if torch.any(seg_image[0, 0, :, i, :] == label):
            label_y_min = i
            break
    for i in range(y-1, -1, -1):
        if torch.any(seg_image[0, 0, :, i, :] == label):
            label_y_max = i
            break
    
    # 根据指定的阈值调整裁剪范围
    crop_y_min = label_y_min
    crop_y_max = label_y_min + threshold
    
    # 将除了指定label范围之外的部分设为0
    seg_image[:, :, :, :crop_y_min, :] = 0
    seg_image[:, :, :, crop_y_max:, :] = 0
    
    return seg_image

def mask_images_noteeth(x_moved, y_fixed, x_seg_moved, y_seg_fixed, type = 'separate'):
    # 创建一个与 x_moved, y_fixed 相同大小的零张量作为新的 x_, y_
    x_ = torch.zeros_like(x_moved)
    y_ = torch.zeros_like(y_fixed)

    # 获取目标标签的位置索引
    target_mask = (x_seg_moved == 0)
    # 将 x_moved, y_fixed 中目标标签对应位置的像素值复制到 x_, y_
    x_[target_mask] = x_moved[target_mask]
    target_mask = (y_seg_fixed == 0)
    y_[target_mask] = y_fixed[target_mask]

    return x_, y_

def make_dsc_dict(dsc_dict, patient, dsc):
    key = str(int(patient))
    if key in dsc_dict:
        dsc_dict[key].append(dsc)
    else:
        dsc_dict[key] = [dsc]
def combined_loss(epoch, total_epochs, mode='linear'):
    """
    根据训练epoch动态调整DSC和加权DSC的权重
    """
    # 基于当前epoch调整权重，开始时dsc权重高，之后dsc_weighted权重高
    if mode == 'linear':
        alpha = max(0, min(1, epoch / total_epochs))  # 线性变化
    elif mode == 'cosine':
        alpha = 0.5 * (1 - np.cos(np.pi * epoch / total_epochs))  # 余弦调整，平滑过渡
    else:
        raise ValueError("Unknown mode: Choose 'linear' or 'cosine'")
    return alpha

def main():
    batch_size = 1
    data_dir = '../../../dataset/CBCT_NII_06/'
    lr = 0.00005  # learning rate
    epoch_start = 0
    max_epoch = 400  #max traning epoch
    checkpoint = 200
    cont_training = False  #if continue training
    type_cran = 'separate'
    # mask_label = [1]  # 用于对原图进行mask
    dsc_label = [1]  # 用于计算dsc的label
    model_firstname = 'nicea1CTFUG5e-5_localgcctooth+semil1rawtestnotoo+0.1grad_norm02k5_mylinearloss_mat_eladecosgirdwin59_attmulty192^3_nox_trainlinadjust'
    save_dir = 'save/'+ model_firstname + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(save_dir+'image/')
    sys.stdout = Logger(save_dir)

    # dict patient 
    train_patient = load_patient_data(os.path.join(data_dir, 'train/'))
    val_patient = load_patient_data_test(os.path.join(data_dir, 'test/'))
    '''
    Initialize model
    '''
    H, W, D = 224, 224, 192

    model = NICEnet.CTFPW_DEF_Trans(window_size=[5,9])
    model.cuda()
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {count_parameters(model)}")
    AffineTransformer_seg = NICEnet.AffineTransformer_block(mode='nearest')
    AffineTransformer_seg.cuda()
    AffineTransformer = NICEnet.AffineTransformer_block('bilinear')
    AffineTransformer.cuda()
    SpatialTransformer_seg = NICEnet.SpatialTransformer_block(mode='nearest')
    SpatialTransformer_seg.cuda()
    # affine_trans = TransMorph.AffineTransform()#AffineTransformer((H, W, D)).cuda()
    
    '''
    Initialize spatial transformation function
    '''
    # reg_model = utils.register_model(config.img_size, 'nearest')
    # reg_model.cuda()
    # reg_model_bilin = utils.register_model(config.img_size, 'bilinear')
    # reg_model_bilin.cuda()
    '''
    If continue from previous training
    '''
    if cont_training:
        epoch_start = 211
        # model_dir = 'experiments/' + save_dir
        model_dir = 'ave/nice3affdefCTFPW5e-5_localgcctooth+semil4+0.1grad_norm02k5_mylinearloss_mat_eladecosdirect_xy_attmulty192^3_nox_trainlin/'
        updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch, 0.9), 8)
        best_model = torch.load(
            model_dir + natsorted(os.listdir(model_dir))[-1])#['state_dict']
        print('Model: {} loaded!'.format(natsorted(os.listdir(model_dir))[-1]))
        model.load_state_dict(best_model)
    else:
        updated_lr = lr
    '''
    Initialize training
    '''
    # train_composed = transforms.Compose([trans.RandomFlip(0),
    #                                      trans.NumpyType((np.float32, np.float32))])
    train_composed  = None
    # 去掉了对label的预处理
    # val_composed = transforms.Compose([trans.Seg_norm(), #rearrange segmentation label to 1 to 46
    #                                    trans.NumpyType((np.float32, np.int16))])
    # 是否norm？
    train_set = datasets_cbct.InferDatasetCBCTtooth(train_patient, transforms=train_composed, norm=True,labels=dsc_label)
    val_set = datasets_cbct.InferDatasetCBCTtooth(val_patient, norm=True,labels=dsc_label)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    # criterion = nn.MSELoss()
    # criterion = NCC()
    # criterion = NCCex0()
    criterion = GlobalNCC()
    criterionlgcc = LGCC_tooth()
    # criterion = GCC()
    # criterion = MaskedGCC()
    criteriongrad = Grad3d()
    dscloss = utils.MulticlassDiceLossVectorize()
    dscloss_mat = utils.MulticlassDiceLossVectorize_mat()
    train_dsc_dict = {}
    eval_dsc_dict = {}
    max_val_dsc = 0
    writer = SummaryWriter(log_dir='logss/'+model_firstname)
    for epoch in range(epoch_start, max_epoch+1):
        # print(torch.cuda.memory_summary())
        adjust_learning_rate(optimizer=optimizer, epoch=epoch, MAX_EPOCHES= max_epoch*2, INIT_LR=updated_lr)
        alpha = combined_loss(epoch, max_epoch, mode='linear')
        # alpha = 1
        print('epoch {} lr {} Training Starts'.format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
        '''
        Training
        '''
        idx = 0
        total_ncc_loss=0
        total_dsc = 0
        total_dsc_weight = 0
        # train_dsc = utils.AverageMeter()
        for data in train_loader:
            # break
            idx += 1
            model.train()
            data = [t.cuda() for t in data]
            x_moved = data[0]
            y_fixed = data[1]  # 设置 x_0为moved，y_1为fixed(atlas)
            x_seg_moved = data[2]
            y_seg_fixed = data[3]
            x_seg_segout = data[4]
            y_seg_segout = data[5]
            x_seg_tooth = data[6]
            # patient_id = data[8]
            # x_moved, x_seg_moved = affine_aug(x_moved, seed=idx, im_label=x_seg_moved) # 是否进行仿射变换增强

            x_, y_ = x_moved, y_fixed
            # affined, current_affine_matrix, affine_para, affine_para_list,affine_matrix_list,warped, flow_last, aff_flow, weight_matrix = model(y_, x_)
            affined, current_affine_matrix,affine_matrix_list,warped, flow_last, aff_flow, weight_matrix = model(y_, x_)

            # x_seg_trans = AffineTransformer_seg(x_seg_moved, affine_para)
            x_seg_trans = AffineTransformer(x_seg_moved, current_affine_matrix)

            dsc = dscloss(x_seg_trans, y_seg_fixed)
            # dsc_weighted = utils.dice_val_local_weighted(x_seg_trans, y_seg_fixed, VOI_lbls=dsc_label)
            dsc_weighted = dscloss_mat(x_seg_trans, y_seg_fixed,weight_matrix)

            with torch.no_grad():
            # 不需要梯度的操作
                x_seg_transflow = SpatialTransformer_seg(x_seg_moved, flow_last)
                dscflow = utils.dice_val_VOI_onehot(x_seg_transflow, y_seg_fixed, VOI_lbls=len(dsc_label))


            loss = criterion(warped, y_)

            # 计算除了牙齿之外的区域的warped, affined的ncc,使得尽量保持一致
            x_seg_tooth_trans = AffineTransformer_seg(x_seg_tooth, current_affine_matrix)
            loss_affdefl1 = criterionlgcc(warped, affined, x_seg_tooth_trans)
            # loss_affdefl1=0

            loss_grad = criteriongrad(flow_last,y_fixed)
            alphadsc = - ((1 - alpha) * dsc + alpha * dsc_weighted)
            loss_all = loss + alphadsc + loss_affdefl1 + 0.1*loss_grad
            # del warped, flow_last, affined, affine_para, aff_flow_5_up1, x_seg_trans, x_seg_transflow, x_seg_tooth_trans,x_seg_tooth,y_seg_fixed
            del aff_flow,affine_matrix_list,current_affine_matrix,data,x_seg_transflow
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()
            # break

            print('Epoch - {} Iter {} of {} dsc_aff {:.4f} dsc_affweight {:.4f}|||ncc {:.4f} grad:{:.4f} dsc_flow:{:.4f} loss_all:{:.4f}'.format(epoch, idx, len(train_loader),
                                                     dsc.item(), dsc_weighted.item(), loss.item(), loss_grad.item(), dscflow.item(), loss_all.item()))
            total_ncc_loss += loss.item()
            total_dsc += dsc.item()
            total_dsc_weight +=dsc_weighted.item()
        # writer.add_scalars('loss', {'total_loss': total_loss}, epoch)
        writer.add_scalar('train_avg_dsc', total_dsc / idx, epoch+1)
        writer.add_scalar('train_avg_dsc_weight', total_dsc_weight / idx, epoch+1)
        writer.add_scalar('ave_loss', total_ncc_loss / idx, epoch+1)
        writer.add_scalar('lr',  optimizer.state_dict()['param_groups'][0]['lr'], epoch+1)
        print('----Tra----Epoch - {} train_avg_dsc {:.4f} train_avg_dsc_weight {:.4f} '.format(epoch, total_dsc / idx, total_dsc_weight / idx))
        if epoch % checkpoint == 0:
            modelname = model_firstname + str(epoch) + '.pth'
            save_checkpoint(model.state_dict(), save_dir=save_dir, filename=modelname)
        '''
        Validation
        '''
        idd = 0
           
        eval_dsc = utils.AverageMeter()
        eval_dsc_weight = utils.AverageMeter()
        eval_dsc_flow_all = utils.AverageMeter()
        eval_dsc_flow = utils.AverageMeter()
        eval_asd = utils.AverageMeter()
        eval_hd95 = utils.AverageMeter()
        eval_asd_flow = utils.AverageMeter()
        eval_hd95_flow = utils.AverageMeter()
        with torch.no_grad():
            for data in val_loader:
                model.eval()
                data = [t.cuda() for t in data]
                x_moved = data[0]
                y_fixed = data[1]  # 设置 x_0为moved，y_1为fixed(atlas)
                x_seg_moved = data[2]
                y_seg_fixed = data[3]
                x_seg_segout = data[4]
                y_seg_segout = data[5]
                # patient_id = data[8]

                # x_moved, x_seg_moved = affine_aug(x_moved, seed=idx, im_label=x_seg_moved) # 是否进行仿射变换增强

                x_, y_ = x_moved, y_fixed
                affined, current_affine_matrix, affine_matrix_list,warped, flow_last, aff_flow, weight_matrix = model(y_, x_)
                # 可变形扭曲的图像计算nccloss, affine变换的分割图计算dsc_loss，半监督
                x_seg_trans = AffineTransformer_seg(x_seg_moved, current_affine_matrix)
                # y_trans = AffineTransformer(y_, affine_para)               

                dsc = dscloss(x_seg_trans, y_seg_fixed)
                dsc_weighted = dscloss_mat(x_seg_trans, y_seg_fixed, weight_matrix=weight_matrix)
                eval_dsc.update(dsc.item(), x_moved.size(0))
                eval_dsc_weight.update(dsc_weighted.item(), x_moved.size(0))
                # print(eval_dsc.avg)
                x_seg_transflow = SpatialTransformer_seg(x_seg_moved, flow_last)
                dsc_flow = utils.dice_val_VOI_onehot(x_seg_transflow, y_seg_fixed, VOI_lbls=len(dsc_label))
                eval_dsc_flow.update(dsc_flow.item(),  x_moved.size(0))

                x_seg_movedall=x_seg_segout
                y_seg_fixedall=y_seg_segout
                x_seg_transflowall = SpatialTransformer_seg(x_seg_movedall, flow_last)
                dsc_flowall = utils.dice_val_VOI_onehot(x_seg_transflowall, y_seg_fixedall, VOI_lbls=len([1,2,4]))
                eval_dsc_flow_all.update(dsc_flowall.item(), x_moved.size(0))
                
                if epoch % checkpoint == 0:
                    plt.figure()
                    plt.subplot(2, 2, 1)
                    plt.imshow(x_.cpu().detach().numpy()[0, 0, :, 99,:], cmap='gray')
                    plt.subplot(2, 2, 2)
                    plt.imshow(y_.cpu().detach().numpy()[0, 0, :, 99,:], cmap='gray')
                    plt.subplot(2, 2, 4)
                    plt.imshow(warped.cpu().detach().numpy()[0, 0, :, 99,:], cmap='gray')
                    plt.subplot(2, 2, 3)
                    plt.imshow(affined.cpu().detach().numpy()[0, 0, :, 99,:], cmap='gray')
                    plt.savefig(save_dir+'/image/reg_results{}-{}'.format(epoch, idd))
                    plt.close()
                
                idd += 1
                if epoch>210:
                    asd = utils.asd_onehot(x_seg_trans, y_seg_fixed, VOI_lbls=len(dsc_label))
                    eval_asd.update(asd.item(), x_moved.size(0))
                    hd95 = utils.hd95_onehot(x_seg_trans, y_seg_fixed, VOI_lbls=len(dsc_label))
                    eval_hd95.update(hd95.item(), x_moved.size(0))

                    asd_f = utils.asd_onehot(x_seg_transflow, y_seg_fixed, VOI_lbls=len(dsc_label))
                    eval_asd_flow.update(asd_f.item(), x_moved.size(0))
                    hd95_f = utils.hd95_onehot(x_seg_transflow, y_seg_fixed, VOI_lbls=len(dsc_label))
                    eval_hd95_flow.update(hd95_f.item(), x_moved.size(0))

                    
            if epoch>210:
                print('----Val----Epoch - {} val_avg_asd {:.4f} val_hd95 {:.4f} val_avg_asd_flow {:.4f} val_hd95_flow {:.4f}'.format(epoch, eval_asd.avg, eval_hd95.avg, eval_asd_flow.avg, eval_hd95_flow.avg))
            print('----Val----Epoch - {} val_avg_dsc {:.4f} val_avg_dscweight {:.4f} dsc_flow {:.4f} dsc_flowall {:.4f}'.format(epoch, eval_dsc.avg, eval_dsc_weight.avg, eval_dsc_flow.avg,eval_dsc_flow_all.avg))
            print('val_dsclist: {}'.format(eval_dsc.vals))
            print('val_dscweightlist: {}'.format(eval_dsc_weight.vals))
            print('val_dsc_flowlist: {}'.format(eval_dsc_flow.vals))
            print('last_mat: {}'.format(current_affine_matrix))
            writer.add_scalar('val_avg_dsc', eval_dsc.avg, epoch+1)
            writer.add_scalar('val_avg_dscweight', eval_dsc_weight.avg, epoch+1)
            writer.add_scalar('val_avg_dscflow', eval_dsc_flow.avg, epoch+1)
            writer.add_scalar('val_avg_dscflowall', eval_dsc_flow_all.avg, epoch+1)
            if max_val_dsc < eval_dsc.avg :
                max_val_dsc = eval_dsc.avg

                pattern = os.path.join(save_dir, 'best' + model_firstname + '*.pth')
                old_models = glob.glob(pattern)
                # 删除所有找到的旧模型文件
                for old_model in old_models:
                    try:
                        os.remove(old_model)
                        print(f"Deleted old model: {old_model}")
                    except Exception as e:
                        print(f"Failed to delete {old_model}: {e}")
                modelname = 'best'+model_firstname + str(epoch) + str(round(eval_dsc.avg, 4)) + '.pth'
                save_checkpoint(model.state_dict(), save_dir=save_dir, filename=modelname)
                print('save best_val_model to: {}'.format(save_dir+modelname))
            print('now max_val_dsc: {}'.format(max_val_dsc))
            

    # with open(save_dir + '/train_dsc.json', 'w') as f:
    #     json.dump(train_dsc_dict, f)
    # with open(save_dir + '/val_dsc.json', 'w') as f:
    #     json.dump(eval_dsc_dict, f)
    print("DSC data has been saved to:", save_dir)

def comput_fig(img):
    img = img.detach().cpu().numpy()[0, 0, 48:64, :, :]
    fig = plt.figure(figsize=(12, 12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)

def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 192, 224)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j+line_thickness-1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i+line_thickness-1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img

def save_checkpoint(state, save_dir='models_affine/', filename='checkpoint.pth.tar', max_model_num=12):
    torch.save(state, save_dir + filename)
    model_lists = natsorted(glob.glob(save_dir + '*.pth*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))


if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 1
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('  GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main()
