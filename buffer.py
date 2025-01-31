
#  Pre-compute expert trajectories using only real data

'''
在指定数据集上训练深度神经网络模型：
数据集加载、数据增强、模型训练、参数保存
'''

import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm  # 进度条可视化库，monitor progress of program execution
from utils import get_dataset, get_network, get_daparam,\
    TensorDataset, epoch, ParamDiffAug
import copy

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main(args):
    # 参数解析
    args.dsa = True if args.dsa == 'True' else False
    # 环境设置
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()

    ## 数据集加载
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args) # dst_train: 一个用于训练的数据集对象，class_map: 字典{原标签x:新的整数标签i}
    
    # print('\n================== Exp %d ==================\n '%exp)
    print('Hyper-parameters: \n', args.__dict__)

    ## 设置保存模型参数trajectory的路径
    save_dir = os.path.join(args.buffer_path, args.dataset)
    if args.dataset == "ImageNet":
        save_dir = os.path.join(save_dir, args.subset, str(args.res))
    if args.dataset in ["CIFAR10", "CIFAR100"] and not args.zca:
        save_dir += "_NO_ZCA"
    save_dir = os.path.join(save_dir, args.model)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ## 数据预处理：将图像数据和对应的标签进行整理和存储
    ''' organize the real dataset '''
    images_all = []
    labels_all = []
    indices_class = [[] for c in range(num_classes)]  # 存储每个类别的图像在 images_all 中的索引
    print("BUILDING DATASET")
    
    # 整理训练集实例中每个样本，并分别添加到两个list中
    for i in tqdm(range(len(dst_train))):
        sample = dst_train[i]  # 当前样本：sample[0]是img; sample[1]是label
        images_all.append(torch.unsqueeze(sample[0], dim=0))  # img增加一个维度（第一维），再append
        labels_all.append(class_map[torch.tensor(sample[1]).item()])  # 通过class_map将原始标签映射为新的标签，再append

    # 遍历整理后的新标签列表
    for i, lab in tqdm(enumerate(labels_all)):
        indices_class[lab].append(i)  # 将当前索引i 添加到对应类别lab 的indices_class列表中
    # 合并为张量
    images_all = torch.cat(images_all, dim=0).to("cpu")  # 将所有图像数据堆叠起来，形成一个张量
    labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")  # list -> tensor

    # 打印每个类别的索引，以及该类别中的真实图像数量
    for c in range(num_classes):
        print('class c = %d: %d real images'%(c, len(indices_class[c])))
        # class c = 0：1200 real images
        # ...
        # class c = 9: 950 real images

    # 计算并打印 real image 每个通道的均值和标准差
    for ch in range(channel):
        print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))

    ##　创建 loss function
    criterion = nn.CrossEntropyLoss().to(args.device)

    trajectories = []  # 存储训练过程中模型的参数变化 to guide the distillation of our synthetic dataset

    dst_train = TensorDataset(copy.deepcopy(images_all.detach()), copy.deepcopy(labels_all.detach()))  #　创建一个新的数据集，detach()返回的tensor和原始tensor共同一个内存，但是该tensor的requires_grad永远为false
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)  # 数据加载器，将数据分成批次进行训练

    ## 数据集扩充
    ''' set augmentation for whole-dataset training '''
    args.dc_aug_param = get_daparam(args.dataset, args.model, args.model, None)  # 数据增强参数
    args.dc_aug_param['strategy'] = 'crop_scale_rotate'  # 数据增强策略：裁剪，缩放，旋转
    print('DC augmentation parameters: \n', args.dc_aug_param)

    ## 模型训练循环，每次迭代都会训练一个 D_syn的 teacher模型，保存对应trajectories
    for it in range(0, args.num_experts):
        # 训练 distilled data D_syn
        ''' Train synthetic data '''
        teacher_net = get_network(args.model, channel, num_classes, im_size).to(args.device)  # get a random model
        teacher_net.train()  # 设置为训练模式，从而启用模型的训练特性，如 Dropout等
        lr = args.lr_teacher
        teacher_optim = torch.optim.SGD(teacher_net.parameters(), lr=lr, momentum=args.mom, weight_decay=args.l2)  # optimizer_img for synthetic data 创建一个随机梯度下降（SGD）优化器，用于更新模型参数
        teacher_optim.zero_grad()  # 梯度置零

        timestamps = []  # 记录训练过程中的参数变化（时间戳）
        timestamps.append([p.detach().cpu() for p in teacher_net.parameters()])  # 记录当前模型的参数，以便在训练过程中跟踪参数变化

        lr_schedule = [args.train_epochs // 2 + 1]  # 学习率调整的时间表，在训练的某些特定 epoch 时，将会减小学习率
        
        # 模型训练的主要循环，调用utils.py的epoch函数
        for e in range(args.train_epochs):

            train_loss, train_acc = epoch("train", dataloader=trainloader, net=teacher_net, optimizer=teacher_optim,
                                        criterion=criterion, args=args, aug=True)

            test_loss, test_acc = epoch("test", dataloader=testloader, net=teacher_net, optimizer=None,
                                        criterion=criterion, args=args, aug=False)

            print("Itr: {}\tEpoch: {}\tTrain Acc: {}\tTest Acc: {}".format(it, e, train_acc, test_acc))
            
            # record model parameters of each epoch
            timestamps.append([p.detach().cpu() for p in teacher_net.parameters()])
            
            # 学习率衰减
            if e in lr_schedule and args.decay:  # args.decay=True 启用了学习率衰减
                lr *= 0.1  # 减小到当前 lr 的1/10
                teacher_optim = torch.optim.SGD(teacher_net.parameters(), lr=lr, momentum=args.mom, weight_decay=args.l2)  # 重新创建一个 SGD 优化器，以应用新的学习率
                teacher_optim.zero_grad()  # 将优化器的梯度置零，为下一个 epoch 的梯度计算做准备

        trajectories.append(timestamps)

        # 参数保存（trajectories）
        if len(trajectories) == args.save_interval:
            n = 0
            while os.path.exists(os.path.join(save_dir, "replay_buffer_{}.pt".format(n))):  # 找到一个未存在的文件名后缀编号n
                n += 1
            print("Saving {}".format(os.path.join(save_dir, "replay_buffer_{}.pt".format(n))))
            torch.save(trajectories, os.path.join(save_dir, "replay_buffer_{}.pt".format(n)))  #文件名：replay_buffer_n.pt，文件格式：PyTorch格式的二进制文件
            trajectories = []  # 清空


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--subset', type=str, default='imagenette', help='subset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--num_experts', type=int, default=100, help='training iterations')
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real loader')
    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')
    parser.add_argument('--train_epochs', type=int, default=50)
    parser.add_argument('--zca', action='store_true', help='do ZCA whitening')
    parser.add_argument('--decay', action='store_true')
    parser.add_argument('--mom', type=float, default=0, help='momentum')
    parser.add_argument('--l2', type=float, default=0, help='l2 regularization')
    parser.add_argument('--save_interval', type=int, default=10, help='trajectories num of each save opt')

    args = parser.parse_args()
    main(args)


