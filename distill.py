
# Dataset Distillation via Trajectory Matching

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from tqdm import tqdm
from utils import get_dataset, get_network, get_eval_pool, evaluate_synset, get_time, DiffAugment, ParamDiffAug
import wandb
import copy
import random
from reparam_module import ReparamModule
import gc

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def main(args):
    # 参数检查
    if args.zca and args.texture:
        raise AssertionError("Cannot use zca and texture together")

    if args.texture and args.pix_init == "real":
        print("WARNING: Using texture with real initialization will take a very long time to smooth out the boundaries between images.")

    if args.max_experts is not None and args.max_files is not None:
        args.total_experts = args.max_experts * args.max_files  # max_experts: per file

    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))

    # # 初始化配置
    # Input: A: Differentiable augmentation function
    # DSA 的目标是生成合成数据 D_syns
    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 评估 D_syn 性能的迭代次数(等差数列) [0, 100, 200, ..., 4900, 5000]
    eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_it).tolist()
    
    # # Reset args

    # load (ZCA过的) dataset
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)
    # 获取用于评估的 model list 
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)
    # record performances of all experiments (not used)
    accs_all_exps = dict()
    for key in model_eval_pool:
        accs_all_exps[key] = []
    data_save = []

    im_res = im_size[0]
    args.im_size = im_size

    # DSA augmentation
    if args.dsa:
        # args.epoch_eval_train = 1000
        args.dc_aug_param = None
    args.dsa_param = ParamDiffAug()  # 创建 ParamDiffAug 类对象，用于控制DSA数据增强方式
    dsa_params = args.dsa_param  # 暂存，后面重新赋值给args对象的属性

    # ZCA whitening
    if args.zca:
        zca_trans = args.zca_trans  # 暂存，后面重新赋值给args对象的属性
    else:
        zca_trans = None
    
    # initialize wandb log
    wandb.init(sync_tensorboard=False,  # 不与 TensorBoard 同步
               project="DatasetDistillation",
               job_type="CleanRepo",  # 工作类型
               config=args,
               )
    
    # 重新定义 args 对象为空的匿名类，为了重置 args 对象的内容，以便后续的参数赋值
    args = type('', (), {})()
    # 将从 W&B 配置中读取的参数 重新赋值给 args 对象
    for key in wandb.config._items:
        setattr(args, key, wandb.config._items[key])

    # 将之前创建的对象 赋值给args对象的属性，以便后续用于 DSA 和 ZCA白化
    args.dsa_param = dsa_params
    args.zca_trans = zca_trans

    if args.batch_syn is None:  # batch_syn: 合成图像的批次大小，内存不足时使用
        args.batch_syn = num_classes * args.ipc  # ipc: 每个类别的图像数量 img_num/class
    
    # 判断当前计算机是否有多个GPU，若为TRUE则启用分布式训练
    args.distributed = torch.cuda.device_count() > 1

    # print('Hyper-parameters: \n', args.__dict__) or: 
    # 分行打印 所有超参数
    print("Hyper-parameters:")
    for key, value in args.__dict__.items():
        print(f"{key}: {value}")
    print('Evaluation model pool: ', model_eval_pool)


    ''' organize the real dataset '''
    images_all = []  # 存储所有真实图像
    labels_all = []  # 和对应的标签
    # 索引list，每个子列表长度为 num_classes，其中存储属于相应类别的图像在 images_all 和 labels_all 中的索引
    indices_class = [[] for c in range(num_classes)]

    # --- --- 以下code直到 'for ch in range(channel)' 同 buffer.py
    print("BUILDING DATASET")
    # 遍历 dst_train 数据集中每个样本
    for i in tqdm(range(len(dst_train))):
        sample = dst_train[i]
        images_all.append(torch.unsqueeze(sample[0], dim=0))  # img: (C, H, W) --> (1, C, H, W) 
        labels_all.append(class_map[torch.tensor(sample[1]).item()])
    # 添加图像索引到相应类别子列表中
    for i, lab in tqdm(enumerate(labels_all)):
        indices_class[lab].append(i)
    
    images_all = torch.cat(images_all, dim=0).to("cpu")
    labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")

    for c in range(num_classes):
        print('class c = %d: %d real images'%(c, len(indices_class[c])))

    for ch in range(channel):
        print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))
    print('\n')
    # --- ---

    def get_images(c, n):  # get random n images from class c (for initialize D_syn from random real images)
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        return images_all[idx_shuffle]

    # # line 1: Initialize distilled data D_syn ~ D_real. (D_syn: label_syn, image_syn, syn_lr)
    # 将合成数据 D_syn 初始化为与真实数据 D_real 类似的分布
    ''' initialize the synthetic data '''
    
    # label_syn 初始化
    # [...]为每个类别生成一个长度为 args.ipc 的二维数组; view(-1)将tensor 展平为一维数组，以便与合成图像的数量相匹配
    # i.e. ipc=3, num_classes=10: label_syn=[0,0,0, 1,1,1, ..., 9,9,9], shape=(num_classes*ipc, )
    label_syn = torch.tensor([np.ones(args.ipc, dtype=np.int_)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1)
    
    # image_syn 初始化（每个类别的合成图像，im_size[0]和[1]是图片高度宽度）
    if args.texture:
        image_syn = torch.randn(size=(num_classes*args.ipc, channel, im_size[0]*args.canvas_size, im_size[1]*args.canvas_size), dtype=torch.float)
    else:
        image_syn = torch.randn(size=(num_classes*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float)
    
    # syn_lr 初始化
    syn_lr = torch.tensor(args.lr_teacher).to(args.device)

    # 是否从真实图像初始化 D_syn
    if args.pix_init == 'real':
        print('initialize synthetic data from random real images')
        if args.texture:
            # 遍历每个类别和合成图像的位置
            for c in range(num_classes):
                for i in range(args.canvas_size):  #　合成图像的行索引
                    for j in range(args.canvas_size):  # 合成图像的列索引
                        # 用随机选取的真实图像的一部分来填充
                        image_syn.data[c*args.ipc:(c+1)*args.ipc, :, i*im_size[0]:(i+1)*im_size[0], j*im_size[1]:(j+1)*im_size[1]] = torch.cat(
                            [get_images(c, 1).detach().data for s in range(args.ipc)])  # .data[...]: 合成图像中当前类别的索引范围, :, 当前位置的行范围; 当前位置的列范围。cat(...): 选取类别c中的ipc个真实图像并将它们堆叠在一起，以填充合成图像中的每个位置。
        # no_texture: 直接将 args.ipc 个真实图像复制到每个类别的合成图像位置上
        for c in range(num_classes):
            image_syn.data[c*args.ipc:(c+1)*args.ipc] = get_images(c, args.ipc).detach().data
    else:  # 'noise' 合成图像的每个像素都将是随机生成的值
        print('initialize synthetic data from random noise')

    ''' training '''
    # line 2: Initialize trainable learning rate α := α0 for apply D_syn
    image_syn = image_syn.detach().to(args.device).requires_grad_(True)
    syn_lr = syn_lr.detach().to(args.device).requires_grad_(True)

    # 创建优化器 to-do
    optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_img, momentum=0.5)  # 用于更新 image_syn
    optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.5)  # 用于更新 syn_lr
    optimizer_img.zero_grad()  # 梯度清零，以便在每次迭代开始时重新计算梯度
    # 定义损失函数
    criterion = nn.CrossEntropyLoss().to(args.device)
    print('%s training begins' % get_time())

    # Input: {τ∗_i}: set of expert parameter trajectories trained on D_real
    # 1. expert_dir 的构建
    expert_dir = os.path.join(args.buffer_path, args.dataset)  # ./buffers/CIFAR10
    if args.dataset == "ImageNet":
        expert_dir = os.path.join(expert_dir, args.subset, str(args.res))
    if args.dataset in ["CIFAR10", "CIFAR100"] and not args.zca:
        expert_dir += "_NO_ZCA"
    expert_dir = os.path.join(expert_dir, args.model)
    print("Expert Dir: {}".format(expert_dir))

    # 2. 加载提前训练好的 expert trajectory
    if args.load_all:  # 从所有可用文件中 load 缓冲区数据，内容合并到 buffer 中
        buffer = []  # all data of {}.pt
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            buffer = buffer + torch.load(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))
    else:  # 从特定的文件中 load 缓冲区数据到 buffer 中，并根据 max_experts 截取出特定数量的专家数据
        expert_files = []  # dirs: [expert_dir/replay_buffer_{}.pt, ..]
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            expert_files.append(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        # change: 注释下面两行
        # if n == 0:
        #     raise AssertionError("No buffers detected at {}".format(expert_dir))
        
        file_idx = 0  # idx of current {}.pt file
        expert_idx = 0  # 选择预训练参数的索引
        random.shuffle(expert_files)

        # 对数据进行限制和截取
        if args.max_files is not None:
            expert_files = expert_files[:args.max_files]  # max_files: max num of {}.pt files. e.g. expert_files包含最多5个.pt文件路径
        print("loading file {}".format(expert_files[file_idx]))
        buffer = torch.load(expert_files[file_idx])  # load data from ./{}.pt
        if args.max_experts is not None:  # per file, default=None
            buffer = buffer[:args.max_experts]
        random.shuffle(buffer)

    best_acc = {m: 0 for m in model_eval_pool}
    best_std = {m: 0 for m in model_eval_pool}

    
    # # line 3: for each distillation step... do
    # 模型训练循环，Iteration即 distillation steps
    for it in range(0, args.Iteration+1):
        save_this_it = False  # 是否在当前步骤保存最佳合成数据

        # writer.add_scalar('Progress', it, it)
        wandb.log({"Progress": it}, step=it)

        # 对 D_syn 进行评估，记录评估结果到 W&B 日志
        ''' Evaluate synthetic data '''
        if it in eval_it_pool:
            # 遍历每个评估模型
            for model_eval in model_eval_pool:
                # 打印训练模型名称、当前评估模型名称、当前迭代数
                print('\n-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))
                # 打印数据增强信息
                if args.dsa:
                    print('DSA augmentation strategy: \n', args.dsa_strategy)
                    print('DSA augmentation parameters: \n', args.dsa_param.__dict__)
                else:  # no dsa
                    print('DC augmentation parameters: \n', args.dc_aug_param)

                accs_test = []
                accs_train = []
                
                # 多次对合成数据进行评估
                for it_eval in range(args.num_eval): # num_eval 在多少个网络上进行评估
                    # get a random model
                    net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device)
                    
                    # 评估用的标签和图像数据
                    eval_labs = label_syn
                    with torch.no_grad():
                        image_save = image_syn
                    
                    # 创建用于评估的图像和标签副本：确保在评估过程中不会对原始数据造成任何影响
                    image_syn_eval, label_syn_eval = copy.deepcopy(image_save.detach()), copy.deepcopy(eval_labs.detach()) # avoid any unaware modification

                    args.lr_net = syn_lr.item()  # lr of evaluation
                    _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args, texture=args.texture)
                    accs_test.append(acc_test)
                    accs_train.append(acc_train)
                
                # 计算评估结果的平均值和标准差
                accs_test = np.array(accs_test)
                accs_train = np.array(accs_train)
                acc_test_mean = np.mean(accs_test)
                acc_test_std = np.std(accs_test)

                # 判断是否要保存当前结果
                if acc_test_mean > best_acc[model_eval]:
                    best_acc[model_eval] = acc_test_mean
                    best_std[model_eval] = acc_test_std
                    save_this_it = True
                # 打印评估结果信息，记录到 W&B 日志
                print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs_test), model_eval, acc_test_mean, acc_test_std))
                wandb.log({'Accuracy/{}'.format(model_eval): acc_test_mean}, step=it)
                wandb.log({'Max_Accuracy/{}'.format(model_eval): best_acc[model_eval]}, step=it)
                wandb.log({'Std/{}'.format(model_eval): acc_test_std}, step=it)
                wandb.log({'Max_Std/{}'.format(model_eval): best_std[model_eval]}, step=it)

        # 在特定的迭代步骤(eval_it_pool中定义的步骤)保存和可视化合成数据，并将相关信息记录到W&B日志中
        if it in eval_it_pool and (save_this_it or it % 1000 == 0):
            with torch.no_grad():  # 确保此代码块内 不会产生任何梯度计算
                image_save = image_syn.cuda()  # 将合成的图像移到GPU上
                # 保存图像和标签的目录路径
                save_dir = os.path.join(".", "logged_files", args.dataset, wandb.run.name)

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                # 1.1 文件保存
                torch.save(image_save.cpu(), os.path.join(save_dir, "images_{}.pt".format(it)))  # 保存为PyTorch张量，文件名包含当前迭代步骤 it
                torch.save(label_syn.cpu(), os.path.join(save_dir, "labels_{}.pt".format(it)))

                # 1.2 当前it 是一个保存最佳合成数据的步骤
                if save_this_it:
                    torch.save(image_save.cpu(), os.path.join(save_dir, "images_best.pt".format(it)))
                    torch.save(label_syn.cpu(), os.path.join(save_dir, "labels_best.pt".format(it)))

                # 2. 可视化
                wandb.log({"Pixels": wandb.Histogram(torch.nan_to_num(image_syn.detach().cpu()))}, step=it)  # 将合成图像的像素值分布 作为直方图 记录到W&B日志中

                # 每个类别的图像数<50，或设置为强制保存
                if args.ipc < 50 or args.force_save:
                    upsampled = image_save  # 之后进行裁剪/上采样
                    # 图像尺寸调整，以便在W&B中显示较小的图像网格
                    if args.dataset != "ImageNet":
                        upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                        upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                    grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)  # 将图像网格制作为torchvision图像，以便在W&B中显示
                    wandb.log({"Synthetic_Images": wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)  # 将图像网格记录为W&B的图像类型，显示合成图像
                    wandb.log({'Synthetic_Pixels': wandb.Histogram(torch.nan_to_num(image_save.detach().cpu()))}, step=it)  # 将合成图像的像素值分布记录到W&B日志中
                    
                    # 对合成图像进行剪裁，使得像素值在一定的标准差范围内
                    for clip_val in [2.5]:
                        std = torch.std(image_save)
                        mean = torch.mean(image_save)
                        upsampled = torch.clip(image_save, min=mean-clip_val*std, max=mean+clip_val*std)
                        if args.dataset != "ImageNet":
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                        grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                        wandb.log({"Clipped_Synthetic_Images/std_{}".format(clip_val): wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)

                    # 经过ZCA白化的数据，进行类似的可视化和记录操作，分别记录重建的图像和像素值分布
                    if args.zca: 
                        image_save = image_save.to(args.device)
                        image_save = args.zca_trans.inverse_transform(image_save)
                        image_save.cpu()

                        torch.save(image_save.cpu(), os.path.join(save_dir, "images_zca_{}.pt".format(it)))

                        upsampled = image_save
                        if args.dataset != "ImageNet":
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                        grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                        wandb.log({"Reconstructed_Images": wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)
                        wandb.log({'Reconstructed_Pixels': wandb.Histogram(torch.nan_to_num(image_save.detach().cpu()))}, step=it)

                        for clip_val in [2.5]:
                            std = torch.std(image_save)
                            mean = torch.mean(image_save)
                            upsampled = torch.clip(image_save, min=mean-clip_val*std, max=mean+clip_val*std)
                            if args.dataset != "ImageNet":
                                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                            grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                            wandb.log({"Clipped_Reconstructed_Images/std_{}".format(clip_val): wandb.Image(
                                torch.nan_to_num(grid.detach().cpu()))}, step=it)

        # record syn_lr of each iter on W&B
        wandb.log({"Synthetic_LR": syn_lr.detach().cpu()}, step=it)

        # 构建和配置 student_net
        student_net = get_network(args.model, channel, num_classes, im_size, dist=False).to(args.device)  # get a random model.  dist表示分布式训练
        student_net = ReparamModule(student_net)  # 将其包装在ReparamModule模块中，为了支持参数的重参数化

        if args.distributed:  # 用 DataParallel 对象将模型包装起来，这允许在多个GPU上并行训练
            student_net = torch.nn.DataParallel(student_net)
        
        student_net.train()  # 设置为训练模式，从而启用一些训练功能，如 Dropout 和批量归一化的更新

        # .paramters(): generate each param tensor; p.size(): shape of each tensor; np.prod(): compute product
        # []: a list of num_elems, each presents num_elem of a param tensor
        num_params = sum([np.prod(p.size()) for p in (student_net.parameters())])  # num_elem of all params in net

        # # line 4: Sample expert trajectory: τ* ~ {τ*(i)}, with τ*={θ*(i)}(0->T)
        if args.load_all:
            expert_trajectory = buffer[np.random.randint(0, len(buffer))]
        else:  # default 'False'
            expert_trajectory = buffer[expert_idx]
            expert_idx += 1
            if expert_idx == len(buffer):
                expert_idx = 0
                file_idx += 1
                # 重置 file_idx
                if file_idx == len(expert_files):
                    file_idx = 0
                    random.shuffle(expert_files)
                print("loading file {}".format(expert_files[file_idx]))
                if args.max_files != 1:
                    del buffer
                    buffer = torch.load(expert_files[file_idx])
                if args.max_experts is not None:
                    buffer = buffer[:args.max_experts]
                random.shuffle(buffer)
        
        # # line 5: Choose a random start epoch (t <= T+)
        start_epoch = np.random.randint(0, args.max_start_epoch)

        # # line 6, 7: Initialize student network with expert params
        # expert_trajectory params of epoch t
        starting_params = expert_trajectory[start_epoch]  # θ*(t), shape=(num_param_tensor, ), each param has diff shape
        
        # target of student_net optimization
        target_params = expert_trajectory[start_epoch+args.expert_epochs]  # θ*(t+M), shape=(num_param_tensor, )
        # concat each p in target_params on dimension 0
        target_params = torch.cat([p.data.to(args.device).reshape(-1) for p in target_params], 0)  # shape=(1, total_num_param_tensor)
        
        # θ^(t) := θ*(t), shape=(1, total_num_param_tensor), require grad, for later updating. θ^(t), then θ^(t+1), ... θ^(t+N)
        student_params = [torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0).requires_grad_(True)]
        
        # after initializing student_params, for calculating param_dist later. shape=(1, total_num_param_tensor)
        starting_params = torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0)

        # as training input
        syn_images = image_syn
        y_hat = label_syn.to(args.device)

        # record loss and distance
        param_loss_list = []
        param_dist_list = []  # Param. Distance to Target
        indices_chunks = []  # restore indices: to get mini-batch of each step

        # # line 8: for n = 0 → N − 1 do
        # update student_params on D_syn, approximating target_params
        for step in range(args.syn_steps):
            # line 9, 10: Sample a mini-batch of distilled images (SGD: using a mini-batch to calculate grad each step)
            # b_(t+n) ~ D_syn (n: step)
            if not indices_chunks:  # re-generate indices (re-sample a mini-batch)
                indices = torch.randperm(len(syn_images))  # indices: tensor([207, 83,...])
                indices_chunks = list(torch.split(indices, args.batch_syn))  # split indices into batch_syn size, each chunk contains a batch of indices (batch_syn=None, one chunk)
            these_indices = indices_chunks.pop()  # pop the last chunk
            # a mini-batch
            x = syn_images[these_indices]
            this_y = y_hat[these_indices]

            if args.texture:  # 纹理增强
                # img 随机平移，增加数据多样性。roll()沿给定维数(1, 2)滚动张量
                x = torch.cat([torch.stack([torch.roll(im, (torch.randint(im_size[0]*args.canvas_size, (1,)), 
                                                            torch.randint(im_size[1]*args.canvas_size, (1,))),  # 生成两个随机整数，分别用于高度和宽度上的平移（im_size[0]和[1]是图像高度和宽度）
                                                       (1, 2))[:, :im_size[0], :im_size[1]] for im in x])  # 从滚动后的图像中截取指定的区域，确保输入输出尺寸一致
                               for _ in range(args.canvas_samples)])  # stack()将得到的多张img叠加，生成一个新的图像列表。cat()将平移后的img列表与原始图像 x 垂直拼接
                # label 进行相同次数的重复
                this_y = torch.cat([this_y for _ in range(args.canvas_samples)])

            if args.dsa and (not args.no_aug):  # DSA数据增强
                x = DiffAugment(x, args.dsa_strategy, param=args.dsa_param)
            
            # # line 11, 12: Update student network w.r.t. classification loss
            if args.distributed:  # 若有多个 GPU 则启用分布式训练，将最新的 student_params 复制为多个副本
                forward_params = student_params[-1].unsqueeze(0).expand(torch.cuda.device_count(), -1)
            else:
                forward_params = student_params[-1]
            
            # 输入 x 进行前向传播，得到输出 x
            x = student_net(x, flat_param=forward_params)
            '''
            buffer = torch.load(expert_files[file_idx])
            expert_trajectory = buffer[expert_idx]            # 相当于buffer[-1]
            starting_params = expert_trajectory[start_epoch]  # 相当于buffer[-1][-1]  start_epoch 第2个 -1
            student_params = [torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0).requires_grad_(True)]
            forward_params = student_params[-1]
            '''
            # 计算交叉熵损失
            ce_loss = criterion(x, this_y)
            # 计算交叉熵损失对 student_params 的梯度
            grad = torch.autograd.grad(ce_loss, student_params[-1], create_graph=True)[0]
            # 更新student_params (减去 梯度与学习率的乘积)，逼近目标参数
            student_params.append(student_params[-1] - syn_lr * grad)

        # # line 14, 15: computing loss between ending student and expert params
        # 初始化为零
        param_loss = torch.tensor(0.0).to(args.device)
        param_dist = torch.tensor(0.0).to(args.device)
        # 计算均方误差损失 ending student & expert params
        param_loss += torch.nn.functional.mse_loss(student_params[-1], target_params, reduction="sum")  # ||θ^(t+N) - θ*(t+M)||
        param_dist += torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")     # ||θ*(t) - θ*(t+M)||
        param_loss_list.append(param_loss)
        param_dist_list.append(param_dist)
        param_loss /= num_params  # 平均 loss 和 distance
        param_dist /= num_params  # num_params: total num of net

        param_loss /= param_dist  # normalized loss
        grand_loss = param_loss   # loss of trajectory matching

        # line 16: Update D_syn and α w.r.t loss
        optimizer_img.zero_grad()  # clear grad of all optimized Variables
        optimizer_lr.zero_grad()

        grand_loss.backward()      # backward: calculate grad

        optimizer_img.step()       # optimize img_syn, syn_lr
        optimizer_lr.step()

        # 记录到 W&B 日志
        wandb.log({"Grand_Loss": grand_loss.detach().cpu(),
                   "Start_Epoch": start_epoch})
        # 释放之前用于计算梯度的 student_params
        for _ in student_params:
            del _

        # 定时清内存
        gc.collect()
        torch.cuda.empty_cache()

        # 打印训练进度：每隔一段迭代次数，打印当前迭代的时间和损失
        if it % 10 == 0:
            print('%s iter = %04d, loss = %.4f' % (get_time(), it, grand_loss.item()))
    # 结束训练：完成训练后，结束 WandB 的记录
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')

    parser.add_argument('--subset', type=str, default='imagenette', help='ImageNet subset. This only does anything when --dataset=ImageNet')

    parser.add_argument('--model', type=str, default='ConvNet', help='model')

    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')

    parser.add_argument('--eval_mode', type=str, default='S',
                        help='eval_mode, check utils.py for more info')

    parser.add_argument('--num_eval', type=int, default=5, help='how many networks to evaluate on')

    parser.add_argument('--eval_it', type=int, default=100, help='how often to evaluate')

    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=5000, help='how many distillation steps to perform')

    parser.add_argument('--lr_img', type=float, default=1000, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_lr', type=float, default=1e-05, help='learning rate for updating... learning rate')
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='initialization for synthetic learning rate')

    parser.add_argument('--lr_init', type=float, default=0.01, help='how to init lr (alpha)')

    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_syn', type=int, default=None, help='should only use this if you run out of VRAM')  # 内存不足时使用
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')

    parser.add_argument('--pix_init', type=str, default='real', choices=["noise", "real"],
                        help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--res', type=int, default=128, help='resolution')
    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')
    # DSA在训练迭代中对采样的真实批次和合成批次 中的所有数据点 应用相同的参数增强（例如旋转）
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')

    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')

    parser.add_argument('--expert_epochs', type=int, default=3, help='how many expert epochs the target params are')  # M+
    parser.add_argument('--syn_steps', type=int, default=20, help='how many steps to take on synthetic data')  # N
    parser.add_argument('--max_start_epoch', type=int, default=25, help='max epoch we can start at')  # T+

    parser.add_argument('--zca', action='store_true', help="do ZCA whitening")  # 降低输入的冗余性

    parser.add_argument('--load_all', action='store_true', help="only use if you can fit all expert trajectories into RAM")

    parser.add_argument('--no_aug', type=bool, default=False, help='this turns off diff aug during distillation')

    parser.add_argument('--texture', action='store_true', help="will distill textures instead")  # False
    parser.add_argument('--canvas_size', type=int, default=2, help='size of synthetic canvas')   # for texture
    parser.add_argument('--canvas_samples', type=int, default=1, help='number of canvas samples per iteration')

    parser.add_argument('--max_files', type=int, default=None, help='number of expert files to read (leave as None unless doing ablations)')
    parser.add_argument('--max_experts', type=int, default=None, help='number of experts to read per file (leave as None unless doing ablations)')

    parser.add_argument('--force_save', action='store_true', help='this will save images for 50ipc')

    args = parser.parse_args()

    main(args)
