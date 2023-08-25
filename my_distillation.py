import os
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import tqdm
import kornia as K
import wandb
import numpy as np
from my_utils import TensorDataset, get_network, ParamDiffAug, DiffAugment


def main(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # args.dataset == 'CIFAR10':
    channel = 3
    im_size = (32, 32)
    num_classes = 10
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    args.im_size = im_size

    # Define data transformations
    if args.zca:
        transform = transforms.Compose([transforms.ToTensor()])
    else:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=mean, std=std)  # Normalize images
                                        ])

    # Load the saved data
    img_path = os.path.join(args.train_path, 'images_best.pt')  # '/logged_files/CIFAR10/sweet-wood-12/images_best.pt'
    lab_path = os.path.join(args.train_path, 'labels_best.pt')
    images = torch.load(img_path)
    labels = torch.load(lab_path)

    # Create Dataset
    train_dataset = TensorDataset(images, labels)
    test_dataset = datasets.CIFAR10(root=args.data_path, train=False, transform=transform, download=True)

    if args.zca:
        print("Train ZCA")
        images, labels = [], []
        for i in tqdm.tqdm(range(len(train_dataset))):
            im, lab = train_dataset[i]
            images.append(im)
            labels.append(lab)
        images = torch.stack(images, dim=0).to(args.device)
        labels = torch.tensor(labels, dtype=torch.long, device="cpu")
        zca = K.enhance.ZCAWhitening(eps=0.1, compute_inv=True)
        zca.fit(images)
        zca_images = zca(images).to("cpu")
        train_dataset = TensorDataset(zca_images, labels)

        print("Test ZCA")
        images, labels = [], []
        for i in tqdm.tqdm(range(len(test_dataset))):
            im, lab = test_dataset[i]
            images.append(im)
            labels.append(lab)
        images = torch.stack(images, dim=0).to(args.device)
        labels = torch.tensor(labels, dtype=torch.long, device="cpu")
        zca_images = zca(images).to("cpu")
        test_dataset = TensorDataset(zca_images, labels)

        args.zca_trans = zca

    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
    
    # Get train_images and train_labels
    best_imgs, best_labs = [], []

    for i in range(len(train_dataset)):
        im, lab = train_dataset[i]
        best_imgs.append(im)
        best_labs.append(lab)

    train_images = torch.stack(best_imgs).detach()
    train_labels = torch.tensor(best_labs).detach()

    args.dsa_param = ParamDiffAug()
    dsa_params = args.dsa_param     # tmp store, re-assign to args
    
    if args.zca:
        zca_trans = args.zca_trans  # tmp store
    else:
        zca_trans = None

    # initialize wandb log
    wandb.init(sync_tensorboard=False,          # not synchro with TensorBoard
               project="My_Synset_Evaluation",
               job_type="CleanRepo",
               config=args)
    args = type('', (), {})()
    for key in wandb.config._items:
        setattr(args, key, wandb.config._items[key])

    args.dsa_param = dsa_params
    args.zca_trans = zca_trans

    print('Hyper-parameters: \n\n', args.__dict__)
    if args.dsa:
        print('DSA augmentation strategy: \n', args.dsa_strategy)
        print('DSA augmentation parameters: \n', args.dsa_param.__dict__, '\n')
    
    best_acc = 0
    
    for it in range(args.Iteration):
        wandb.log({"Progress": it}, step=it)
        print('\n-------------------------\nIteration = %d'%(it))
        # Initialize the model
        model = get_network('ConvNet', channel, num_classes, im_size).to(args.device)
        model.train()

        num_epochs = args.Epoch
        lr = float(args.syn_lr)
        lr_schedule = [num_epochs//2+1]

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss().to(args.device)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)  # for updating params

        for epoch in range(args.Epoch):  # 0~1999
            '''Train'''
            # using 'all data' each epoch (GD)
            with torch.no_grad():
                x = train_images.to(args.device)
            y = train_labels.to(args.device)
            
            if args.dsa:
                x = DiffAugment(x, args.dsa_strategy, param=args.dsa_param)
            
            y_pred = model(x)
            loss = criterion(y_pred, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if epoch % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

            '''Test'''
            # once a iter
            if (epoch+1) == num_epochs:
                model.eval()
                test_loss, test_acc, num_exp = 0, 0, 0
                with torch.no_grad():
                    for i_batch, datum in enumerate(test_loader):
                        img = datum[0].float().to(args.device)
                        lab = datum[1].long().to(args.device)

                        if args.dsa:
                            img = DiffAugment(img, args.dsa_strategy, param=args.dsa_param)

                        n_b = lab.shape[0]  # num of this batch

                        output = model(img)
                        batch_loss = criterion(output, lab)
                        batch_acc = np.sum(np.equal( np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy() ))
                        
                        test_loss += batch_loss.item()*n_b
                        test_acc += batch_acc
                        num_exp += n_b
                    
                    test_loss /= num_exp
                    test_acc /= num_exp

                    if test_acc > best_acc:
                        best_acc = test_acc
                
                print(f'\nEpoch [{epoch+1}], Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
                wandb.log({'Accuracy': test_acc}, step=it)
                wandb.log({'Max Accuracy': best_acc}, step=it)
            
            if epoch in lr_schedule:
                lr *= 0.1
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
            
    print(f'Max Accuracy = {best_acc}')

    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNetD3', help='model')
    parser.add_argument('--ipc', type=int, default=50, help='image(s) per class')
    
    parser.add_argument('--zca', action='store_true', help="do ZCA whitening")
    parser.add_argument('--dsa', type=bool, default=True, help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate')

    parser.add_argument('--Iteration', type=int, default=100, help='how many Iterations to perform')
    parser.add_argument('--Epoch', type=int, default=1000, help='how many Epochs to perform per iter')

    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')

    parser.add_argument('--syn_lr', type=float, default=0.01, help='initialization for optimizer learning rate')
    
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--train_path', type=str, default='./logged_files/CIFAR10/sweet-wood-12', help='best image path')
    
    args = parser.parse_args()
    main(args)