import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import numpy as np
from PIL import Image
from torchvision.transforms.autoaugment import AutoAugment, AutoAugmentPolicy
from sklearn.model_selection import train_test_split
from collections import Counter
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import roc_auc_score

from InceptionMamba import VSSM as InceptionMamba  # import model


class NpyDataset(Dataset):
    def __init__(self, image_file, label_file, transform=None):
        # 加载图像和标签数据
        self.images = np.load(image_file)
        self.labels = np.load(label_file)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 获取单个样本
        image = self.images[idx]
        label = self.labels[idx]

        # 如果标签是数组，取第一个元素
        if isinstance(label, np.ndarray):
            label = label.item()  # 转换为标量

        # 如果图像是numpy数组，转换为PIL图像
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # 如果定义了transform，应用它
        if self.transform:
            image = self.transform(image)

        return image, label


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([
                                    transforms.RandomResizedCrop(224),
                                    transforms.Grayscale(num_output_channels=3),  
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                                    # transforms.Normalize((0.5,), (0.5,))]),#灰度图
        "val": transforms.Compose([
                                    transforms.Resize(224),  
                                    transforms.CenterCrop(224),  
                                    transforms.Grayscale(num_output_channels=3),  
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
                                    # transforms.Normalize((0.5,), (0.5,))])}#灰度图
    
    train_image_file = "/node2_data/fanggang_1/InceptionMamba-main/Medmnist/organcmnist_224/train_images.npy"
    train_label_file = "/node2_data/fanggang_1/InceptionMamba-main/Medmnist/organcmnist_224/train_labels.npy"
    val_image_file = "/node2_data/fanggang_1/InceptionMambaa-main/Medmnist/organcmnist_224/val_images.npy"
    val_label_file = "/node2_data/fanggang_1/InceptionMamba-main/Medmnist/organcmnist_224/val_labels.npy"

    
    
    train_dataset = NpyDataset(image_file=train_image_file,
                               label_file=train_label_file,
                               transform=data_transform["train"])
    val_dataset = NpyDataset(image_file=val_image_file,
                             label_file=val_label_file,
                             transform=data_transform["val"])

    train_num = len(train_dataset)
    val_num = len(val_dataset)

    labels = [label for _, label in train_dataset]
    class_counts = Counter(labels)
    num_classes = len(class_counts)

    print("类别分布：", class_counts)

    batch_size = 128
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=nw)
    validate_loader = DataLoader(val_dataset,
                                 batch_size=batch_size, shuffle=False,
                                 num_workers=nw)
    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    net = InceptionMamba(num_classes=11)

    net.to(device)

    
    loss_function = nn.CrossEntropyLoss()

    
    optimizer = optim.AdamW(net.parameters(), lr=0.001)
    
    
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)

    epochs = 150
    best_acc = 0.0
    # best_auc = 0.0
    score = 0.0
    best_score = 0.0
    model_name = "InceptionMamba_best"
    save_path = './{}Net.pth'.format(model_name)
    train_steps = len(train_loader)

    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

        lr_scheduler.step()

    print(f'best_acc: {best_acc}')
    print('Finished Training')


if __name__ == '__main__':
    main()