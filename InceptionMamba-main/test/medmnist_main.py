import sys
import os
sys.path.append(os.path.dirname((os.path.realpath(__file__))))
sys.path.append('/node2_data/fanggang_1/InceptionMamba-main')
print(sys.path)
import os
import json

import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.optim as optim
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from torch.utils.data import ConcatDataset
from sklearn.metrics import roc_auc_score  
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from thop import profile
# from model import MobileNetV2

from InceptionMamba import VSSM as InceptionMamba # import model

class ConfusionMatrix(object):

    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity", "F1"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            F1 = round(2 * (Precision * Recall) / (Precision + Recall), 3) if Precision + Recall != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, Specificity,F1])
        print(table)

    def plot(self):
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        plt.xticks(range(self.num_classes), self.labels, rotation=45)

        plt.yticks(range(self.num_classes), self.labels)

        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()

class NpyDataset(Dataset):
    def __init__(self, image_file, label_file, transform=None):
        
        self.images = np.load(image_file)
        self.labels = np.load(label_file)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        image = self.images[idx]
        label = self.labels[idx]

        
        if isinstance(label, np.ndarray):
            label = label.item()  

        
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        
        if self.transform:
            image = self.transform(image)

        return image, label


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_flops(model, input_shape):
    input_tensor = torch.randn(*input_shape).to(device)
    flops, params = profile(model, inputs=(input_tensor,), verbose=False)
    return flops / 1e9  

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    data_transform = transforms.Compose([transforms.Resize(224),  
                                        transforms.CenterCrop(224),  
                                         transforms.Grayscale(num_output_channels=3),  
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    
    test_data_path = '/node2_data/fanggang_1/InceptionMamba-main/Medmnist/organcmnist_224/test_images.npy'  # 请替换为实际的测试数据路径
    test_label_path = '/node2_data/fanggang_1/InceptionMamba-main/Medmnist/organcmnist_224/test_labels.npy'  # 请替换为实际的标签路径

    batch_size = 128

    
    test_dataset = NpyDataset(image_file=test_data_path, label_file=test_label_path, transform=data_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)


    unique_labels = np.unique(test_dataset.labels)  

    
    labels = list(range(len(unique_labels)))  

    net = InceptionMamba(num_classes=len(labels))
    # load pretrain weights
    model_weight_path = "/node2_data/fanggang_1/InceptionMamba-main/InceptionMamba_bestNet.pth"
    assert os.path.exists(model_weight_path), "cannot find {} file".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    net.to(device)

    # 在模型定义后调用
    print(f"🔹 模型参数量: {count_parameters(net) / 1e6:.2f} M")
    print(f"🔹 模型 FLOPs: {compute_flops(net, (1, 3, 224, 224)):.2f} G")

    # # read class_indict
    # json_label_path = '/node2_data/fanggang_1/InceptionMamba-main/class_indices.json'
    # assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    # json_file = open(json_label_path, 'r')
    # class_indict = json.load(json_file)

    # labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=len(labels), labels=labels)
    net.eval()

    all_labels = []  
    all_outputs = []  

    with torch.no_grad():
        for test_data in tqdm(test_loader):
            test_images, test_labels = test_data
            outputs = net(test_images.to(device))
            softmax_outputs = torch.softmax(outputs, dim=1)
            all_labels.append(test_labels.numpy())
            all_outputs.append(softmax_outputs.cpu().numpy())

            confusion.update(torch.argmax(softmax_outputs, dim=1).to("cpu").numpy(), test_labels.to("cpu").numpy())


    
    all_labels = np.concatenate(all_labels)
    all_outputs = np.concatenate(all_outputs)

    # 计算AUC（对于多分类任务）
    auc = roc_auc_score(all_labels, all_outputs, average='macro', multi_class='ovr')
    print(f"AUC: {auc:.4f}")  # 输出 AUC 结果

    # 计算AUC（对于二分类任务）
    # auc = roc_auc_score(all_labels, all_outputs[:, 1], average='macro', multi_class='ovr')
    # print(f"AUC: {auc:.4f}")

    confusion.plot()
    confusion.summary()
