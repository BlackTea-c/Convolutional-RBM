from __future__ import print_function
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.utils.data
import torch.optim as optim
import torch.nn as nn


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        image = self.data[index]
        label = self.labels[index]
        return image, label

    def __len__(self):
        return len(self.data)


train_data=torch.load('train_features.pth')
train_labels=torch.load('train_labels.pth')
test_data=torch.load('test_features.pth')
test_labels=torch.load('test_labels.pth')
train_dataset = CustomDataset(train_data, train_labels)

test_dataset=CustomDataset(test_data,test_labels)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader=torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)
#CNN Classifier...
class CNNClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CNNClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义超参数
num_classes = 10
learning_rate = 0.001
num_epochs = 500

# 初始化模型和损失函数
model = CNNClassifier(num_classes).to(device)
criterion = nn.CrossEntropyLoss()


# 训练模型
for epoch in range(num_epochs):
    print('lr:',learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for images, labels in train_loader:
        #print(images.shape)
        images = images.to(device).float()
        labels = labels.to(device)

        # 前向传播
        outputs = model.forward(images)
        loss = criterion(outputs, labels.long())

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Epoch [{}/{}], Loss: {:.4f}".format(epoch+1, num_epochs, loss.item()))
    # 测试模型
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in train_loader:
            # print(images)
            images = images.to(device).float()
            labels = labels.to(device)

            outputs = model.forward(images)

            _, predicted = torch.max(outputs.data, 1)
            # print(predicted,labels)

            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
    accuracy = 100 * total_correct / total_samples
    print("Test Accuracy: {:.2f}%".format(accuracy))
    with torch.no_grad():
        for images, labels in test_loader:
            # print(images)
            images = images.to(device).float()
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            # print(predicted,labels)

            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

    accuracy = 100 * total_correct / total_samples
    print("Test Accuracy: {:.2f}%".format(accuracy))
    if accuracy>=69:

        learning_rate=0.0005






