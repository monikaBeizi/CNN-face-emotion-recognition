import torch
import torch.nn as nn

VGG16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
VGG19 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']

class VGG(nn.Module):

    def __init__(self, cfg) -> None:
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg)
        self.fc = nn.Linear(512, 7)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3

        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True)]
                in_channels = x
        
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = nn.functional.dropout(out, p=0.5, training=self.training)
        out = self.fc(out)
        return out


class EmotionRecognition:
    '''
    使用VGG模型

    # __init__ 中cfg是指使用VGG16还是VGG19
    例子: 
    cfg = VGG16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

    # train:
    用于训练模型
    trainLoader: pytorch的DataLoader, 用来载入训练集
    num_epochs: epoch 的数量
    load: True or Not, 用于判断类是否加载本地之前训练好的模型参数

    # test:
    用于检测模型的正确率, return accuracy of model
    '''
    def __init__(self, cfg) -> None:
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
        self.model = VGG(cfg).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)

    def train(self, trainLoader, num_epochs, load=True):
        if load == True:
            self.model.load_state_dict(torch.load('emotion.pth'))
            print('load finish')

        else:
            self.model.train()
            for epoch in range(num_epochs):
                running_loss = 0.0
                for i,data in enumerate(trainLoader, 0):
                    images, labels = data[0].to(self.device), data[1].to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.model.forward(images)
                    loss = self.criterion(outputs, labels.reshape(-1,).long())
                    loss.backward()
                    self.optimizer.step()

                    if i % 100 == 0:
                        print(f'epoch: {epoch + 1}/{num_epochs}, step: {i}/{len(trainLoader)}, loss: {loss.item()}')
                self.scheduler.step()

            torch.save(self.model.state_dict(), 'emotion.pth')
            print('finish')

    def evaluate(self, test_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.model.forward(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.reshape(-1,).long().size(0)
                correct += (predicted == labels.reshape(-1,).long()).sum().item()
        accuracy = correct / total
        return accuracy