# -*- coding: utf-8 -*-
#%%
import torch
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

#%%
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 5) # TODO: Increase out channels to more than 80
        self.conv2 = nn.Conv2d(64, 128, 5) # TODO: Change 80 to match ^.
        self.batch1 = nn.BatchNorm2d(64)
        self.batch2 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 5 * 5, 2048) # TODO: Increase out features to more than 220
        self.fc2 = nn.linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 184) # TODO: Change 220 to match ^. Increase out feature to more than 184
        self.fc4 = nn.Linear(184, 10) # TODO: Change 184 to match ^
        
    def forward(self, x):
        x = self.pool(F.relu(self.batch1(self.conv1(x))))
        x = self.pool(F.relu(self.batch2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
net = Net()
#%%

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)
net.to(device)
#%%

import torch.optim as optim

train_transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),transforms.RandomHorizontalFlip(),transforms.RandomCrop(32, padding=4)]) # TODO: Try more transformations to augment data
    
batch_size = 16 # TODO: Experiment with different batch sizes
    
trainset = torchvision.datasets.CIFAR10(root='./data', train = True,
                                            download=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

test_transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # Don't tranform test data
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0) # TODO: Look at ADAM optimizer
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3)
#%%

def train(net):
    
    
    #running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
       
        # print statistics
        #running_loss += loss.item()
        if i % (len(trainloader) // 3) == 0: # i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f lr: %.5f' %
                    (epoch + 1, i + 1, loss.item(), optimizer.param_groups[0]['lr']))
            # running_loss = 0.0
#%%

def test(net):
    correct = 0
    total = 0
    test_loss = 0
    net.eval()
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network 
            outputs = net(images)
            test_loss += criterion(outputs, labels).item()
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_loss /= total
    scheduler.step(test_loss)
    print('Accuracy of the network on the 10000 test images: %d %% \n' % (
        100 * correct / total))
#%%

def class_accuracy(net):
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    
    net.eval()
    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)    
            outputs = net(images)    
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
    
      
    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname, 
                                                       accuracy))
#%%

if __name__ == '__main__':  
    for epoch in range(20):
        train(net)
        test(net)
        
    class_accuracy(net)