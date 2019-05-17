## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        # Convolution layer 1: input = 1 output = 32, kernel = 5x5
        # input-size of Image = (1,224,224)
        # output size = (W-F)/S +1 = (224-5)/1 +1 = (32,220,220)
        # after one pool shape = (32,110,110)
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv1_bn = nn.BatchNorm2d(32)

        self.pool = nn.MaxPool2d(2, 2)

        # Convolution layer 2: input = 32 output = 64, kernel = 5x5
        # output size = (W-F)/S +1 = (110-5)/1 +1 = (32,106,106)
        # after one pool shape = (32,53,53)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv2_bn = nn.BatchNorm2d(64)
        
        # Convolution layer 3: input = 64 output = 128, kernel = 5x5
        ## output size = (W-F)/S +1 = (53-5)/1 +1 = (128,49,49)
        # after one pool shape = (128,24,24)
        self.conv3 = nn.Conv2d(64,128, 5)
        self.conv3_bn = nn.BatchNorm2d(128)
        
        # Convolution layer 4: input = 128 output = 256, kernel = 5x5
        ## output size = (W-F)/S +1 = (24-5)/1 +1 = (256,20,20)
        # after one pool shape = (256,10,10)
        self.conv4 = nn.Conv2d(128,256, 5)
        self.conv4_bn = nn.BatchNorm2d(256)
        
        # Convolution layer 5: input = 256 output = 512, kernel = 5x5
        ## output size = (W-F)/S +1 = (10-5)/1 +1 = (512,6,6)
        # after one pool shape = (512,3,3)
        self.conv5 = nn.Conv2d(256,512, 5)
        self.conv5_bn = nn.BatchNorm2d(512)

        # Fully connected layer 1 : 512 outputs * the 3*3 filtered/pooled map size
        self.fc1 = nn.Linear(512* 3* 3, 4096)
        self.fc1_bn = nn.BatchNorm1d(4096)
        
        self.drop1 = nn.Dropout(p=0.1)
        self.drop2 = nn.Dropout(p=0.2)
        self.drop3 = nn.Dropout(p=0.3)
        self.drop4 = nn.Dropout(p=0.4)
        self.drop5 = nn.Dropout(p=0.5)
        
        self.fc_drop = nn.Dropout(p=0.3)
        
        # Fully connected layer 2 
        self.fc2 = nn.Linear(4096,2048)
        self.fc2_bn = nn.BatchNorm1d(2048)

        # Fully connected layer 3
        self.fc3 = nn.Linear(2048, 1024)
        self.fc3_bn = nn.BatchNorm1d(1024)
        
        # Fully connected layer 4
        self.fc4 = nn.Linear(1024,512)
        self.fc4_bn = nn.BatchNorm1d(512)
        
         # Fully connected layer 5
        self.fc5 = nn.Linear(512,256)
        self.fc5_bn = nn.BatchNorm1d(256)

        # Output layer
        self.out = nn.Linear(256, 136)

        
    def forward(self, x):
        
        # first convol layer
        x = self.drop1(self.conv1_bn(self.pool(F.relu(self.conv1(x)))))

        # second convol layer
        x = self.drop2(self.conv2_bn(self.pool(F.relu(self.conv2(x)))))

        # third convol layer
        x = self.drop3(self.conv3_bn(self.pool(F.relu(self.conv3(x)))))
        
        # forth convol layer
        x = self.drop4(self.conv4_bn(self.pool(F.relu(self.conv4(x)))))
        
        # fifth convol layer
        x = self.drop5(self.conv5_bn(self.pool(F.relu(self.conv5(x)))))

        # flatten
        x = x.view(x.size(0), -1)

        # first fc layer
        x = self.fc1_bn(F.relu(self.fc1(x)))
        x= self.fc_drop(x)

        # second fc layer
        x = self.fc2_bn(F.relu(self.fc2(x)))
        x = self.fc_drop(x)

        # third fc layer
        x = self.fc3_bn(F.relu(self.fc3(x)))
        x = self.fc_drop(x)
        
        # forth fc layer
        x = self.fc4_bn(F.relu(self.fc4(x)))
        x = self.fc_drop(x)
        
         # third fc layer
        x = self.fc5_bn(F.relu(self.fc5(x)))
        x = self.fc_drop(x)

        # output layer
        x = F.relu(self.out(x))

        return x
       
