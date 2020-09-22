import torch
import torch.nn as nn
import torchvision


class Model(nn.Module):
    def __init__(self, input_dim=(64, 64, 3), n_classes=10):
        super(Model, self).__init__()
        h, w, ch = input_dim
        self.n_classes = n_classes

        self.conv1 = nn.Conv2d(ch, 32, (3, 3), (1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(32, 32, (3, 3), (1, 1), padding=(1, 1))
        self.mp1 = nn.MaxPool2d((2, 2), (2, 2))

        self.conv3 = nn.Conv2d(32, 64, (3, 3), (1, 1), padding=(1, 1))
        self.conv4 = nn.Conv2d(64, 64, (3, 3), (1, 1), padding=(1, 1))
        self.mp2 = nn.MaxPool2d((2, 2), (2, 2))

        self.fc = nn.Linear(64*7*7, n_classes)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(-1)

    def forward(self, inputs):

        x = self.relu(self.conv1(inputs))
        x = self.relu(self.conv2(x))
        x = self.mp1(x)

        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.mp2(x)

        x = x.view(-1, 64*7*7)

        x = self.softmax(self.fc(x))

        return x


def conv_layer(in_ch, out_ch, bn=False):
    return [nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU()]


class VGGModel(nn.Module):
    def __init__(self, input_dim=(64, 64, 3), n_classes=10, pretrained=False, bn=True):
        super(VGGModel, self).__init__()

        if bn:
            vgg = torchvision.models.vgg16_bn(pretrained=pretrained)
        else:
            vgg = torchvision.models.vgg16(pretrained=pretrained)        

        modules = list(vgg.children())

        final_layers = list(modules[-1].children())[:-1] + [nn.Linear(in_features=4096, out_features=n_classes, bias=True)]   

        all_layers = modules[:-1] + [nn.Flatten(), nn.Sequential(*final_layers)] 

        self.vgg = nn.Sequential(*all_layers)
        
    def forward(self, x):
        return self.vgg(x)

        
class VGGModel_old(nn.Module):
    def __init__(self, input_dim=(64, 64, 3), n_classes=10):
        super(VGGModel, self).__init__()
        h, w, ch = input_dim
        self.n_classes = n_classes

        self.conv1 = nn.Sequential(*conv_layer(ch, 64))
        self.conv2 = nn.Sequential(*conv_layer(64, 64))

        self.conv3 = nn.Sequential(*conv_layer(64, 128))
        self.conv4 = nn.Sequential(*conv_layer(128, 128))

        self.conv5 = nn.Sequential(*conv_layer(128, 256))
        self.conv6 =  nn.Sequential(*conv_layer(256, 256))
        self.conv7 = nn.Sequential(*conv_layer(256, 256))
        
        self.conv8 = nn.Sequential(*conv_layer(256, 512))
        self.conv9 = nn.Sequential(*conv_layer(512, 512))
        self.conv10 = nn.Sequential(*conv_layer(512, 512))
        
        self.conv11 = nn.Sequential(*conv_layer(512, 512))
        self.conv12 = nn.Sequential(*conv_layer(512, 512))
        self.conv13 = nn.Sequential(*conv_layer(512, 512))

        self.fc1 = nn.Linear(512*2*2, 4096)
        self.dropout1 = nn.Dropout()
        self.fc2 = nn.Linear(4096, 1000)
        self.dropout2 = nn.Dropout()
        self.fc3 = nn.Linear(1000, n_classes)

        self.max_pool = nn.MaxPool2d((2, 2), (2, 2))
        self.softmax = nn.Softmax(-1)
        self.relu = nn.ReLU()

    def forward(self, inputs):

        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.max_pool(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.max_pool(x)
        
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.max_pool(x)
        
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.max_pool(x)
        
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.max_pool(x)

        x = x.view(-1, 512*2*2)

        x = self.dropout1(self.relu(self.fc1(x)))
        x = self.dropout2(self.relu(self.fc2(x)))
        x = self.softmax(self.fc3(x))

        return x
