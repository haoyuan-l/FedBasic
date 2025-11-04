import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer_names = []
        last_planes = self.in_planes
        strides = [1, 2, 2, 2]
        planes_list = [64, 128, 256, 512]

        for idx, num_block in enumerate(num_blocks):
            if num_block > 0:
                planes = planes_list[idx]
                stride = strides[idx]
                layer = self._make_layer(block, planes, num_block, stride)
                layer_name = f'layer{idx + 1}'
                setattr(self, layer_name, layer)
                self.layer_names.append(layer_name)
                last_planes = planes * block.expansion

        self.linear = nn.Linear(last_planes, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        for layer_name in self.layer_names:
            out = getattr(self, layer_name)(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        # embedding_output = out  # If you need the embedding
        out = self.linear(out)
        return out

def ResNet8(num_classes=10):
    return ResNet(BasicBlock, [1, 1, 1, 1], num_classes=num_classes)

def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def ResNet34(num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def ResNet50(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


def ResNet101(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)


def ResNet152(num_classes=10):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)

class LinearProbe(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(LinearProbe, self).__init__()
        self.fc = nn.Linear(embedding_dim, num_classes)
        self.fc.bias.data.fill_(0)

    def forward(self, x):
        return self.fc(x)
    
""" Helper function to get the network model """

def get_resnet18_network(input_shape=None, num_classes=10, weights_fp=None):
    model = ResNet18(num_classes=num_classes)
    if weights_fp is not None:
        model.load_state_dict(torch.load(weights_fp))
    return model

def get_linear_network(input_shape, num_classes, weights_fp=None):
    """
    Initializes the LinearClassifier model.
    """
    emb_dim = input_shape[0]
    model = LinearProbe(embedding_dim=emb_dim, num_classes=num_classes)
    if weights_fp is not None:
        model.load_state_dict(torch.load(weights_fp))
    return model
