import torch
import torch.nn as nn
import torch.nn.functional as F

# --- CORE GATING MECHANISM ---
class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(GatedConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.gate_logits = nn.Parameter(torch.ones(out_channels) * 2.0) # Start open

    def forward(self, x, temperature=1.0):
        out = self.conv(x)
        out = self.bn(out)
        gates = torch.sigmoid(self.gate_logits / temperature)
        gates_reshaped = gates.view(1, -1, 1, 1)
        return out * gates_reshaped, gates

# --- VGG FAMILY ---
class VGG_Base(nn.Module):
    def __init__(self, features, num_classes):
        super(VGG_Base, self).__init__()
        self.features = features
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x, temperature=1.0):
        out = x
        all_gates = []
        for layer in self.features:
            if isinstance(layer, GatedConv2d):
                out, g = layer(out, temperature)
                all_gates.append(g)
            else:
                out = layer(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out, all_gates

def make_vgg_layers(cfg, gated=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            if gated:
                conv = GatedConv2d(in_channels, v, kernel_size=3, padding=1)
            else:
                conv = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                conv = nn.Sequential(conv, nn.BatchNorm2d(v))
            
            layers += [conv, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers) if not gated else nn.ModuleList(layers)

# VGG Configurations
VGG11_CFG = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
VGG13_CFG = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

def VGG11_Gated(num_classes): return VGG_Base(make_vgg_layers(VGG11_CFG, True), num_classes)
def VGG11_Baseline(num_classes): return VGG_Base(make_vgg_layers(VGG11_CFG, False), num_classes)
def VGG13_Gated(num_classes): return VGG_Base(make_vgg_layers(VGG13_CFG, True), num_classes)
def VGG13_Baseline(num_classes): return VGG_Base(make_vgg_layers(VGG13_CFG, False), num_classes)

# --- RESNET FAMILY ---
class GatedBasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(GatedBasicBlock, self).__init__()
        self.conv1 = GatedConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = GatedConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x, temperature):
        out, g1 = self.conv1(x, temperature)
        out = F.relu(out)
        out, g2 = self.conv2(out, temperature)
        out += self.shortcut(x)
        out = F.relu(out)
        return out, [g1, g2]

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
    def forward(self, x, t=None): # t arg for compatibility
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, gated=False):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.gated = gated
        
        if gated:
            self.conv1 = GatedConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        layers = []
        layers.append(block(self.in_planes, planes, stride))
        self.in_planes = planes
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes))
        return nn.ModuleList(layers) if self.gated else nn.Sequential(*layers)

    def forward(self, x, temperature=1.0):
        all_gates = []
        
        if self.gated:
            out, g = self.conv1(x, temperature)
            all_gates.append(g)
            out = F.relu(out)
            for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
                for block in layer:
                    out, gates = block(out, temperature)
                    all_gates.extend(gates)
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, all_gates

def ResNet18_Gated(nc): return ResNet(GatedBasicBlock, [2,2,2,2], nc, True)
def ResNet18_Baseline(nc): return ResNet(BasicBlock, [2,2,2,2], nc, False)
def ResNet34_Gated(nc): return ResNet(GatedBasicBlock, [3,4,6,3], nc, True)
def ResNet34_Baseline(nc): return ResNet(BasicBlock, [3,4,6,3], nc, False)