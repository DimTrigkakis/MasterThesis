import torch
import torch.nn as nn

from torch.autograd import Variable
import math

from collections import OrderedDict
# Alexnet model definition

class AlexNet(nn.Module):
 
    def __init__(self, num_classes=3):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

# Vgg model

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    return nn.Sequential(*layers)


class VGG_viewpoints(nn.Module):

    def __init__(self, num_classes=3, mode = "predictions"):
        super(VGG_viewpoints, self).__init__()

        mytype = 'A'

        self.mode = mode

        self.features = make_layers(cfg[mytype])
        self.classifier = self.classifier_type(num_classes)
        self.soft = nn.LogSoftmax()
        self._initialize_weights()

    def classifier_type(self, num_classes):
        s = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

        return s


    def forward(self, x):
        print(x.size())
        x = self.features(x)
        print(x.size())
        y = x.view(x.size(0),-1)
        print(y.size())
        z = self.classifier(y)

        if self.mode == "features": # this assumes you have popped the last 3 layers of the sequential module in the classifier
            return z
        
        if self.soft != None:
            z = self.soft(z)

        return z

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                
class VGG(nn.Module):

    def __init__(self, num_classes):
        super(VGG, self).__init__()

        mytype = 'A'

        self.features = make_layers(cfg[mytype])
        self.classifier = self.classifier_type(mytype,num_classes)
        self.soft = nn.LogSoftmax()
        self._initialize_weights()

    def classifier_type(self, mytype, num_classes):
        if mytype != '0':
            s = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )
        else:
            s = nn.Sequential(
                nn.Linear(128 * 7 * 7, 512),
                nn.ReLU(True),
                nn.Linear(512, num_classes),
            )

        return s


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        if self.soft != None:
            x = self.soft(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

# Resnet model

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, num_classes=3):
        self.inplanes = 64
        super(ResNet, self).__init__()
        block, layers = BasicBlock, [3, 4, 6, 3]
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class DimoResNet(nn.Module):

    def __init__(self, num_classes=3):
        super(DimoResNet, self).__init__()
        self.inplanes = 64
        self.block, layers = BasicBlock, [3, 4, 6, 3]
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(self.block, 64, layers[0])
        self.layer2 = self._make_layer(self.block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(self.block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(self.block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * self.block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.fc != None:
            x = self.fc(x)

        return x


class ResNet(nn.Module):

    def __init__(self, num_classes=3):
        self.inplanes = 64
        super(ResNet, self).__init__()
        block, layers = BasicBlock, [3, 4, 6, 3]
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

'''
# Dimonet definition

class DimoNet(nn.Module):

    def __init__(self, num_classes=3):
        super(DimoNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            #nn.Linear(2048, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
'''
'''
class DimoRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DimoRNN, self).__init__()
        
        self.hidden_size = hidden_size
        
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax()
    
    def forward(self, input, hidden):

        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        # Needs to be cuda Tensor, otherwise there's a mismatch between input and hidden layers, causing it to be a tuple instead of a sequence
        return Variable(torch.zeros(1, self.hidden_size).cuda())
'''



class DimoLSTM(nn.RNNBase):
    def __init__(self, *args, **kwargs):
        super(DimoLSTM, self).__init__('LSTM', *args, **kwargs)

    def init_hidden(self, batch_size, sequence_length, directions):
        # Needs to be cuda Tensor, otherwise there's a mismatch between input and hidden layers, causing it to be a tuple instead of a sequence
        h0 = Variable(torch.zeros(sequence_length*directions, batch_size, self.hidden_size).cuda())
        c0 = Variable(torch.zeros(sequence_length*directions, batch_size, self.hidden_size).cuda())
        return (h0,c0)

import torch
import torch.nn as nn

from torch.autograd import Variable
import math

# Vgg model


class VGG(nn.Module):

    def __init__(self, num_classes=1000):
        super(VGG, self).__init__()
        self.features = make_layers(cfg['A'])
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class DimoAutoSequence(nn.Module):

    def __init__(self, num_nodes=[64,96], conv_len=5, n_classes=3, n_features=4096, max_len=128,pretrained_model=None,decode=False,num_inner_nodes=16): # convlen 9, num nodes 64 96, n features 4096
        super(DimoAutoSequence, self).__init__()

        self.decode = decode
        self.num_nodes = num_inner_nodes
        self.num_classes = n_classes
        self.max_len = max_len
        self.vgg_type = 'A'
        self.features = make_layers(cfg[self.vgg_type])

        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 32),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(32, 32),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(32, 3),
        )

        if pretrained_model != None:
            state_dict = torch.load(pretrained_model)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            self.load_state_dict(new_state_dict)

        classifier = list(self.classifier.children())
        classifier.pop()
        new_classifier = torch.nn.Sequential(*classifier)
        self.classifier = new_classifier

        # Num_nodes is a list, containing the number of kernels per layer
        num_layers = len(num_nodes)
        in_channels = n_features
        scaling_size = max_len

        layers = []
        for i in range(num_layers):
            conve2d = nn.Conv2d(in_channels,num_nodes[i],kernel_size=(1,conv_len),padding=(0,(conv_len-1)/2))
            drope2d = nn.Dropout(0.3)
            relue2d = nn.ReLU(True)
            maxpoole2d = nn.MaxPool2d((1,2), stride=(1,2))#,padding=(0,1))
            layers += [conve2d, drope2d, relue2d, maxpoole2d]#, maxpoole2d]
            in_channels = num_nodes[i]

        self.encoder = nn.Sequential(*layers)

        if self.decode:
            layers_d = []
            scaling_size = scaling_size/2**num_layers
            scaling_size *= 2

            for j in range(num_layers):
                upsample = nn.UpsamplingBilinear2d(size=(1,scaling_size))
                conv2d = nn.Conv2d(in_channels, num_nodes[-j-1], kernel_size=(1,conv_len),padding=(0,(conv_len-1)/2))
                drop2d = nn.Dropout(0.3)
                relu2d = nn.ReLU(True)
                layers_d += [upsample, conv2d, drop2d, relu2d]
                in_channels = num_nodes[-j-1]
                scaling_size *= 2

            self.decoder = nn.Sequential(*layers_d)

            self.final_classifier = nn.Sequential(
                nn.Linear(in_channels, n_classes*self.num_nodes),
                nn.ReLU(True),
            )

            self.batch_to_classifier = nn.Sequential(
                nn.Linear(self.max_len*n_classes*self.num_nodes,n_classes)
            )
        else:
            self.decoder_to_classifier = nn.Sequential(
                nn.Linear(32*1*self.max_len/4,16),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(16,self.num_classes)
            )

        self.soft = nn.LogSoftmax()

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
    '''
    before batching x worked with 
    torch.Size([32, 3, 224, 224])
    torch.Size([32, 512, 7, 7])
    torch.Size([32, 25088])
    torch.Size([32, 4096])
    torch.Size([32, 1, 4096])
    torch.Size([1, 4096, 32])
    torch.Size([1, 4096, 1, 32])
    torch.Size([1, 96, 1, 8])
    torch.Size([1, 64, 1, 32])
    torch.Size([32, 1, 64, 1])
    torch.Size([32, 64])
    torch.Size([32, 3])
    torch.Size([32, 3])
    '''

    def forward(self, t):
        # We only have one batch per sequence
        # VGG will extract only 16 frames at a time (for any proper length)
        vgg_types = {'A':self.max_len, 'B':self.max_len, 'D':self.max_len, 'E':self.max_len}
        max_vgg_gpu_batch = vgg_types[self.vgg_type]

        #print(t.size(), "t")

        input_shape = t.data.size()
        y = torch.split(t, 1)
        z = torch.cat(y,1)
        w = torch.squeeze(z,0)

        w = self.features(w)
        #print(w.size(), "w")
        w = w.view(w.size(0), -1)
        w = self.classifier(w)
        w = w.unsqueeze(1)

        # input of form B x 1 x F
        # transformed into 1 x F x B where we convolve over B in time, with F channels per position
        #print(x.size())

        # before permuting, split the tensor back to its constituents
        # Tensor of the form (BL) x 1 x 4096
        # We want it to become, L x B x 4096
        #print(w.size(),"w2")
        r = torch.split(w,self.max_len)
        s = torch.cat(r,1)

        # Now we want it to become B x 4096 x L
        x = s.permute(1,2,0)
        #print(x.size(),"x")

        # Now we want it to become B x 4096 x 1 x L, so we can convolve on 1 x L in time
        x = x.unsqueeze(2)
        #print(x.size(),"x2")
        x = self.encoder(x)
        #print(x)
        #print(x.size(),"x3")

        if self.decode:
            x = self.decoder(x)
            x = x.permute(3,0,1,2)
            x = x.squeeze(3)
            f = torch.split(x, 1, 1)
            h = torch.cat(f,0)
            h = h.squeeze(1)
            p = self.final_classifier(h)

            mylength = self.max_len
            if not self.decode:
                mylength = self.max_len/4

            final_splits = torch.split(p,self.max_len,0)
            splits = []
            for split in range(len(final_splits)):
                splits.append(final_splits[split].view(1,self.max_len*self.num_classes*self.num_nodes))
                splits[split] = self.batch_to_classifier(splits[split])

            u = torch.cat(splits,0)

            u = self.soft(u)
        else:
            #print "before",x.size()
            p = x.view(-1,32*1*self.max_len/4)
            #print(p.size(),"p")
            u = self.decoder_to_classifier(p)
            #print(u.size(),"u")
            u = self.soft(u)
            #print "after",u.size()

        return u

        # Next one became this
        input_shape = t.data.size()
        print(input_shape,"input shape") 
        # B, L, C, W, H

        y = torch.split(t,  vgg_types[self.vgg_type] , 1)
        for k in y:
            print(k.size())
        # B, L, C, W, H

        wz_list = []
        for i in range(len(y)):
            y_current = y[i]

            #print y_current.size()
            w = self.features(y_current[0])
            #print(w.size())
            wz = w.view(w.size(0), -1)
            #print(wz.size())
            wz = self.classifier(wz)
            wz_list.append(wz)
            #print(wz.size())
        final_wz = torch.cat(wz_list,0)
        print(final_wz.size())

        w = final_wz.unsqueeze(1)

        # input of form B x 1 x F
        # transformed into 1 x F x B where we convolve over B in time, with F channels per position
        #print(x.size())

        # before permuting, split the tensor back to its constituents
        # Tensor of the form (BL) x 1 x 4096
        # We want it to become, L x B x 4096


        r = torch.split(w,self.max_len)
        s = torch.cat(r,1)

        # Now we want it to become B x 4096 x L
        x = s.permute(1,2,0)
        print(x.size())
        # Now we want it to become B x 4096 x 1 x L, so we can convolve on 1 x L in time
        print(x.size())
        x = x.unsqueeze(2)
        x = self.encoder(x)
        print(x.size())
        if self.decode:
            x = self.decoder(x)
            x = x.permute(3,0,1,2)
            x = x.squeeze(3)
            f = torch.split(x, 1, 1)
            h = torch.cat(f,0)
            h = h.squeeze(1)
            p = self.final_classifier(h)

            mylength = self.max_len
            if not self.decode:
                mylength = self.max_len/4
            final_splits = torch.split(p,self.max_len,0)
            splits = []
            for split in range(len(final_splits)):
                splits.append(final_splits[split].view(1,self.max_len*self.num_classes*self.num_nodes))
                splits[split] = self.batch_to_classifier(splits[split])

            u = torch.cat(splits,0)

            u = self.soft(u)
        else:
            #print "before",x.size()
            print(x.size())
            exit()
            batch_size = x.size()
            p = x.view(-1,96*1*self.max_len/4)
            u = self.decoder_to_classifier(p)
            u = self.soft(u)
            #print "after",u.size()

        return u

class EncoderDecoderViewpoints(nn.Module):

    def classifier_type(self, mytype, num_classes):
            if mytype != '0':
                s = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 4096),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(4096, 4096),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(4096, num_classes),
                )
            else:
                s = nn.Sequential(
                    nn.Linear(128 * 7 * 7, 512),
                    nn.ReLU(True),
                    nn.Linear(512, num_classes),
                )

            return s

    def classifier_type(self, num_classes):
        s = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

        return s

    def init_VGG(self, model=None, num_classes = 3, mode = "predictions"):

        mytype = 'E'

        self.mode = mode
        self.features = make_layers(cfg[mytype])
        self.classifier = self.classifier_type(num_classes)
        self.soft = nn.LogSoftmax()


        print self


        state_dict = torch.load(model)
        current_dict = self.state_dict()
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name=k#name = k[7:] # remove `module.`
            new_state_dict[name] = v

        current_dict.update(new_state_dict)
        self.load_state_dict(current_dict)

    def __init__(self, num_nodes=[64,96], conv_len=7, n_classes=3, n_features=4096, max_len=256,decode=True,num_inner_nodes=1):
        super(EncoderDecoderViewpoints, self).__init__()

        
        self.decode = decode
        self.num_nodes = num_inner_nodes
        self.num_classes = n_classes
        self.max_len = max_len

        # Num_nodes is a list, containing the number of kernels per layer
        num_layers = len(num_nodes)
        in_channels = n_features
        scaling_size = max_len

        layers = []
        for i in range(num_layers):
            conve2d = nn.Conv2d(in_channels,num_nodes[i],kernel_size=(1,conv_len),padding=(0,(conv_len-1)/2))
            drope2d = nn.Dropout(0.3)
            relue2d = nn.ReLU(True)
            maxpoole2d = nn.MaxPool2d((1,2), stride=(1,2))#,padding=(0,1))
            layers += [conve2d, drope2d, relue2d, maxpoole2d]#, maxpoole2d]
            in_channels = num_nodes[i]

        self.encoder = nn.Sequential(*layers)

        if self.decode:
            layers_d = []
            scaling_size = scaling_size/2**num_layers
            scaling_size *= 2

            # possible bug, last layer does not return to Feature size, but first num_nodes size
            for j in range(num_layers):
                upsample = nn.UpsamplingBilinear2d(size=(1,scaling_size))
                conv2d = nn.Conv2d(in_channels, num_nodes[-j-1], kernel_size=(1,conv_len),padding=(0,(conv_len-1)/2))
                drop2d = nn.Dropout(0.3)
                relu2d = nn.ReLU(True)
                layers_d += [upsample, conv2d, drop2d, relu2d]
                in_channels = num_nodes[-j-1]
                scaling_size *= 2

            self.decoder = nn.Sequential(*layers_d)

            self.final_classifier = nn.Sequential(
                nn.Linear(in_channels, n_classes*self.num_nodes),
                nn.ReLU(True),
            )
        else:
            self.decoder_to_classifier = nn.Sequential(
                nn.Linear(96*1*self.max_len/4,16),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(16,n_classes),
            )

        self.soft = nn.LogSoftmax()
        

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
    '''
    before batching x worked with 
    torch.Size([32, 3, 224, 224])
    torch.Size([32, 512, 7, 7])
    torch.Size([32, 25088])
    torch.Size([32, 4096])
    torch.Size([32, 1, 4096])
    torch.Size([1, 4096, 32])
    torch.Size([1, 4096, 1, 32])
    torch.Size([1, 96, 1, 8])
    torch.Size([1, 64, 1, 32])
    torch.Size([32, 1, 64, 1])
    torch.Size([32, 64])
    torch.Size([32, 3])
    torch.Size([32, 3])
    '''

    def forward(self, t):
        # The input should be of the form Batch x Length x Channels(3) x S1 x S2
        # input is from features of VGG

        #x = self.features(t)
        #y = x.view(x.size(0),-1)
        #t3 = self.classifier(y) # returns VGG features

        # for EB only, remove otherwise
        #return t3

        #w = t3.unsqueeze(1)
        ################# EB
        print(t.size())
        w = t.unsqueeze(1)

        # input of form B x 1 x F
        # transformed into 1 x F x B where we convolve over B in time, with F channels per position
        #print(x.size())

        # before permuting, split the tensor back to its constituents
        # Tensor of the form (BL) x 1 x 4096
        # We want it to become, L x B x 4096
        #r = torch.split(w,self.max_len)
        #s = torch.cat(r,1)

        # Now we want it to become B x 4096 x L
        x = w.permute(1,2,0)

        # Now we want it to become B x 4096 x 1 x L, so we can convolve on 1 x L in time
        x = x.unsqueeze(2)
        print(x.size())
        x = self.encoder(x)
        if self.decode:
            x = self.decoder(x)
            x = x.permute(3,0,1,2)
            x = x.squeeze(3)
            f = torch.split(x, 1, 1)
            h = torch.cat(f,0)
            h = h.squeeze(1)
            p = self.final_classifier(h)
            p = self.soft(p)
            return p

            mylength = self.max_len
            if not self.decode:
                mylength = self.max_len/4
            final_splits = torch.split(p,self.max_len,0)
            splits = []
            for split in range(len(final_splits)):
                splits.append(final_splits[split].view(1,self.max_len*self.num_classes*self.num_nodes))
                splits[split] = self.batch_to_classifier(splits[split])

            u = torch.cat(splits,0)

            u = self.soft(u)
        else:
            print(x.size())
            #print "before",x.size()
            p = x.view(-1,96*1*self.max_len/4)
            u = self.decoder_to_classifier(p)
            u = self.soft(u)
            #print "after",u.size()

        return u


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class netG(nn.Module):
    def __init__(self, ngpu, nz, ngf,nc):
        super(netG, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
        self.apply(weights_init)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class netD(nn.Module):
    def __init__(self, ngpu,nz,ndf,nc):
        super(netD, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.apply(weights_init)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


cfg = {
    '0': [16, 'M', 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M'],
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'K': [16, 'M', 16, 'M', 16, 16, 'M', 16, 16, 'M', 16, 16, 'M'],
    'K2': [64, 'M', 64, 'M', 64, 64, 'M', 64, 64, 'M', 64, 64, 'M'],
}

cfg_class = {
    'A' : [4096],
    'D' : [4096],
    'K' : [16],
    'K2' : [64]
}


class TCNPlaytypes(nn.Module):


    def __init__(self, num_nodes=[64,96], conv_len=3, n_classes=3, max_len=128,pretrained_model=None,vgg_type="A", encoder_nodes=-1): # convlen 9, num nodes 64 96, n features 4096
        super(TCNPlaytypes, self).__init__()

        while conv_len**len(num_nodes) <= max_len:
            conv_len += 2

        self.num_classes = n_classes
        self.max_len = max_len
        self.vgg_type = vgg_type
        self.features = make_layers(cfg[self.vgg_type])
        self.num_nodes = num_nodes
        self.n_features = cfg[self.vgg_type][-2]
        self.cl_features = cfg_class[self.vgg_type][0]

        self.classifier = nn.Sequential(
            nn.Linear(self.n_features * 7 * 7, self.cl_features),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(self.cl_features, self.cl_features),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(self.cl_features, 3),
        )


        classifier = list(self.classifier.children())
        classifier.pop()
        classifier.pop()
        classifier.pop()
        new_classifier = torch.nn.Sequential(*classifier)
        self.classifier = new_classifier

        # Num_nodes is a list, containing the number of kernels per layer
        in_channels = self.cl_features
        
        layers = []
        for i in range(len(num_nodes)):
            conve2d = nn.Conv2d(in_channels,num_nodes[i],kernel_size=(1,conv_len),padding=(0,(conv_len-1)/2))
            relue2d = nn.ReLU(True)
            drope2d = nn.Dropout(0.3)
            maxpoole2d = nn.MaxPool2d((1,2), stride=(1,2))#,padding=(0,1))
            layers += [conve2d, drope2d, relue2d, maxpoole2d]
            #layers += [conve2d, relue2d, maxpoole2d]
            in_channels = num_nodes[i]

        self.encoder = nn.Sequential(*layers)

        
        self.decoder_to_classifier = nn.Sequential(
            nn.Linear(num_nodes[-1]*1*self.max_len/(2**len(num_nodes)),encoder_nodes), # 160, /8
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(encoder_nodes,encoder_nodes),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(encoder_nodes,self.num_classes)
        )
            

        self.soft = nn.LogSoftmax()

        self._initialize_weights()

        '''
        # these lines are commented when not training from scratch using a vgg imagenet model
        pretrained_model = "./experiments/pretrained_models/vgg16.pth"
        pretrained_dict = torch.load(pretrained_model)
        model_dict = self.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        self.load_state_dict(model_dict)
        '''

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, t):

        input_shape = t.data.size()
        y = torch.split(t, 1)
        z = torch.cat(y,1)
        w = torch.squeeze(z,0)

        w = self.features(w)
        w = w.view(w.size(0), -1)
        w = self.classifier(w)
        w = w.unsqueeze(1)

        r = torch.split(w,self.max_len)
        s = torch.cat(r,1)

        x = s.permute(1,2,0)

        x = x.unsqueeze(2)
        x = self.encoder(x)
        p = x.view(-1,self.num_nodes[-1]*1*self.max_len/(2**len(self.num_nodes)))
        u = self.decoder_to_classifier(p)
        u = self.soft(u)

        return u


class model_analysis():

    def __init__(self):
        pass

    @staticmethod
    def view_first_weight(self, model):
        print("First weight is"+str(1))
        return 1