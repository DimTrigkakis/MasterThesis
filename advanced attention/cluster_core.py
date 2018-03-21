import fnmatch
import os
from PIL import Image
import numpy as np

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import torch

import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader


class DimoCAE(nn.Module):

    def __init__(self, kernel_cores=[32], encoding=16, input_size=[64,64], num_channels=3):
        super(DimoCAE, self).__init__()

        self.kernel_cores = kernel_cores
        self._initialize_weights()
        self.encoding=encoding


        in_channels = num_channels
        layers = []
        scaling_size = input_size[0]
        num_layers = len(kernel_cores)
        for i in range(num_layers):
            conve2d = nn.Conv2d(in_channels,kernel_cores[i],kernel_size=(3,3),padding=(1,1))
            #drope2d = nn.Dropout(0.3)
            #relue2d = nn.ReLU(True)

            maxpoole2d = nn.MaxPool2d(2)#,padding=(0,1))
            layers += [conve2d, maxpoole2d]#, maxpoole2d]
            in_channels = kernel_cores[i]

        self.encoder = nn.Sequential(*layers)

        layers_d = []
        scaling_size = scaling_size/2**num_layers
        scaling_size *= 2

        layers_fc, layers_dc = [], []
        dim = input_size[0]/(2**num_layers)
        self.dim = dim
        layers_fc += [nn.Linear(kernel_cores[-1]*dim*dim,encoding),nn.ReLU()]
        self.fc = nn.Sequential(*layers_fc)
        layers_dc += [nn.Linear(encoding,kernel_cores[-1]*dim*dim,nn.ReLU())]
        self.dc = nn.Sequential(*layers_dc)


        # possible bug, last layer does not return to Feature size, but first kernel_cores size
        for j in range(num_layers):
            upsample = nn.UpsamplingBilinear2d(scaling_size)

            if j != num_layers -1:
                conv2d = nn.Conv2d(in_channels, kernel_cores[-j-2], kernel_size=(3,3),padding=(1,1))
            else:
                conv2d = nn.Conv2d(in_channels, num_channels, kernel_size=(3,3),padding=(1,1))
            #drop2d = nn.Dropout(0.3)
            if j != num_layers -1:
                layers_d += [upsample, conv2d]
            else:
                layers_d += [upsample, conv2d]

            if j != num_layers -1 :
                in_channels = kernel_cores[-j-2]
            scaling_size *= 2

        self.decoder = nn.Sequential(*layers_d)

    def forward(self, x):

        #print x
        input_shape = x.data.size()

        z = self.encoder(x)
        z_view = z.view(x.data.size()[0],-1)
        z_squashed = self.fc(z_view)
        z = self.dc(z_squashed)
        z = z.resize(x.data.size()[0],self.kernel_cores[-1],self.dim,self.dim)

        y = self.decoder(z)
        #y = self.tan()

        return y, z_squashed

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

def mse_loss(inp, target):
    return torch.sum(torch.pow(inp - target,2)) / inp.data.nelement()

t = transforms.Compose([    transforms.Scale((64,64)),  transforms.ToTensor(), transforms.Normalize( ( 0.48075756  ,0.44863049  ,0.36471718), ( 0.28171279  ,0.26927939  ,0.27728584)) ])
t_inverse = transforms.Compose([      transforms.Normalize( ( -0.48075756/0.28171279  ,-0.44863049/0.26927939  ,-0.36471718/0.27728584), ( 1/0.28171279  ,1/0.26927939  ,1/0.27728584)) , transforms.ToPILImage()])
data = ImageFolder(root='./data/data2', transform=t)
loader = DataLoader(data,shuffle=True,batch_size=64)

n_cluster = 3
total_loss = 0
batches = 0.0
epochs = 300
encoding = 64
kernel_cores = [256,128]

dcae = DimoCAE(kernel_cores=kernel_cores,encoding=encoding).cuda()
optimizer = optim.Adam(dcae.parameters(), lr= 0.001)
#dcae = nn.DataParallel(dcae,device_ids=[0,1,2,3]).cuda()

for epoch in range(epochs+1):
    print "epoch : ", epoch

    for x,y in loader:
        x_c = Variable(x).cuda()
        y_c = Variable(y).cuda()
        batches += x_c.size()[1]

        x_re, z = dcae(x_c)
        print z[0].size()
        loss = mse_loss(x_re, x_c).cuda()
        total_loss += loss.data.cpu().numpy()
        if epoch != 0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch > 200:
            optimizer = optim.Adam(dcae.parameters(), lr= 0.0001)
        if epoch > 260:
            optimizer = optim.Adam(dcae.parameters(), lr= 0.00001)
        if epoch % 20 == 0:
            print "Showing image"
            x_c_show = t_inverse(x_c.data[0].cpu()) 
            x_c_show.save("./image/thumbnail"+str(kernel_cores)+str(epochs)+"_"+str(epoch), "JPEG")
            x_re.data[0] = torch.clamp(x_re.data[0],x_re.data[0].min(),x_re.data[0].max())
            x_re_show = t_inverse(x_re.data[0].cpu())
            x_re_show.save("./image/thumbnail_encoding"+str(kernel_cores)+str(epochs)+"_"+str(epoch), "JPEG")


    print total_loss/batches

exit()

im = Image.open(matches[0])

numpy_matches = []
labels = []
load_number = 20
i = 0
for match in matches:
    if i >= load_number:
        break

    if "buddha" in match and i > 10:
        continue

    im = Image.open(match)
    sss = 16
    im = im.resize((sss,sss))
    im=im.convert('L')
    in_data = np.asarray(im, dtype=np.float32)
    in_data = np.multiply(in_data,1/255.0)
    in_data = np.add(in_data, -0.5)
    if in_data.shape[0] != sss or in_data.shape[1] != sss:
        continue

    numpy_matches.append(in_data)
    labels.append(match)
    i += 1

npm = np.asarray(numpy_matches)
n_samples, n_features = load_number, sss*sss*1
sample_size = load_number

# Can we createa an autoencoder????

dcae = DimoCAE()
optimizer = optim.Adam(dcae.parameters(), lr= 0.001)

epochs = 100
print "training for epochs",epochs
losses = 0.0
for e in range(1):
    for i in range(len(npm)):

        n = torch.from_numpy(npm[i])
        s = Variable(n)
        s = s.unsqueeze(0)
        s = s.unsqueeze(0)
        s2, z = dcae(s)
        loss = mse_loss(s, s2)
        losses += loss.data.numpy()
    print losses/len(npm)

new_numpy = []
for i in range(len(npm)):

    n = torch.from_numpy(npm[i])
    s = Variable(n)
    s = s.unsqueeze(0)
    s = s.unsqueeze(0)
    s2, z = dcae(s)
    new_numpy.append(z.data.numpy())
    if i == 0:
        print z.size()

new_numpy = np.asarray(new_numpy)
print new_numpy.shape
estimator = KMeans(init='k-means++', n_clusters=3, n_init=10)
print "fitting"
estimator.fit(new_numpy)
print estimator.labels_

print len(npm), "length of dataset"

i = 0
conf = [[0,0,0],[0,0,0],[0,0,0]]

def label_to_int(s):
    if "buddha" in s:
        return 0
    if "bonsai" in s:
        return 1
    return 2

for label in estimator.labels_:
    conf[label_to_int(labels[i])][label] += 1

    i += 1
print conf
print "fit"

for e in range(epochs):
    print "epoch",e
    losses = 0.0
    for i in range(len(npm)):

        n = torch.from_numpy(npm[i])
        s = Variable(n)
        s = s.unsqueeze(0)
        s = s.unsqueeze(0)
        s2, z = dcae(s)
        loss = mse_loss(s, s2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses += loss.data.numpy()

    print losses/len(npm)

new_numpy = []
for i in range(len(npm)):

    n = torch.from_numpy(npm[i])
    s = Variable(n)
    s = s.unsqueeze(0)
    s = s.unsqueeze(0)
    s2, z = dcae(s)
    new_numpy.append(z.data.numpy())
    print z

new_numpy = np.asarray(new_numpy)
print new_numpy.shape
estimator = KMeans(init='random', n_clusters=3, n_init=10)
print "fitting"
estimator.fit(new_numpy)
print estimator.labels_

print len(npm), "length of dataset"


i = 0
conf = [[0,0,0],[0,0,0],[0,0,0]]

def label_to_int(s):
    if "buddha" in s:
        return 0
    if "bonsai" in s:
        return 1
    return 2

for label in estimator.labels_:
    conf[label_to_int(labels[i])][label] += 1

    i += 1
print conf
print "fit"

estimator = KMeans(init='random', n_clusters=2, n_init=10)
print "fitting"
estimator.fit(new_numpy)
print estimator.labels_

print len(npm), "length of dataset"


i = 0
conf = [[0,0,0],[0,0,0],[0,0,0]]

def label_to_int(s):
    if "buddha" in s:
        return 0
    if "bonsai" in s:
        return 1
    return 2

for label in estimator.labels_:
    conf[label_to_int(labels[i])][label] += 1

    i += 1
print conf
print "fit"
