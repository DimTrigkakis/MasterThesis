import torch
import torch.nn as nn

from torch.autograd import Variable
import math

from collections import OrderedDict

class ConvVAE(nn.Module):

    def cpu_mode_sampling(self, cpu_mode=True):
        self.cpu_mode = cpu_mode
        
    def sample_z(self, mu, log_var):

        # Using reparameterization trick to sample from a gaussian
        if self.inference:
            return mu
        if self.cpu_mode:
            eps = Variable(torch.randn(mu.size()[0], self.z_spatial_dim))
        else:
            eps = Variable(torch.randn(mu.size()[0], self.z_spatial_dim)).cuda()
        return mu + torch.exp(log_var / 2) * eps
    
    def make_layers_encoder(self, cfg):
        layers = []
        in_channels = 1
        
        i = 0
        
        for v in cfg:
            r = self.kernel_multipliers[i]
            if v == 'CM':
                conv2d = nn.Conv2d(in_channels, r, kernel_size=3,stride=2,padding=1)
                if self.batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(r), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = r
                i += 1
            elif v == 'C':
                conv2d = nn.Conv2d(in_channels, r, kernel_size=3,stride=1, padding=1)
                if self.batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(r), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = r
                i += 1
                
        return nn.Sequential(*layers)
    
    def make_layers_decoder(self, cfg):
        
        layers = []
        in_channels = self.kernel_multipliers[self.last_conv]
        
        i = 0
        
        for v in cfg:
            
            r =  self.kernel_multipliers[self.last_conv-i-1]
            if self.last_conv-i-1 < 0:
                r = 1
                
            if v == 'CM': 
                conv2d = nn.ConvTranspose2d(in_channels, r, kernel_size=2, stride=2, padding=0, output_padding=0)
                
                layers += [nn.ReLU(inplace=True), conv2d]
                in_channels = r
                i += 1
            elif v == 'C':
                conv2d = nn.Conv2d(in_channels,r, kernel_size=3,stride=1, padding=1)
                
                layers += [nn.ReLU(inplace=True), conv2d]
                in_channels = r
                i += 1
                
        return nn.Sequential(*layers)

    def __init__(self,cfg_encoder=None, cfg_decoder=None, linear_nodes=None, kernel_base = None, z_spatial_dim=None, sizes=None, batch_norm = None, kernel_multipliers = None):
        super(ConvVAE, self).__init__()
        self.cpu_mode = False
        self.inference = False
        self.z_spatial_dim = z_spatial_dim
        self.kernel_base = kernel_base
        self.linear_nodes = linear_nodes
        self.sizes = sizes

        self.max_pools = len([d for d in cfg_encoder if d == 'CM'])
        self.kernel_multipliers = [x*self.kernel_base for x in kernel_multipliers]
        self.batch_norm = batch_norm
        self.last_conv = len(cfg_encoder)-1
        
        self.encoder_spatial_cnn = self.make_layers_encoder(cfg=cfg_encoder)
        self.encoder_spatial_linear = nn.Sequential(

            nn.Linear(int(self.kernel_multipliers[self.last_conv]*(self.sizes[0]/(2**self.max_pools))**2), self.linear_nodes),
            nn.ReLU(),
            nn.Linear(self.linear_nodes, self.linear_nodes),
            nn.ReLU(),
            nn.Linear(self.linear_nodes, self.linear_nodes),
            nn.ReLU(),
        )


        self.z_mu = nn.Linear(int(self.linear_nodes),z_spatial_dim) # mean of Z
        self.z_var = nn.Linear(int(self.linear_nodes),z_spatial_dim) # Log variance s^2 of Z
        self.decode_z = nn.Linear(z_spatial_dim, int(self.linear_nodes))

        self.decoder_spatial_linear = nn.Sequential(

            nn.ReLU(),
            nn.Linear(int(self.linear_nodes), self.linear_nodes),
            nn.ReLU(),
            nn.Linear(self.linear_nodes, self.linear_nodes),
            nn.ReLU(),
            nn.Linear(self.linear_nodes, int(kernel_base *kernel_multipliers[self.last_conv] * (self.sizes[0]/(2**self.max_pools))**2)),
        )

        self.decoder_spatial_cnn = self.make_layers_decoder(cfg=cfg_decoder)

        self.sigmoid = nn.Sigmoid()
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
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

    def forward(self, myinput):

        e_l = self.encoder_spatial_cnn(myinput)
        e_l = e_l.view(myinput.size()[0],-1)
        e_l = self.encoder_spatial_linear(e_l)
        
        z_mu = self.z_mu(e_l)
        z_var = self.z_var(e_l)
        Z_sample = self.sample_z(z_mu, z_var)
        
        Z = self.decode_z(Z_sample)
        Z = self.decoder_spatial_linear(Z)
        Z = Z.view(myinput.size()[0],self.kernel_multipliers[self.last_conv],int(self.sizes[0]/(2**self.max_pools)),int(self.sizes[1]/(2**self.max_pools)))
        Z = self.decoder_spatial_cnn(Z)
        
        output = self.sigmoid(Z)

        return Z_sample, z_mu, z_var, output

    def decode(self, Z_sample):
        Z = self.decode_z(Z_sample)
        Z = self.decoder_spatial_linear(Z)
        Z = Z.view(Z_sample.size()[0],self.kernel_multipliers[self.last_conv],int(self.sizes[0]/(2**self.max_pools)),int(self.sizes[1]/(2**self.max_pools)))
        
        Z = self.decoder_spatial_cnn(Z)
        output = self.sigmoid(Z)

        return output

########################################### VAE model

class VAE(nn.Module):

    def cpu_mode_sampling(self,cpu_mode=True):
        self.cpu_mode = cpu_mode
        
    def sample_z(self, mu, log_var):

        if self.inference:
            return mu
        if not self.cpu_mode:
            eps = Variable(torch.randn(mu.size()[0], self.z_spatial_dim)).cuda()
        else:
            eps = Variable(torch.randn(mu.size()[0], self.z_spatial_dim))
        return mu + torch.exp(log_var / 2) * eps

    def __init__(self, z_spatial_dim=None, linear_nodes=None, sizes=None):
        super(VAE, self).__init__()

        self.inference = False
        self.z_spatial_dim = z_spatial_dim
        self.linear_nodes = linear_nodes
        self.sizes = sizes
        self.cpu_mode = False

        self.simplicity = nn.Sequential(
            nn.Linear(self.sizes[0]*self.sizes[1],self.linear_nodes),
            nn.ReLU(),
            nn.Linear(self.linear_nodes,self.linear_nodes),
            nn.ReLU(),
            nn.Linear(self.linear_nodes,self.linear_nodes),
            nn.ReLU()
        )

        self.z_mu = nn.Linear(self.linear_nodes,z_spatial_dim) # mean of Z
        self.z_var = nn.Linear(self.linear_nodes,z_spatial_dim) # Log variance s^2 of Z

        self.decode_z = nn.Linear(z_spatial_dim, self.linear_nodes)
        self.deplicity = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.linear_nodes,self.linear_nodes),
            nn.ReLU(),
            nn.Linear(self.linear_nodes,self.linear_nodes),
            nn.ReLU(),
            nn.Linear(self.linear_nodes,self.sizes[0]*self.sizes[1])
        )

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def decode(self, Z_sample):
        Z = self.decode_z(Z_sample)
        Z = self.deplicity(Z)
        Z = Z.view(Z_sample.size()[0],1,self.sizes[0],self.sizes[1])
        output = self.sigmoid(Z)
        return output
        
    def forward(self, myinput):

        e_l = myinput.view(myinput.size()[0],-1)
        e_l = self.simplicity(e_l)
        z_mu = self.z_mu(e_l)
        z_var = self.z_var(e_l)

        Z_sample = self.sample_z(z_mu, z_var) 
        Z = self.decode_z(Z_sample)
        Z = self.deplicity(Z)
        Z = Z.view(myinput.size()[0],1,self.sizes[0],self.sizes[1])
        output = self.sigmoid(Z)

        return Z_sample, z_mu, z_var, output
    
class CVAE(nn.Module):

    def cpu_mode_sampling(self, cpu_mode=True):
        self.cpu_mode = cpu_mode
    def sample_z(self, mu, log_var):

        if self.inference:
            return mu
        if self.cpu_mode:       
            eps = Variable(torch.randn(mu.size()[0], self.z_spatial_dim))
        else:
            eps = Variable(torch.randn(mu.size()[0], self.z_spatial_dim)).cuda()
        return mu + torch.exp(log_var / 2) * eps

    def __init__(self, z_spatial_dim=None, linear_nodes=None, sizes=None):
        super(CVAE, self).__init__()

        self.inference = False
        self.z_spatial_dim = z_spatial_dim
        self.linear_nodes = linear_nodes
        self.sizes = sizes
        self.cpu_mode = False

        self.simplicity = nn.Sequential(
            nn.Linear(self.sizes*2,self.linear_nodes),
            nn.ReLU(),
            nn.Linear(self.linear_nodes,self.linear_nodes),
            nn.ReLU(),
            nn.Linear(self.linear_nodes,self.linear_nodes),
            nn.ReLU()
        )

        self.z_mu = nn.Linear(self.linear_nodes,z_spatial_dim) # mean of Z
        self.z_var = nn.Linear(self.linear_nodes,z_spatial_dim) # Log variance s^2 of Z

        self.decode_z = nn.Linear(z_spatial_dim+self.sizes, self.linear_nodes)
        self.deplicity = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.linear_nodes,self.linear_nodes),
            nn.ReLU(),
            nn.Linear(self.linear_nodes,self.linear_nodes),
            nn.ReLU(),
            nn.Linear(self.linear_nodes,self.sizes)
        )

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def decode(self, Z_sample, myconditional):
        Z_concat = torch.cat([Z_sample, myconditional],1) # sample concatenation
        Z = self.decode_z(Z_concat)
        Zfinal = self.deplicity(Z)
        Zbar = Zfinal.view(Z_sample.size()[0],self.sizes)
        
        return Zbar
        
    def forward(self, myconditional, myconcat):

        e_l = myconcat.view(-1,self.sizes*2)
        e_l = self.simplicity(e_l)
        z_mu = self.z_mu(e_l)
        z_var = self.z_var(e_l)

        Z_sample = self.sample_z(z_mu, z_var)
        Z_cat = torch.cat([Z_sample, myconditional],1) # sample concatenation
        
        Z = self.decode_z(Z_cat)
        Z = self.deplicity(Z)
        Zbar = Z.view(-1,self.sizes)

        return Z_sample, z_mu, z_var, Zbar

class W2V(nn.Module):


    def __init__(self, dict_size=6, hidden_features=2):
        super(W2V, self).__init__()

        self.relu = nn.ReLU()
        self.embedding = nn.Linear(dict_size,hidden_features)
        self.out = nn.Linear(hidden_features, dict_size)

        self.sigmoid = nn.Sigmoid()
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, input):

        fv = self.embedding(input)
        #fv = self.relu(fv)
        out = self.out(fv)
        #out = self.relu(out)
        # if one-hot encoding, then softmax over codewords
        output = self.sigmoid(out)

        return fv, output
   

class Analogy(nn.Module):

    def __init__(self, dict_size=6, embedded_features=8, hidden_features=8):
        super(Analogy, self).__init__()

        self.embedding = nn.Linear(dict_size,embedded_features)
        self.encoding = nn.Linear(embedded_features, hidden_features)
        self.decoding = nn.Linear(hidden_features, embedded_features)
        self.relu = nn.LeakyReLU(0.2)
        self._initialize_weights()
        
    def embed(self, word):
        out_fe = self.embedding(word)
        out_fe = self.relu(out_fe)
        return out_fe

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, input):

        
        hidden = self.relu(self.encoding(input))
        out = self.relu(self.decoding(hidden))

        return hidden, out
