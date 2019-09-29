import torch.nn as nn
import math

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, retain_activation=True, activation='ReLU'):
        super(ConvBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        if retain_activation:
            if activation == 'ReLU':
                self.block.add_module("ReLU", nn.ReLU(inplace=True))
            elif activation == 'LeakyReLU':
                self.block.add_module("LeakyReLU", nn.LeakyReLU(0.1))
            elif activation == 'Softplus':
                self.block.add_module("Softplus", nn.Softplus())
        self.block.add_module("MaxPool2d", nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        
    def forward(self, x):
        out = self.block(x)
        return out

# Embedding network used in Matching Networks (Vinyals et al., NIPS 2016), Meta-LSTM (Ravi & Larochelle, ICLR 2017),
# MAML (w/ h_dim=z_dim=32) (Finn et al., ICML 2017), Prototypical Networks (Snell et al. NIPS 2017).

class ProtoNetEmbedding(nn.Module):
    def __init__(self, x_dim=3, h_dim=64, z_dim=64, retain_last_activation=True, activation='ReLU'):
        super(ProtoNetEmbedding, self).__init__()
        self.encoder = nn.Sequential(
          ConvBlock(x_dim, h_dim, activation=activation),
          ConvBlock(h_dim, h_dim, activation=activation),
          ConvBlock(h_dim, h_dim, activation=activation),
          ConvBlock(h_dim, z_dim, retain_activation=retain_last_activation, activation=activation),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)
