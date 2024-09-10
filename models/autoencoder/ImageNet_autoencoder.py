"""
Implementation of an Autoencoder model for image classification on Imagenet based on VGG16.
Taken from the GitHub repository by Horizon2333 (https://github.com/Horizon2333/imagenet-autoencoder)
"""

import torch
import torch.nn as nn

def initialize_VGGAutoencoder(device:torch.device= 'cpu') -> nn.DataParallel:
    """
    Initializes the VGG Autoencoder model
    :param device: Device where the model is stored
    :return: VGG Autoencoder model
    """

    configs = [2, 2, 3, 3, 3]
    autoencoder_model = VGGAutoencoder(configs)
    autoencoder_model = nn.DataParallel(autoencoder_model).to(device)

    return autoencoder_model

def initialize_pretrained_VGGAutoencoder(device:torch.device= 'cpu') -> nn.DataParallel:
    """
    Initializes pretrained VGG Autoencoder model.
    :param device: Device where the model is stored
    :return: VGG Autoencoder model
    """

    checkpoint = torch.load('models/autoencoder/pretrained/imagenet-vgg16_AE_baseline.pth')

    autoencoder_model = initialize_VGGAutoencoder(device)
    autoencoder_model.load_state_dict(checkpoint['state_dict'])

    return autoencoder_model

class VGGAutoencoder(nn.Module):
    def __init__(self, configs):
        super(VGGAutoencoder, self).__init__()

        # VGG without Bn as Autoencoder is hard to train
        self.encoder = VGGEncoder(configs=configs,       enable_bn=True)
        self.decoder = VGGDecoder(configs=configs[::-1], enable_bn=True)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class VGGEncoder(nn.Module):
    def __init__(self, configs, enable_bn=True):
        super(VGGEncoder, self).__init__()

        if len(configs) != 5:
            raise ValueError('VGGEncoder configs must have 5 layers')

        self.conv1 = EncoderBlock(input_dim=3,   output_dim=64,  hidden_dim=64,  layers=configs[0], enable_bn=enable_bn)
        self.conv2 = EncoderBlock(input_dim=64,  output_dim=128, hidden_dim=128, layers=configs[1], enable_bn=enable_bn)
        self.conv3 = EncoderBlock(input_dim=128, output_dim=256, hidden_dim=256, layers=configs[2], enable_bn=enable_bn)
        self.conv4 = EncoderBlock(input_dim=256, output_dim=512, hidden_dim=512, layers=configs[3], enable_bn=enable_bn)
        self.conv5 = EncoderBlock(input_dim=512, output_dim=512, hidden_dim=512, layers=configs[4], enable_bn=enable_bn)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

class VGGDecoder(nn.Module):
    def __init__(self, configs, enable_bn=True):
        super(VGGDecoder, self).__init__()

        if len(configs) != 5:
            raise ValueError('VGGDecoder configs must have 5 layers')

        self.conv1 = DecoderBlock(input_dim=512, output_dim=512, hidden_dim=512, layers=configs[0], enable_bn=enable_bn)
        self.conv2 = DecoderBlock(input_dim=512, output_dim=256, hidden_dim=512, layers=configs[1], enable_bn=enable_bn)
        self.conv3 = DecoderBlock(input_dim=256, output_dim=128, hidden_dim=256, layers=configs[2], enable_bn=enable_bn)
        self.conv4 = DecoderBlock(input_dim=128, output_dim=64,  hidden_dim=128, layers=configs[3], enable_bn=enable_bn)
        self.conv5 = DecoderBlock(input_dim=64,  output_dim=3,   hidden_dim=64,  layers=configs[4], enable_bn=enable_bn)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.gate(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, layers, enable_bn=True):
        super(EncoderBlock, self).__init__()

        if layers == 1:
            layer = EncoderLayer(input_dim=input_dim, output_dim=output_dim, enable_bn=enable_bn)
            self.add_module('0 EncoderLayer', layer)
        else:
            for i in range(layers):
                if i == 0:
                    layer = EncoderLayer(input_dim=input_dim, output_dim=hidden_dim, enable_bn=enable_bn)
                elif i == (layers - 1):
                    layer = EncoderLayer(input_dim=hidden_dim, output_dim=output_dim, enable_bn=enable_bn)
                else:
                    layer = EncoderLayer(input_dim=hidden_dim, output_dim=hidden_dim, enable_bn=enable_bn)

                self.add_module('%d EncoderLayer' % i, layer)

        maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.add_module('%d MaxPooling' % layers, maxpool)

    def forward(self, x):
        for name, layer in self.named_children():
            x = layer(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, layers, enable_bn=True):
        super(DecoderBlock, self).__init__()

        upsample = nn.ConvTranspose2d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=2, stride=2)
        self.add_module('0 UpSampling', upsample)

        if layers == 1:
            layer = DecoderLayer(input_dim=input_dim, output_dim=output_dim, enable_bn=enable_bn)
            self.add_module('1 DecoderLayer', layer)
        else:
            for i in range(layers):
                if i == 0:
                    layer = DecoderLayer(input_dim=input_dim, output_dim=hidden_dim, enable_bn=enable_bn)
                elif i == (layers - 1):
                    layer = DecoderLayer(input_dim=hidden_dim, output_dim=output_dim, enable_bn=enable_bn)
                else:
                    layer = DecoderLayer(input_dim=hidden_dim, output_dim=hidden_dim, enable_bn=enable_bn)

                self.add_module('%d DecoderLayer' % (i + 1), layer)

    def forward(self, x):
        for name, layer in self.named_children():
            x = layer(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, input_dim, output_dim, enable_bn=True):
        super(EncoderLayer, self).__init__()

        if enable_bn:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(output_dim),
                nn.ReLU(inplace=True),
            )
        else:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.layer(x)

class DecoderLayer(nn.Module):
    def __init__(self, input_dim, output_dim, enable_bn=True):
        super(DecoderLayer, self).__init__()

        if enable_bn:
            self.layer = nn.Sequential(
                nn.BatchNorm2d(input_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1),
            )
        else:
            self.layer = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1),
            )

    def forward(self, x):
        return self.layer(x)
