import torch.nn as nn
import torch.nn.functional as F

class CIFAR10_Autoencoder(nn.Module):
    def __init__(self):
        super(CIFAR10_Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(12, 24, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 48, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(48, 24, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, input):
        encoded = self.encoder(input)
        decoded = self.decoder(encoded)
        return decoded

class CIFAR10_Autoencoder_MaxPooling(nn.Module):
    def __init__(self):
        super(CIFAR10_Autoencoder_MaxPooling, self).__init__()

        self.encoder = MaxPoolingEncoder()
        self.decoder = MaxPoolingDecoder()

    def forward(self, input):
        encoded, ind1, ind2, ind3 = self.encoder(input)
        decoded = self.decoder(encoded, ind1, ind2, ind3)
        return decoded

class MaxPoolingEncoder(nn.Module):
    def __init__(self):
        super(MaxPoolingEncoder, self).__init__()

        self.en_conv11 = nn.Conv2d(3, 12,  kernel_size=3, stride=1, padding=1)
        self.en_bn11 = nn.BatchNorm2d(12, momentum=0.1)
        self.en_conv12 = nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1)
        self.en_bn12 = nn.BatchNorm2d(12, momentum=0.1)

        self.en_conv21 = nn.Conv2d(12, 24, kernel_size=3, stride=1, padding=1)
        self.en_bn21 = nn.BatchNorm2d(24, momentum=0.1)
        self.en_conv22 = nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1)
        self.en_bn22 = nn.BatchNorm2d(24, momentum=0.1)

        self.en_conv31 = nn.Conv2d(24, 48, kernel_size=3, stride=1, padding=1)
        self.en_bn31 = nn.BatchNorm2d(48, momentum=0.1)
        self.en_conv32 = nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1)
        self.en_bn32 = nn.BatchNorm2d(48, momentum=0.1)

    def forward(self, x):
        x = F.relu(self.en_bn11(self.en_conv11(x)), inplace=True)
        x = F.relu(self.en_bn12(self.en_conv12(x)), inplace=True)
        x, ind1 = F.max_pool2d(x, 2, 2, return_indices=True)

        x = F.relu(self.en_bn21(self.en_conv21(x)), inplace=True)
        x = F.relu(self.en_bn22(self.en_conv22(x)), inplace=True)
        x, ind2 = F.max_pool2d(x, 2, 2, return_indices=True)

        x = F.relu(self.en_bn31(self.en_conv31(x)), inplace=True)
        x = F.relu(self.en_bn32(self.en_conv32(x)), inplace=True)
        x, ind3 = F.max_pool2d(x, 2, 2, return_indices=True)

        return x, ind1, ind2, ind3

class MaxPoolingDecoder(nn.Module):
    def __init__(self):
        super(MaxPoolingDecoder, self).__init__()

        self.de_conv11 = nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1)
        self.de_bn11 = nn.BatchNorm2d(48, momentum=0.1)
        self.de_conv12 = nn.Conv2d(48, 24, kernel_size=3, stride=1, padding=1)
        self.de_bn12 = nn.BatchNorm2d(24, momentum=0.1)

        self.de_conv21 = nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1)
        self.de_bn21 = nn.BatchNorm2d(24, momentum=0.1)
        self.de_conv22 = nn.Conv2d(24, 12, kernel_size=3, stride=1, padding=1)
        self.de_bn22 = nn.BatchNorm2d(12, momentum=0.1)

        self.de_conv31 = nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1)
        self.de_bn31 = nn.BatchNorm2d(12, momentum=0.1)
        self.de_conv32 = nn.Conv2d(12, 3,  kernel_size=3, stride=1, padding=1)

    def forward(self, x, ind1, ind2, ind3):
        x = F.max_unpool2d(x, ind3, 2, 2)
        x = F.relu(self.de_bn11(self.de_conv11(x)), inplace=True)
        x = F.relu(self.de_bn12(self.de_conv12(x)), inplace=True)

        x = F.max_unpool2d(x, ind2, 2, 2)
        x = F.relu(self.de_bn21(self.de_conv21(x)), inplace=True)
        x = F.relu(self.de_bn22(self.de_conv22(x)), inplace=True)

        x = F.max_unpool2d(x, ind1, 2, 2)
        x = F.relu(self.de_bn31(self.de_conv31(x)), inplace=True)

        x = F.tanh(self.de_conv32(x))
        return x
