import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): 
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv3 = DoubleConv(128, 64)
        
        self.outc = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv1(x)
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv2(x)
        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv3(x)
        logits = self.outc(x)
        return torch.sigmoid(logits)


class ResNetUNetDecoder(nn.Module):
    """UNet decoder that accepts encoder feature channels from shallow to deep."""

    def __init__(self, encoder_channels, out_channels=1):
        super().__init__()
        shallow, layer1, layer2, layer3, deep = encoder_channels
        self.decode3 = DoubleConv(deep + layer3, layer3)
        self.decode2 = DoubleConv(layer3 + layer2, layer2)
        self.decode1 = DoubleConv(layer2 + layer1, layer1)
        self.decode0 = DoubleConv(layer1 + shallow, shallow)
        self.output = nn.Sequential(
            nn.Conv2d(shallow, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1),
        )

    @staticmethod
    def _upsample_to(source, target):
        return F.interpolate(
            source,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, features, output_size):
        shallow, layer1, layer2, layer3, deep = features
        x = self.decode3(
            torch.cat((self._upsample_to(deep, layer3), layer3), dim=1)
        )
        x = self.decode2(
            torch.cat((self._upsample_to(x, layer2), layer2), dim=1)
        )
        x = self.decode1(
            torch.cat((self._upsample_to(x, layer1), layer1), dim=1)
        )
        x = self.decode0(
            torch.cat((self._upsample_to(x, shallow), shallow), dim=1)
        )
        x = F.interpolate(
            x,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        return torch.sigmoid(self.output(x))


class ResNet18UNet(nn.Module):
    """ResNet-18 encoder with a U-Net decoder for binary segmentation."""

    def __init__(self, pretrained=True, freeze_encoder=True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.encoder = resnet18(weights=weights)
        self.encoder.fc = nn.Identity()
        self.decoder = ResNetUNetDecoder((64, 64, 128, 256, 512))
        self.encoder_frozen = False
        self.final_block_finetuning = False
        if freeze_encoder:
            self.freeze_encoder()

    def freeze_encoder(self):
        self.encoder.requires_grad_(False)
        self.encoder_frozen = True
        self.final_block_finetuning = False

    def unfreeze_final_block(self):
        self.encoder.requires_grad_(False)
        self.encoder.layer4.requires_grad_(True)
        self.encoder_frozen = False
        self.final_block_finetuning = True

    def train(self, mode=True):
        super().train(mode)
        if self.encoder_frozen:
            self.encoder.eval()
        elif self.final_block_finetuning:
            self.encoder.eval()
            self.encoder.layer4.train(mode)
            for module in self.encoder.layer4.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def forward(self, x):
        output_size = x.shape[-2:]
        shallow = self.encoder.relu(self.encoder.bn1(self.encoder.conv1(x)))
        x = self.encoder.maxpool(shallow)
        layer1 = self.encoder.layer1(x)
        layer2 = self.encoder.layer2(layer1)
        layer3 = self.encoder.layer3(layer2)
        deep = self.encoder.layer4(layer3)
        return self.decoder(
            (shallow, layer1, layer2, layer3, deep),
            output_size=output_size,
        )