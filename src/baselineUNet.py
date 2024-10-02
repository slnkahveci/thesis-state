#   VERSION ONE FROM MAY

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF



class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation):
        super(EncoderBlock, self).__init__()
        self.activation = self._get_activation(activation)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False), # same convolution: input size = output size
            nn.BatchNorm2d(out_channels), # bias=False because batchnorm already has bias
            self.activation,
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            self.activation
        )

    def _get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            return nn.LeakyReLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        
    def forward(self, x):
        return self.block(x)
    
    


# each layer in decoder has 2 3x3conv+ReLU followed by maxpool
class DoubleEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation):
        super(DoubleEncoderBlock, self).__init__()
        self.activation = self._get_activation(activation)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False), # batchnorm nullifies any effect that the bias might have ()
            nn.BatchNorm2d(out_channels),
            self.activation,
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            self.activation,
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            self.activation,
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            self.activation

        )
    def _get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            return nn.LeakyReLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        
    def forward(self, x):
        return self.block(x)





class baselineUNet(nn.Module): # modules needed for forward pass

    def __init__(self, in_channels=6, out_channels=1, depth = 5, num_filters=64, activation='relu'):
        features = [num_filters * (2 ** i) for i in range(depth)]
        super(baselineUNet, self).__init__()

        # ModuleList instead of normal list to register list elements as submodules of the module UNet
        self.downs = nn.ModuleList() # downsampling, encoder
        self.ups = nn.ModuleList() # upsampling, decoder
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # maxpooling floors the division, this might cause problems when upsampling

        # encoder
        for feature in features:
            self.downs.append(DoubleEncoderBlock(in_channels, out_channels=feature, activation=activation))
            in_channels = feature

        self.bottleneck = DoubleEncoderBlock(
            features[-1],
            features[-1]*2,
            activation=activation
            )

        # decoder
        # transpose convolution (they sometimes create artifacts) or bilinear followed by EncoderBlock (can bypass artifacts)
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d( # up-conv 2x2
                in_channels=feature*2, # x2 because skip connections
                out_channels=feature,
                kernel_size=2,
                stride=2))
            self.ups.append(EncoderBlock(feature*2, feature,activation=activation)) # double conv3x3,ReLU

        # final layer 1x1conv
        self.final_conv = nn.Conv2d(in_channels=features[0], out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            # save output of each layer for skip connections before pooling
            # first element has the highest resolution and least features
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1] # reverse list for correct order

        for idx in range(0, len(self.ups), 2): # iterate by 2 because there is two elements in each layer
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                # sizes might mismatch due to maxpooling/upconv (original paper uses cropping instead of resizing)
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1) # concatenate along the channel dimension
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

def test():
    x = torch.randn((3, 1, 160, 160))
    model = baselineUNet(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()
