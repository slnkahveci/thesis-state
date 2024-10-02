# in progress new arch in may style
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
    
    

#preceeded by up-conv
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation):
        super(DecoderBlock, self).__init__()
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
    
# last one is succeeded by a 1x1 conv


class newUNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=1, depth=5, num_filters=64, activation= 'relu'): # bottleneck is the nth depth

        filters = [num_filters * (2 ** i) for i in range(depth-1)]  # Adjust filters based on depth
        super(newUNet, self).__init__()
        self.depth = depth

        self.encoder_joint = nn.ModuleList()
        self.encoder_skips = nn.ModuleList()
        self.encoder_maxs = nn.ModuleList()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        for filter in filters:
            self.encoder_joint.append(EncoderBlock(in_channels, filter,activation=activation))
            self.encoder_maxs.append(EncoderBlock(filter, filter,activation=activation))
            in_channels = filter

        self.bottleneck = EncoderBlock(filters[-1], filters[-1]*2,activation=activation)

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for filter in reversed(filters):
            self.upconvs.append(nn.ConvTranspose2d(filter*2, filter, kernel_size=2, stride=2))
            self.decoders.append(DecoderBlock(filter*2, filter,activation=activation))
            self.encoder_skips.append(EncoderBlock(filter, filter,activation=activation))

        self.output = nn.Conv2d(filters[0], out_channels, kernel_size=1)



    def forward(self, x):
        skips = []

        for i in range(self.depth -1 ):
            x = self.encoder_joint[i](x)
            skips.append(x)
            x = self.encoder_maxs[i](x)
            x = self.maxpool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for i in range(self.depth -1):

            x = self.upconvs[i](x)
            skip = skips[i]

            if x.shape != skip.shape:
                x = TF.resize(x, size=skip.shape[2:])

            skip = self.encoder_skips[i](skip) #############
            concat_skip = torch.cat((skip, x), dim=1)
            x = self.decoders[i](concat_skip)

        return self.output(x)

def test():
    x = torch.randn((3, 6, 512, 512))
    model = newUNet(in_channels=6, out_channels=1)
    preds = model(x)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(preds.shape)
    print(x.shape)
    # assert preds.shape == x.shape
    # visualise the model

    #make_dot(preds, params=dict(model.named_parameters()), show_attrs=True, show_saved=True).render("newmodel", format="png")
    #torchsummary.summary(model, input_size=(6, 160, 160))  ################################################################
    """
    # check cudnn
    with open("model.onnx", "wb") as f:
      torch.onnx.export(model, torch.randn(1, 6, 512, 512),
                        f, verbose=True, input_names=["input"], output_names=["output"]) # try export in training.mode


    print(torch.backends.cudnn.version())
    print(torch.backends.cudnn.enabled)
    print(torch.backends.cudnn.deterministic)
    print(torch.backends.cudnn.benchmark)
    """

if __name__ == "__main__":
    test()
