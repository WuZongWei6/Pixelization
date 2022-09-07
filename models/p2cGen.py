from .basic_layer import *


class P2CGen(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_downsample, n_res, activ='relu', pad_type='reflect'):
        super(P2CGen, self).__init__()
        self.RGBEnc = RGBEncoder(input_dim, dim, n_downsample, n_res, "in", activ, pad_type=pad_type)
        self.RGBDec = RGBDecoder(self.RGBEnc.output_dim, output_dim, n_downsample, n_res, res_norm='in',
                                      activ=activ, pad_type=pad_type)

    def forward(self, x):
        x = self.RGBEnc(x)
        # print("encoder->>", x.shape)
        x = self.RGBDec(x)
        # print(x_small.shape)
        # print(x_middle.shape)
        # print(x_big.shape)
        #return y_small, y_middle, y_big
        return x


class RGBEncoder(nn.Module):
    def __init__(self, input_dim, dim, n_downsample, n_res, norm, activ, pad_type):
        super(RGBEncoder, self).__init__()
        self.model = []
        self.model += [ConvBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [ConvBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)


class RGBDecoder(nn.Module):
    def __init__(self, dim, output_dim, n_upsample, n_res, res_norm, activ='relu', pad_type='zero'):
        super(RGBDecoder, self).__init__()
        # self.model = []
        # # AdaIN residual blocks
        # self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # # upsampling blocks
        # for i in range(n_upsample):
        #     self.model += [nn.Upsample(scale_factor=2, mode='nearest'),
        #                    ConvBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
        #     dim //= 2
        # # use reflection padding in the last conv layer
        # self.model += [ConvBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
        # self.model = nn.Sequential(*self.model)
        self.Res_Blocks = ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)
        self.upsample_block1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_1 = ConvBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)
        dim //= 2
        self.upsample_block2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_2 = ConvBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)
        dim //= 2
        self.conv_3 = ConvBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)

    def forward(self, x):
        x = self.Res_Blocks(x)
        # print(x.shape)
        x = self.upsample_block1(x)
        # print(x.shape)
        x = self.conv_1(x)
        # print(x_small.shape)
        x = self.upsample_block2(x)
        # print(x.shape)
        x = self.conv_2(x)
        # print(x_middle.shape)
        x = self.conv_3(x)
        # print(x_big.shape)
        return x
