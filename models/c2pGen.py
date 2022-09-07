from .basic_layer import *
import torchvision.models as models



class AliasNet(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_downsample, n_res, activ='relu', pad_type='reflect'):
        super(AliasNet, self).__init__()
        self.RGBEnc = AliasRGBEncoder(input_dim, dim, n_downsample, n_res, "in", activ, pad_type=pad_type)
        self.RGBDec = AliasRGBDecoder(self.RGBEnc.output_dim, output_dim, n_downsample, n_res, res_norm='in',
                                      activ=activ, pad_type=pad_type)

    def forward(self, x):
        x = self.RGBEnc(x)
        x = self.RGBDec(x)
        return x


class AliasRGBEncoder(nn.Module):
    def __init__(self, input_dim, dim, n_downsample, n_res, norm, activ, pad_type):
        super(AliasRGBEncoder, self).__init__()
        self.model = []
        self.model += [AliasConvBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [AliasConvBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        # residual blocks
        self.model += [AliasResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)


class AliasRGBDecoder(nn.Module):
    def __init__(self, dim, output_dim, n_upsample, n_res, res_norm, activ='relu', pad_type='zero'):
        super(AliasRGBDecoder, self).__init__()
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
        self.Res_Blocks = AliasResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)
        self.upsample_block1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_1 = AliasConvBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)
        dim //= 2
        self.upsample_block2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_2 = AliasConvBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)
        dim //= 2
        self.conv_3 = AliasConvBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)

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


class C2PGen(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_downsample, n_res, style_dim, mlp_dim, activ='relu', pad_type='reflect'):
        super(C2PGen, self).__init__()
        self.PBEnc = PixelBlockEncoder(input_dim, dim, style_dim, norm='none', activ=activ, pad_type=pad_type)
        self.RGBEnc = RGBEncoder(input_dim, dim, n_downsample, n_res, "in", activ, pad_type=pad_type)
        self.RGBDec = RGBDecoder(self.RGBEnc.output_dim, output_dim, n_downsample, n_res, res_norm='adain',
                                      activ=activ, pad_type=pad_type)
        self.MLP = MLP(style_dim, 2048, mlp_dim, 3, norm='none', activ=activ)

    def forward(self, clipart, pixelart, s=1):
        feature = self.RGBEnc(clipart)
        code = self.PBEnc(pixelart)
        result, cellcode = self.fuse(feature, code, s)
        return result#, cellcode   #return cellcode when visualizing the cell size code

    def fuse(self, content, style_code, s=1):
        #print("MLP input:code's shape:", style_code.shape)
        adain_params = self.MLP(style_code) * s # [batch,2048]
        #print("MLP output:adain_params's shape", adain_params.shape)
        #self.assign_adain_params(adain_params, self.RGBDec)
        images = self.RGBDec(content, adain_params)
        return images, adain_params

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2 * m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2 * m.num_features
        return num_adain_params


class PixelBlockEncoder(nn.Module):
    def __init__(self, input_dim, dim, style_dim, norm, activ, pad_type):
        super(PixelBlockEncoder, self).__init__()
        vgg19 = models.vgg.vgg19(pretrained=False)
        vgg19.classifier._modules['6'] = nn.Linear(4096, 7, bias=True)
        vgg19.load_state_dict(torch.load('./pixelart_vgg19.pth'))
        self.vgg = vgg19.features
        for p in self.vgg.parameters():
            p.requires_grad = False
        # vgg19 = models.vgg.vgg19(pretrained=False)
        # vgg19.load_state_dict(torch.load('./vgg.pth'))
        # self.vgg = vgg19.features
        # for p in self.vgg.parameters():
        #     p.requires_grad = False


        self.conv1 = ConvBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)  # 3->64,concat
        dim = dim * 2
        self.conv2 = ConvBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)  # 128->128
        dim = dim * 2
        self.conv3 = ConvBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)  # 256->256
        dim = dim * 2
        self.conv4 = ConvBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)  # 512->512
        dim = dim * 2

        self.model = []
        self.model += [nn.AdaptiveAvgPool2d(1)]  # global average pooling
        self.model += [nn.Conv2d(dim, style_dim, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def get_features(self, image, model, layers=None):
        if layers is None:
            layers = {'0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1', '19': 'conv4_1'}
        features = {}
        x = image
        # model._modules is a dictionary holding each module in the model
        for name, layer in model._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
        return features

    def componet_enc(self, x):
        # x [16,3,256,256]
        # factor_img [16,7,256,256]
        vgg_aux = self.get_features(x, self.vgg)  # x是3通道灰度图
        #x = torch.cat([x, factor_img], dim=1)  # [16,3+7,256,256]
        x = self.conv1(x) # 64 256 256
        x = torch.cat([x, vgg_aux['conv1_1']], dim=1)  # 128 256 256
        x = self.conv2(x)  #  128 128 128
        x = torch.cat([x, vgg_aux['conv2_1']], dim=1)  # 256 128 128
        x = self.conv3(x)  # 256 64 64
        x = torch.cat([x, vgg_aux['conv3_1']], dim=1)  # 512 64 64
        x = self.conv4(x)  # 512 32 32
        x = torch.cat([x, vgg_aux['conv4_1']], dim=1)  # 1024 32 32
        x = self.model(x)
        return x

    def forward(self, x):
        code = self.componet_enc(x)
        return code

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
        #self.Res_Blocks = ModulationResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)
        self.mod_conv_1 = ModulationConvBlock(256,256,3)
        self.mod_conv_2 = ModulationConvBlock(256,256,3)
        self.mod_conv_3 = ModulationConvBlock(256,256,3)
        self.mod_conv_4 = ModulationConvBlock(256,256,3)
        self.mod_conv_5 = ModulationConvBlock(256,256,3)
        self.mod_conv_6 = ModulationConvBlock(256,256,3)
        self.mod_conv_7 = ModulationConvBlock(256,256,3)
        self.mod_conv_8 = ModulationConvBlock(256,256,3)
        self.upsample_block1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_1 = ConvBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)
        dim //= 2
        self.upsample_block2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_2 = ConvBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)
        dim //= 2
        self.conv_3 = ConvBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)

    # def forward(self, x):
    #     residual = x
    #     out = self.model(x)
    #     out += residual
    #     return out
    def forward(self, x, code):
        residual = x
        x = self.mod_conv_1(x, code[:, :256])
        x = self.mod_conv_2(x, code[:, 256*1:256*2])
        x += residual
        residual = x
        x = self.mod_conv_2(x, code[:, 256*2:256 * 3])
        x = self.mod_conv_2(x, code[:, 256*3:256 * 4])
        x += residual
        residual =x
        x = self.mod_conv_2(x, code[:, 256*4:256 * 5])
        x = self.mod_conv_2(x, code[:, 256*5:256 * 6])
        x += residual
        residual = x
        x = self.mod_conv_2(x, code[:, 256*6:256 * 7])
        x = self.mod_conv_2(x, code[:, 256*7:256 * 8])
        x += residual
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

