from .basic_layer import *
import math
from torch.nn import Parameter
#from pytorch_metric_learning import losses

'''
Margin code is borrowed from https://github.com/MuggleWang/CosFace_pytorch and https://github.com/wujiyang/Face_Pytorch.
'''
def cosine_sim(x1, x2, dim=1, eps=1e-8):
    ip = torch.mm(x1, x2.t())  #  w  7*512
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return ip / torch.ger(w1,w2).clamp(min=eps)

class MarginCosineProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.40):
        super(MarginCosineProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_features, in_features)) # 7 512
        nn.init.xavier_uniform_(self.weight)
        #stdv = 1. / math.sqrt(self.weight.size(1))
        #self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, label):
        cosine = cosine_sim(input, self.weight)  # 1*512  7*512
        # cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        # --------------------------- convert label to one-hot ---------------------------
        # https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = self.s * (cosine - one_hot * self.m)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'

class ArcMarginProduct(nn.Module):
    def __init__(self, in_feature=128, out_feature=10575, s=32.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_feature, in_feature))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label):
        # cos(theta)
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        # cos(theta + m)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        #one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        return output


class MultiMarginProduct(nn.Module):
    def __init__(self, in_feature=128, out_feature=10575, s=32.0, m1=0.20, m2=0.35, easy_margin=False):
        super(MultiMarginProduct, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.s = s
        self.m1 = m1
        self.m2 = m2
        self.weight = Parameter(torch.Tensor(out_feature, in_feature))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m1 = math.cos(m1)
        self.sin_m1 = math.sin(m1)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m1)
        self.mm = math.sin(math.pi - m1) * m1

    def forward(self, x, label):
        # cos(theta)
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        # cos(theta + m1)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m1 - sine * self.sin_m1

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)


        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine) # additive angular margin
        output = output - one_hot * self.m2 # additive cosine margin
        output = output * self.s

        return output


class CPDis(nn.Module):
    """PatchGAN."""
    def __init__(self, image_size=256, conv_dim=64, repeat_num=3, norm='SN'):
        super(CPDis, self).__init__()

        layers = []
        if norm == 'SN':
            layers.append(spectral_norm(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1)))
        else:
            layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01, inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            if norm == 'SN':
                layers.append(spectral_norm(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1)))
            else:
                layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            curr_dim = curr_dim * 2

        # k_size = int(image_size / np.power(2, repeat_num))
        if norm == 'SN':
            layers.append(spectral_norm(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=1, padding=1)))
        else:
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=1, padding=1))
        layers.append(nn.LeakyReLU(0.01, inplace=True))
        curr_dim = curr_dim * 2

        self.main = nn.Sequential(*layers)
        if norm == 'SN':
            self.conv1 = spectral_norm(nn.Conv2d(curr_dim, 1, kernel_size=4, stride=1, padding=1, bias=False))
        else:
            self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=4, stride=1, padding=1, bias=False)

    def forward(self, x):
        if x.ndim == 5:
            x = x.squeeze(0)
        assert x.ndim == 4, x.ndim
        h = self.main(x)
        # out_real = self.conv1(h)
        out_makeup = self.conv1(h)
        # return out_real.squeeze(), out_makeup.squeeze()
        return out_makeup


class CPDis_cls(nn.Module):
    """PatchGAN."""
    def __init__(self, image_size=256, conv_dim=64, repeat_num=3, norm='SN'):
        super(CPDis_cls, self).__init__()

        layers = []
        if norm == 'SN':
            layers.append(spectral_norm(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1)))
        else:
            layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01, inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            if norm == 'SN':
                layers.append(spectral_norm(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1)))
            else:
                layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            curr_dim = curr_dim * 2

        # k_size = int(image_size / np.power(2, repeat_num))
        if norm == 'SN':
            layers.append(spectral_norm(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=1, padding=1)))
        else:
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=1, padding=1))
        layers.append(nn.LeakyReLU(0.01, inplace=True))
        curr_dim = curr_dim * 2

        self.main = nn.Sequential(*layers)
        if norm == 'SN':
            self.conv1 = spectral_norm(nn.Conv2d(curr_dim, 1, kernel_size=4, stride=1, padding=1, bias=False))
            self.classifier_pool = nn.AdaptiveAvgPool2d(1)
            self.classifier_conv = nn.Conv2d(512, 512, 1, 1, 0)
            self.classifier = MarginCosineProduct(512,7)#ArcMarginProduct(512, 7)
            print("Using Large Margin Cosine Loss.")

        else:
            self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=4, stride=1, padding=1, bias=False)

    def forward(self, x, label):
        if x.ndim == 5:
            x = x.squeeze(0)
        assert x.ndim == 4, x.ndim
        h = self.main(x)  # ([1, 512, 31, 31])
        #print(out_cls.shape)
        out_cls = self.classifier_pool(h)
        #print(out_cls.shape)
        out_cls = self.classifier_conv(out_cls)
        #print(out_cls.shape)
        out_cls = torch.squeeze(out_cls, -1)
        out_cls = torch.squeeze(out_cls, -1)
        out_cls = self.classifier(out_cls, label)
        out_makeup = self.conv1(h)  # torch.Size([1, 1, 30, 30])
        # return out_real.squeeze(), out_makeup.squeeze()
        return out_makeup, out_cls

class SpectralNorm(object):
    def __init__(self):
        self.name = "weight"
        # print(self.name)
        self.power_iterations = 1

    def compute_weight(self, module):
        u = getattr(module, self.name + "_u")
        v = getattr(module, self.name + "_v")
        w = getattr(module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))
        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        return w / sigma.expand_as(w)

    @staticmethod
    def apply(module):
        name = "weight"
        fn = SpectralNorm()

        try:
            u = getattr(module, name + "_u")
            v = getattr(module, name + "_v")
            w = getattr(module, name + "_bar")
        except AttributeError:
            w = getattr(module, name)
            height = w.data.shape[0]
            width = w.view(height, -1).data.shape[1]
            u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
            v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
            w_bar = Parameter(w.data)

            # del module._parameters[name]

            module.register_parameter(name + "_u", u)
            module.register_parameter(name + "_v", v)
            module.register_parameter(name + "_bar", w_bar)

        # remove w from parameter list
        del module._parameters[name]

        setattr(module, name, fn.compute_weight(module))

        # recompute weight before every forward()
        module.register_forward_pre_hook(fn)

        return fn

    def remove(self, module):
        weight = self.compute_weight(module)
        delattr(module, self.name)
        del module._parameters[self.name + '_u']
        del module._parameters[self.name + '_v']
        del module._parameters[self.name + '_bar']
        module.register_parameter(self.name, Parameter(weight.data))

    def __call__(self, module, inputs):
        setattr(module, self.name, self.compute_weight(module))

def spectral_norm(module):
    SpectralNorm.apply(module)
    return module

def remove_spectral_norm(module):
    name = 'weight'
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, SpectralNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError("spectral_norm of '{}' not found in {}"
                     .format(name, module))
