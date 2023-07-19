from torch import nn
import torch


def get_acti_func(acti_func, params):
    acti_func = acti_func.lower()
    if (acti_func == 'relu'):
        inplace = params.get('relu_inplace', False)
        return nn.ReLU(inplace)

    elif (acti_func == 'leakyrelu'):
        slope = params.get('leakyrelu_negative_slope', 1e-2)
        inplace = params.get('leakyrelu_inplace', False)
        return nn.LeakyReLU(slope, inplace)

    elif (acti_func == 'prelu'):
        num_params = params.get('prelu_num_parameters', 1)
        init_value = params.get('prelu_init', 0.25)
        return nn.PReLU(num_params, init_value)

    elif (acti_func == 'rrelu'):
        lower = params.get('rrelu_lower', 1.0 / 8)
        upper = params.get('rrelu_upper', 1.0 / 3)
        inplace = params.get('rrelu_inplace', False)
        return nn.RReLU(lower, upper, inplace)

    elif (acti_func == 'elu'):
        alpha = params.get('elu_alpha', 1.0)
        inplace = params.get('elu_inplace', False)
        return nn.ELU(alpha, inplace)

    elif (acti_func == 'celu'):
        alpha = params.get('celu_alpha', 1.0)
        inplace = params.get('celu_inplace', False)
        return nn.CELU(alpha, inplace)

    elif (acti_func == 'selu'):
        inplace = params.get('selu_inplace', False)
        return nn.SELU(inplace)

    elif (acti_func == 'glu'):
        dim = params.get('glu_dim', -1)
        return nn.GLU(dim)

    elif (acti_func == 'sigmoid'):
        return nn.Sigmoid()

    elif (acti_func == 'logsigmoid'):
        return nn.LogSigmoid()

    elif (acti_func == 'tanh'):
        return nn.Tanh()

    elif (acti_func == 'hardtanh'):
        min_val = params.get('hardtanh_min_val', -1.0)
        max_val = params.get('hardtanh_max_val', 1.0)
        inplace = params.get('hardtanh_inplace', False)
        return nn.Hardtanh(min_val, max_val, inplace)

    elif (acti_func == 'softplus'):
        beta = params.get('softplus_beta', 1.0)
        threshold = params.get('softplus_threshold', 20)
        return nn.Softplus(beta, threshold)

    elif (acti_func == 'softshrink'):
        lambd = params.get('softshrink_lambda', 0.5)
        return nn.Softshrink(lambd)

    elif (acti_func == 'softsign'):
        return nn.Softsign()

    else:
        raise ValueError("Not implemented: {0:}".format(acti_func))


def interleaved_concate(f1, f2):
    f1_shape = list(f1.shape)
    f2_shape = list(f2.shape)
    c1 = f1_shape[1]
    c2 = f2_shape[1]

    f1_shape_new = f1_shape[:1] + [c1, 1] + f1_shape[2:]
    f2_shape_new = f2_shape[:1] + [c2, 1] + f2_shape[2:]

    f1_reshape = torch.reshape(f1, f1_shape_new)
    f2_reshape = torch.reshape(f2, f2_shape_new)
    output = torch.cat((f1_reshape, f2_reshape), dim=2)
    out_shape = f1_shape[:1] + [c1 + c2] + f1_shape[2:]
    output = torch.reshape(output, out_shape)
    return output


class ConvolutionLayer(nn.Module):
    """
    A compose layer with the following components:
    convolution -> (batch_norm / layer_norm / group_norm / instance_norm) -> (activation) -> (dropout)
    Batch norm and activation are optional.

    :param in_channels: (int) The input channel number.
    :param out_channels: (int) The output channel number.
    :param kernel_size: The size of convolution kernel. It can be either a single
        int or a tupe of two or three ints.
    :param dim: (int) The dimention of convolution (2 or 3).
    :param stride: (int) The stride of convolution.
    :param padding: (int) Padding size.
    :param dilation: (int) Dilation rate.
    :param conv_group: (int) The groupt number of convolution.
    :param bias: (bool) Add bias or not for convolution.
    :param norm_type: (str or None) Normalization type, can be `batch_norm`, 'group_norm'.
    :param norm_group: (int) The number of group for group normalization.
    :param acti_func: (str or None) Activation funtion.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dim=3,
                 stride=1, padding=0, dilation=1, conv_group=1, bias=True,
                 norm_type='batch_norm', norm_group=1, acti_func=None):
        super(ConvolutionLayer, self).__init__()
        self.n_in_chns = in_channels
        self.n_out_chns = out_channels
        self.norm_type = norm_type
        self.norm_group = norm_group
        self.acti_func = acti_func

        assert (dim == 2 or dim == 3)
        if (dim == 2):
            self.conv = nn.Conv2d(in_channels, out_channels,
                                  kernel_size, stride, padding, dilation, conv_group, bias)
            if (self.norm_type == 'batch_norm'):
                self.bn = nn.BatchNorm2d(out_channels)
            elif (self.norm_type == 'group_norm'):
                self.bn = nn.GroupNorm(self.norm_group, out_channels)
            elif (self.norm_type == 'instance_norm'):
                self.bn = nn.InstanceNorm2d(out_channels)
            elif (self.norm_type is not None):
                raise ValueError("unsupported normalization method {0:}".format(norm_type))
        else:
            self.conv = nn.Conv3d(in_channels, out_channels,
                                  kernel_size, stride, padding, dilation, conv_group, bias)
            if (self.norm_type == 'batch_norm'):
                self.bn = nn.BatchNorm3d(out_channels)
            elif (self.norm_type == 'group_norm'):
                self.bn = nn.GroupNorm(self.norm_group, out_channels)
            elif (self.norm_type == 'instance_norm'):
                self.bn = nn.InstanceNorm3d(out_channels)
            elif (self.norm_type is not None):
                raise ValueError("unsupported normalization method {0:}".format(norm_type))

    def forward(self, x):
        f = self.conv(x)
        if (self.norm_type is not None):
            f = self.bn(f)
        if (self.acti_func is not None):
            f = self.acti_func(f)
        return f


class DeconvolutionLayer(nn.Module):
    """
    A compose layer with the following components:
    deconvolution -> (batch_norm / layer_norm / group_norm / instance_norm) -> (activation) -> (dropout)
    Batch norm and activation are optional.

    :param in_channels: (int) The input channel number.
    :param out_channels: (int) The output channel number.
    :param kernel_size: The size of convolution kernel. It can be either a single
        int or a tupe of two or three ints.
    :param dim: (int) The dimention of convolution (2 or 3).
    :param stride: (int) The stride of convolution.
    :param padding: (int) Padding size.
    :param dilation: (int) Dilation rate.
    :param groups: (int) The groupt number of convolution.
    :param bias: (bool) Add bias or not for convolution.
    :param batch_norm: (bool) Use batch norm or not.
    :param acti_func: (str or None) Activation funtion.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 dim=3, stride=1, padding=0, output_padding=0,
                 dilation=1, groups=1, bias=True,
                 norm_type="batch_norm", acti_func=None):
        super(DeconvolutionLayer, self).__init__()
        self.n_in_chns = in_channels
        self.n_out_chns = out_channels
        self.norm_type = norm_type
        self.acti_func = acti_func

        assert (dim == 2 or dim == 3)
        if (dim == 2):
            self.conv = nn.ConvTranspose2d(in_channels, out_channels,
                                           kernel_size, stride, padding, output_padding,
                                           groups, bias, dilation)
            if (self.norm_type == "group_norm"):
                self.bn = nn.GroupNorm(groups, out_channels)
            elif (self.norm_type == "batch_norm"):
                self.bn = nn.BatchNorm2d(out_channels)
            elif (self.norm_type == "instance_norm"):
                self.bn = nn.InstanceNorm2d(out_channels)
        else:
            self.conv = nn.ConvTranspose3d(in_channels, out_channels,
                                           kernel_size, stride, padding, output_padding,
                                           groups, bias, dilation)
            if (self.norm_type == "group_norm"):
                self.bn = nn.GroupNorm(groups, out_channels)
            elif (self.norm_type == "batch_norm"):
                self.bn = nn.BatchNorm3d(out_channels)
            elif (self.norm_type == "instance_norm"):
                self.bn = nn.InstanceNorm3d(out_channels)

    def forward(self, x):
        f = self.conv(x)
        if (self.norm_type in ["normal", "grouped"]):
            f = self.bn(f)
        if (self.acti_func is not None):
            f = self.acti_func(f)
        return f


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type, groups, acti_func):
        super(UNetBlock, self).__init__()

        self.in_chns = in_channels
        self.out_chns = out_channels
        self.acti_func = acti_func

        group1 = 1 if (in_channels < 8) else groups
        self.conv1 = ConvolutionLayer(in_channels, out_channels, 1,
                                      dim=2, padding=0, conv_group=group1, norm_type=norm_type, norm_group=group1,
                                      acti_func=acti_func)
        self.conv2 = ConvolutionLayer(out_channels, out_channels, 3,
                                      dim=2, padding=1, conv_group=groups, norm_type=norm_type, norm_group=groups,
                                      acti_func=acti_func)

    def forward(self, x):
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        return f2


class MGUNet(nn.Module):
    def __init__(self, params):
        super(MGUNet, self).__init__()
        self.params = params
        self.ft_chns = [self.params['ndf'] * i for i in range(1, 6)]
        self.in_chns = self.params['in_chns']
        self.ft_groups = self.params['feature_grps']
        self.norm_type = self.params['norm_type']
        self.n_class = self.params['class_num']
        self.acti_func = get_acti_func(self.params['acti_func'], self.params)
        self.dropout = self.params['dropout']
        self.deep_supervision = self.params["deep_supervision"]
        self.loose_sup = self.params["loose_sup"]

        self.block1 = UNetBlock(self.in_chns, self.ft_chns[0], self.norm_type[0], self.ft_groups,
                                self.acti_func)

        self.block2 = UNetBlock(self.ft_chns[0], self.ft_chns[1], self.norm_type[0], self.ft_groups,
                                self.acti_func)

        self.block3 = UNetBlock(self.ft_chns[1], self.ft_chns[2], self.norm_type[0], self.ft_groups,
                                self.acti_func)

        self.block4 = UNetBlock(self.ft_chns[2], self.ft_chns[3], self.norm_type[0], self.ft_groups,
                                self.acti_func)

        self.block5 = UNetBlock(self.ft_chns[3], self.ft_chns[4], self.norm_type[0], self.ft_groups,
                                self.acti_func)

        self.block6 = UNetBlock(self.ft_chns[3] * 2, self.ft_chns[3], self.norm_type[0], self.ft_groups,
                                self.acti_func)

        self.block7 = UNetBlock(self.ft_chns[2] * 2, self.ft_chns[2], self.norm_type[0], self.ft_groups,
                                self.acti_func)

        self.block8 = UNetBlock(self.ft_chns[1] * 2, self.ft_chns[1], self.norm_type[0], self.ft_groups,
                                self.acti_func)

        self.block9 = UNetBlock(self.ft_chns[0] * 2, self.ft_chns[0], self.norm_type[0], self.ft_groups,
                                self.acti_func)

        self.down1 = nn.MaxPool2d(kernel_size=2)
        self.down2 = nn.MaxPool2d(kernel_size=2)
        self.down3 = nn.MaxPool2d(kernel_size=2)

        self.down4 = nn.MaxPool2d(kernel_size=2)
        self.up1 = DeconvolutionLayer(self.ft_chns[4], self.ft_chns[3], kernel_size=2,
                                      dim=2, stride=2, groups=self.ft_groups,
                                      acti_func=self.acti_func, norm_type=self.norm_type[1])
        self.up2 = DeconvolutionLayer(self.ft_chns[3], self.ft_chns[2], kernel_size=2,
                                      dim=2, stride=2, groups=self.ft_groups,
                                      acti_func=self.acti_func, norm_type=self.norm_type[1])
        self.up3 = DeconvolutionLayer(self.ft_chns[2], self.ft_chns[1], kernel_size=2,
                                      dim=2, stride=2, groups=self.ft_groups,
                                      acti_func=self.acti_func, norm_type=self.norm_type[1])
        self.up4 = DeconvolutionLayer(self.ft_chns[1], self.ft_chns[0], kernel_size=2,
                                      dim=2, stride=2, groups=self.ft_groups,
                                      acti_func=self.acti_func, norm_type=self.norm_type[1])

        if (self.dropout):
            self.drop1 = nn.Dropout(p=0.1)
            self.drop2 = nn.Dropout(p=0.2)
            self.drop3 = nn.Dropout(p=0.3)
            self.drop4 = nn.Dropout(p=0.4)
            self.drop5 = nn.Dropout(p=0.5)

        self.conv9 = nn.Conv2d(self.ft_chns[0], self.n_class * self.ft_groups,
                               kernel_size=3, padding=1, groups=self.ft_groups)

        if self.deep_supervision == "normal":
            self.out_conv1 = nn.Conv2d(self.ft_chns[3] * 2, self.n_class, kernel_size=1, groups=self.ft_groups)
            self.out_conv2 = nn.Conv2d(self.ft_chns[2] * 2, self.n_class, kernel_size=1, groups=self.ft_groups)
            self.out_conv3 = nn.Conv2d(self.ft_chns[1] * 2, self.n_class, kernel_size=1, groups=self.ft_groups)
        elif self.deep_supervision == "grouped":
            self.out_conv1 = nn.Conv2d(self.ft_chns[3] * 2, self.n_class * self.ft_groups, kernel_size=1,
                                       groups=self.ft_groups)
            self.out_conv2 = nn.Conv2d(self.ft_chns[2] * 2, self.n_class * self.ft_groups, kernel_size=1,
                                       groups=self.ft_groups)
            self.out_conv3 = nn.Conv2d(self.ft_chns[1] * 2, self.n_class * self.ft_groups, kernel_size=1,
                                       groups=self.ft_groups)
        else:
            pass

        if self.loose_sup:
            self.out_adjust = nn.Conv2d(self.n_class * self.ft_groups, self.n_class, kernel_size=1)

    def forward(self, x):
        x_shape = list(x.shape)
        if (len(x_shape) == 5):
            [N, C, D, H, W] = x_shape
            new_shape = [N * D, C, H, W]
            x = torch.transpose(x, 1, 2)
            x = torch.reshape(x, new_shape)
        f1 = self.block1(x)
        if (self.dropout):
            f1 = self.drop1(f1)
        d1 = self.down1(f1)

        f2 = self.block2(d1)
        if (self.dropout):
            f2 = self.drop2(f2)
        d2 = self.down2(f2)

        f3 = self.block3(d2)
        if (self.dropout):
            f3 = self.drop3(f3)
        d3 = self.down3(f3)

        f4 = self.block4(d3)
        if (self.dropout):
            f4 = self.drop4(f4)

        d4 = self.down4(f4)
        f5 = self.block5(d4)
        if (self.dropout):
            f5 = self.drop5(f5)

        f5up = self.up1(f5)
        f4cat = interleaved_concate(f4, f5up)
        f6 = self.block6(f4cat)
        f6up = self.up2(f6)
        f3cat = interleaved_concate(f3, f6up)

        f7 = self.block7(f3cat)
        f7up = self.up3(f7)

        f2cat = interleaved_concate(f2, f7up)
        f8 = self.block8(f2cat)
        f8up = self.up4(f8)

        f1cat = interleaved_concate(f1, f8up)
        f9 = self.block9(f1cat)

        output = self.conv9(f9)

        mulpred = None
        if self.deep_supervision == "normal":
            mulpred = [self.out_conv1(f4cat),
                       self.out_conv2(f3cat),
                       self.out_conv3(f2cat)]
        elif self.deep_supervision == "grouped":
            mulpred = [torch.chunk(self.out_conv1(f4cat), self.ft_groups, dim=1),
                       torch.chunk(self.out_conv2(f3cat), self.ft_groups, dim=1),
                       torch.chunk(self.out_conv3(f2cat), self.ft_groups, dim=1)]
        feature = [f4cat, f3cat, f2cat]

        if not self.loose_sup:
            return torch.chunk(output, self.ft_groups, dim=1), mulpred, feature

        return [self.out_adjust(output)], mulpred, feature
