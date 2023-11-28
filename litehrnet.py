import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
import torch.utils.checkpoint as cp

def channel_shuffle(x, groups):
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    # b, c, h, w =======>  b, g, c_per, h, w
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batch_size, -1, height, width)
    return x

class ConvModule(nn.Module):
    def __init__(self,
                 in_channels, 
                 out_channels,
                 kernel_size, 
                 stride=1, 
                 padding=0,
                 dilation=1, 
                 groups=1,
                 bias='auto',
                 use_bn=False,
                 act_type='ReLU'):
        super(ConvModule, self).__init__()
        self.use_bn = use_bn
        self.act_type = act_type

        if bias == 'auto':
            bias = not self.use_bn
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn = nn.BatchNorm2d(out_channels) if self.use_bn else nn.Identity()
        if self.act_type is None:
            self.act = nn.Identity()
        elif self.act_type == "ReLU":
            self.act = nn.ReLU()
        elif self.act_type == "Sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError()
            
    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        return x

class SpatialWeighting(nn.Module):
    def __init__(self,
                 channels,
                 ratio=16,
                 act_type=('ReLU', 'Sigmoid')):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = ConvModule(
            in_channels=channels,
            out_channels=int(channels / ratio),
            kernel_size=1,
            stride=1,
            act_type=act_type[0])
        self.conv2 = ConvModule(
            in_channels=int(channels / ratio),
            out_channels=channels,
            kernel_size=1,
            stride=1,
            act_type=act_type[1])

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.conv2(out)
        return x * out

class CrossResolutionWeighting(nn.Module):
    def __init__(self,
                 channels,
                 ratio=16,
                 use_bn=True,
                 act_type=('ReLU', 'Sigmoid')):
        super().__init__()

        self.channels = channels
        total_channel = sum(channels)
        self.conv1 = ConvModule(
            in_channels=total_channel,
            out_channels=int(total_channel / ratio),
            kernel_size=1,
            stride=1,
            use_bn=use_bn,
            act_type=act_type[0])
        self.conv2 = ConvModule(
            in_channels=int(total_channel / ratio),
            out_channels=total_channel,
            kernel_size=1,
            stride=1,
            use_bn=use_bn,
            act_type=act_type[1])

    def forward(self, x):
        mini_size = x[-1].size()[-2:]
        out = [F.adaptive_avg_pool2d(s, mini_size) for s in x[:-1]] + [x[-1]]
        out = torch.cat(out, dim=1)
        out = self.conv1(out)
        out = self.conv2(out)
        out = torch.split(out, self.channels, dim=1)
        out = [
            s * F.interpolate(a, size=s.size()[-2:], mode='nearest')
            for s, a in zip(x, out)
        ]
        return out

class ConditionalChannelWeighting(nn.Module):
    def __init__(self,
                 in_channels,
                 stride,
                 reduce_ratio,
                 use_bn=True,
                 with_cp=False):
        super().__init__()
        self.with_cp = with_cp
        self.stride = stride
        assert stride in [1, 2]

        branch_channels = [channel // 2 for channel in in_channels]

        self.cross_resolution_weighting = CrossResolutionWeighting(
            branch_channels,
            ratio=reduce_ratio,
            use_bn=use_bn)

        self.depthwise_convs = nn.ModuleList([
            ConvModule(
                channel,
                channel,
                kernel_size=3,
                stride=self.stride,
                padding=1,
                groups=channel,
                use_bn=use_bn,
                act_type=None) for channel in branch_channels
        ])

        self.spatial_weighting = nn.ModuleList([
            SpatialWeighting(channels=channel, ratio=4)
            for channel in branch_channels
        ])

    def forward(self, x):

        def _inner_forward(x):
            x = [s.chunk(2, dim=1) for s in x]
            x1 = [s[0] for s in x]
            x2 = [s[1] for s in x]

            x2 = self.cross_resolution_weighting(x2)
            x2 = [dw(s) for s, dw in zip(x2, self.depthwise_convs)]
            x2 = [sw(s) for s, sw in zip(x2, self.spatial_weighting)]

            out = [torch.cat([s1, s2], dim=1) for s1, s2 in zip(x1, x2)]
            out = [channel_shuffle(s, 2) for s in out]

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out

class LiteHRModule(nn.Module):
    def __init__(
            self,
            num_branches,
            num_blocks,
            in_channels,
            reduce_ratio,
            module_type,
            multiscale_output=False,
            with_fuse=True,
            use_bn=True,
            with_cp=False,
            branches_no_fuse=None,
    ):
        super().__init__()
        self._check_branches(num_branches, in_channels)

        self.in_channels = in_channels
        self.num_branches = num_branches

        self.module_type = module_type
        self.multiscale_output = multiscale_output
        self.with_fuse = with_fuse
        self.use_bn = use_bn
        self.with_cp = with_cp
        self.branches_no_fuse = branches_no_fuse

        if self.module_type == 'LITE':
            self.layers = self._make_weighting_blocks(num_blocks, reduce_ratio)
        elif self.module_type == 'NAIVE':
            raise NotImplementedError()
            # self.layers = self._make_naive_branches(num_branches, num_blocks)
        if self.with_fuse:
            self.fuse_layers = self._make_fuse_layers()
            self.relu = nn.ReLU()

    def _check_branches(self, num_branches, in_channels):
        """Check input to avoid ValueError."""
        if num_branches != len(in_channels):
            error_msg = f'NUM_BRANCHES({num_branches}) ' \
                f'!= NUM_INCHANNELS({len(in_channels)})'
            raise ValueError(error_msg)

    def _make_weighting_blocks(self, num_blocks, reduce_ratio, stride=1):
        layers = []
        for i in range(num_blocks):
            layers.append(
                ConditionalChannelWeighting(
                    self.in_channels,
                    stride=stride,
                    reduce_ratio=reduce_ratio,
                    use_bn=self.use_bn,
                    with_cp=self.with_cp))

        return nn.Sequential(*layers)

    def _make_fuse_layers(self):
        """Make fuse layer."""
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        in_channels = self.in_channels
        fuse_layers = []
        num_out_branches = num_branches if self.multiscale_output else 1
        for i in range(num_out_branches):
            if self.branches_no_fuse is not None and i in self.branches_no_fuse:
                fuse_layers.append(None)
                continue

            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                in_channels[j],
                                in_channels[i],
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False),
                            nn.BatchNorm2d( in_channels[i]),
                            nn.Upsample(
                                scale_factor=2**(j - i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv_downsamples = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv_downsamples.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        in_channels[j],
                                        in_channels[j],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        groups=in_channels[j],
                                        bias=False),
                                    nn.BatchNorm2d(in_channels[j]),
                                    nn.Conv2d(
                                        in_channels[j],
                                        in_channels[i],
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        bias=False),
                                    nn.BatchNorm2d(in_channels[i])))
                        else:
                            conv_downsamples.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        in_channels[j],
                                        in_channels[j],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        groups=in_channels[j],
                                        bias=False),
                                    nn.BatchNorm2d(in_channels[j]),
                                    nn.Conv2d(
                                        in_channels[j],
                                        in_channels[j],
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        bias=False),
                                    nn.BatchNorm2d(in_channels[j]),
                                    nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv_downsamples))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        """Forward function."""
        if self.num_branches == 1:
            return [self.layers[0](x[0])]
        if self.module_type == 'LITE':
            out = self.layers(x)
        elif self.module_type == 'NAIVE':
            for i in range(self.num_branches):
                x[i] = self.layers[i](x[i])
            out = x

        if self.with_fuse:
            out_fuse = []
            for i in range(len(self.fuse_layers)):
                if self.fuse_layers[i] is None:
                    out_fuse.append(None)
                else:
                    y = out[0] if i == 0 else self.fuse_layers[i][0](out[0])
                    for j in range(self.num_branches):
                        if i == j:
                            y += out[j]
                        else:
                            y += self.fuse_layers[i][j](out[j])
                    out_fuse.append(self.relu(y))
            out = out_fuse
        elif not self.multiscale_output:
            out = [out[0]]
        return out

class Stem(nn.Module):
    def __init__(self,
                 in_channels,
                 stem_channels,
                 out_channels,
                 expand_ratio,
                 stem_quarter=True,
                 use_bn=True,
                 with_cp=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bn = use_bn
        self.with_cp = with_cp

        self.conv1 = ConvModule(
            in_channels=in_channels,
            out_channels=stem_channels,
            kernel_size=3,
            stride=2 if stem_quarter else 1,
            padding=1,
            use_bn=self.use_bn,
            act_type="ReLU")

        mid_channels = int(round(stem_channels * expand_ratio))
        branch_channels = stem_channels // 2
        if stem_channels == self.out_channels:
            inc_channels = self.out_channels - branch_channels
        else:
            inc_channels = self.out_channels - stem_channels

        self.branch1 = nn.Sequential(
            ConvModule(
                branch_channels,
                branch_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=branch_channels,
                use_bn=use_bn,
                act_type=None),
            ConvModule(
                branch_channels,
                inc_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                use_bn=use_bn,
                act_type="ReLU"),
        )

        self.expand_conv = ConvModule(
            branch_channels,
            mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            use_bn=use_bn,
            act_type="ReLU")
        self.depthwise_conv = ConvModule(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=mid_channels,
            use_bn=use_bn,
            act_type=None)
        self.linear_conv = ConvModule(
            mid_channels,
            branch_channels
            if stem_channels == self.out_channels else stem_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            use_bn=use_bn,
            act_type="ReLU")

    def forward(self, x):

        def _inner_forward(x):
            x = self.conv1(x)
            x1, x2 = x.chunk(2, dim=1)

            x2 = self.expand_conv(x2)
            x2 = self.depthwise_conv(x2)
            x2 = self.linear_conv(x2)

            out = torch.cat((self.branch1(x1), x2), dim=1)

            out = channel_shuffle(out, 2)

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out

class LiteHRNet(nn.Module):
    def __init__(self,
                 extra=None,
                 model_cfg=None,
                 in_channels=1,
                 use_bn=True,
                 norm_eval=False,
                 with_cp=False,
                 zero_init_residual=False):
        super().__init__()

        if extra is not None:
            self.extra = extra
        else:
            if model_cfg is None:
                raise Exception("No config for model!")
            
            block_dims = model_cfg["block_dims"]
            initial_dim = model_cfg["initial_dim"]
            self.extra = dict(
                stem=dict(stem_channels=initial_dim, out_channels=initial_dim, expand_ratio=1, size_quarter=model_cfg["stem_quarter"]),
                num_stages=model_cfg["num_stages"],
                stages_spec=dict(
                    num_modules=model_cfg["num_modules"],
                    num_branches=model_cfg["num_branches"],
                    branches_no_fuse=model_cfg["branches_no_fuse"],
                    num_blocks=model_cfg["num_blocks"],
                    module_type=model_cfg["module_type"],
                    with_fuse=model_cfg["with_fuse"],
                    reduce_ratios=model_cfg["reduce_ratios"],
                    num_channels=[
                                    [block_dims[b_ij] for b_ij in range(model_cfg["num_branches"][b_i])]
                                                    for b_i in range(model_cfg["num_stages"])
                                  ],
                )
            )

        self.use_bn = use_bn
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.zero_init_residual = zero_init_residual

        self.stem = Stem(
            in_channels,
            stem_channels=self.extra['stem']['stem_channels'],
            out_channels=self.extra['stem']['out_channels'],
            expand_ratio=self.extra['stem']['expand_ratio'],
            stem_quarter=self.extra['stem']['size_quarter'],
            use_bn=self.use_bn)

        self.num_stages = self.extra['num_stages']
        self.stages_spec = self.extra['stages_spec']

        num_channels_last = [
            self.stem.out_channels,
        ]
        for i in range(self.num_stages):
            num_channels = self.stages_spec['num_channels'][i]
            num_channels = [num_channels[i] for i in range(len(num_channels))]
            setattr(
                self, 'transition{}'.format(i),
                self._make_transition_layer(num_channels_last, num_channels))

            stage, num_channels_last = self._make_stage(
                self.stages_spec, i, num_channels, multiscale_output=True)
            setattr(self, 'stage{}'.format(i), stage)

        self.init_weights()

    def _make_transition_layer(self, num_channels_pre_layer,
                               num_channels_cur_layer):
        """Make transition layer."""
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_pre_layer[i],
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                groups=num_channels_pre_layer[i],
                                bias=False),
                            nn.BatchNorm2d(num_channels_pre_layer[i]),
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            nn.ReLU()))
                else:
                    transition_layers.append(None)
            else:
                conv_downsamples = []
                for j in range(i + 1 - num_branches_pre):
                    in_channels = num_channels_pre_layer[-1]
                    out_channels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else in_channels
                    conv_downsamples.append(
                        nn.Sequential(
                            nn.Conv2d(
                                in_channels,
                                in_channels,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                groups=in_channels,
                                bias=False),
                            nn.BatchNorm2d(in_channels),
                            nn.Conv2d(
                                in_channels,
                                out_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU()))
                transition_layers.append(nn.Sequential(*conv_downsamples))

        return nn.ModuleList(transition_layers)

    def _make_stage(self,
                    stages_spec,
                    stage_index,
                    in_channels,
                    multiscale_output=True):
        self.num_modules = stages_spec['num_modules'][stage_index]
        self.num_branches = stages_spec['num_branches'][stage_index]
        self.num_blocks = stages_spec['num_blocks'][stage_index]
        self.reduce_ratio = stages_spec['reduce_ratios'][stage_index]
        self.with_fuse = stages_spec['with_fuse'][stage_index]
        self.module_type = stages_spec['module_type'][stage_index]
        self.branches_no_fuse = stages_spec['branches_no_fuse'][stage_index]
        
        modules = []
        for i in range(self.num_modules):
            # multi_scale_output is only used last module
            if not multiscale_output and i == self.num_modules - 1:
                reset_multiscale_output = False
            else:
                reset_multiscale_output = True

            modules.append(
                LiteHRModule(
                    num_branches=self.num_branches,
                    num_blocks=self.num_blocks,
                    in_channels=in_channels,
                    reduce_ratio=self.reduce_ratio,
                    module_type=self.module_type,
                    multiscale_output=reset_multiscale_output,
                    with_fuse=self.with_fuse,
                    use_bn=self.use_bn,
                    with_cp=self.with_cp,
                    branches_no_fuse=self.branches_no_fuse if i == self.num_modules - 1 else None))
            in_channels = modules[-1].in_channels            

        return nn.Sequential(*modules), in_channels

    def init_weights(self):
        """Initialize the weights in backbone.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                if m.bias is not None:
                    nn.init.normal_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward function."""
        x = self.stem(x)

        y_list = [x]
        for i in range(self.num_stages):
            x_list = []
            transition = getattr(self, 'transition{}'.format(i))
            for j in range(self.stages_spec['num_branches'][i]):
                if transition[j]:
                    if j >= len(y_list):
                        x_list.append(transition[j](y_list[-1]))
                    else:
                        x_list.append(transition[j](y_list[j]))
                else:
                    x_list.append(y_list[j])
            y_list = getattr(self, 'stage{}'.format(i))(x_list)
        return [y_list[2], y_list[0]]

    def train(self, mode=True):
        """Convert the model into training mode."""
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

def test_demo():
    base_channel = [128, 192, 256, 256]
    extra=dict(
            stem=dict(stem_channels=32, out_channels=32, expand_ratio=1, size_quarter=False),
            num_stages=3,
            stages_spec=dict(
                num_modules=(2, 4, 2),
                num_branches=(2, 3, 4),
                branches_no_fuse=(None, None, [1,3]),
                num_blocks=(2, 2, 2),
                module_type=('LITE', 'LITE', 'LITE'),
                with_fuse=(True, True, True),
                reduce_ratios=(8, 8, 8),
                num_channels=(
                    (base_channel[0], base_channel[1]),
                    (base_channel[0], base_channel[1], base_channel[2]),
                    (base_channel[0], base_channel[1], base_channel[2], base_channel[3]),
                )),
        )

    model = LiteHRNet(extra=extra, in_channels=1)
    model.eval()
    inputs = torch.rand(1, 1, 128, 128)
    level_outputs = model.forward(inputs)

    # print(model.transition1)
    # print(model.stage0)
    for output in level_outputs:
        print(output.shape)

    print('test ok')

if __name__ == '__main__':
    test_demo()