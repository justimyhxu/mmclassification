import torch
import logging
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.utils import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional
import numpy as np
from ..builder import BACKBONES
from mmcv.runner import load_checkpoint
from mmcv.utils.parrots_wrapper import _BatchNorm
from mmcv.cnn import (ConvModule, build_conv_layer, build_norm_layer,
                      constant_init, kaiming_init)


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, padding=None) -> nn.Conv2d:
    """3x3 convolution with padding"""
    if padding is None:
        padding = dilation
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, groups=groups, dilation=dilation, bias=True)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, bias=None) -> nn.Conv2d:
    """1x1 convolution"""
    if bias is None:
       bias = True
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        if stride == 1:
            self.conv1 = conv3x3(inplanes, planes, stride)
        elif stride == 2:
            self.conv1 = conv3x3(inplanes, planes, stride, padding=0)
        self.bn1 = norm_layer(planes)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        self.pth_to_tf_var_mapping = {}
        map_dict = dict(conv1='Conva', conv2='Convb') 
        for conv_name, bn_name in zip(['conv1', 'conv2'], ['bn1', 'bn2']):
            map_name = map_dict.get(conv_name)
            self.pth_to_tf_var_mapping[f'{conv_name}.weight'] = f'{map_name}/weight'
            self.pth_to_tf_var_mapping[f'{conv_name}.bias'] = f'{map_name}/bias' 
            self.pth_to_tf_var_mapping[f'{bn_name}.weight'] = f'{map_name}/batch_norm/gamma'
            self.pth_to_tf_var_mapping[f'{bn_name}.bias'] = f'{map_name}/batch_norm/beta'
            self.pth_to_tf_var_mapping[f'{bn_name}.running_var'] = f'{map_name}/batch_norm/moving_variance'
            self.pth_to_tf_var_mapping[f'{bn_name}.running_mean'] = f'{map_name}/batch_norm/moving_mean'
        if downsample is not None:
            self.pth_to_tf_var_mapping[f'downsample.0.weight'] = (f'Shortcut/weight')
            self.pth_to_tf_var_mapping[f'downsample.1.weight'] = (f'Shortcut/batch_norm/gamma')
            self.pth_to_tf_var_mapping[f'downsample.1.bias'] = (f'Shortcut/batch_norm/beta')
            self.pth_to_tf_var_mapping[f'downsample.1.running_var'] = (f'Shortcut/batch_norm/moving_variance')
            self.pth_to_tf_var_mapping[f'downsample.1.running_mean'] = (f'Shortcut/batch_norm/moving_mean')

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        if self.stride == 2:
            out = F.pad(x, pad=(0,1,0,1))
        elif self.stride == 1:
            out = x 
        out = self.conv1(out)
            
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        if stride == 2:
            self.conv2 = conv3x3(width, width, stride, groups, dilation, padding=0)
        elif stride == 1:
            self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=False)
        # self.relu = nn.Identity()
        self.downsample = downsample
        self.stride = stride

        self.pth_to_tf_var_mapping = {}
        map_dict = dict(conv1='Conva', conv2='Convb', conv3='Convc')
        for conv_name, bn_name in zip(['conv1', 'conv2', 'conv3'], ['bn1', 'bn2', 'bn3']):
            map_name = map_dict.get(conv_name)
            self.pth_to_tf_var_mapping[f'{conv_name}.weight'] = f'{map_name}/weight'
            self.pth_to_tf_var_mapping[f'{conv_name}.bias'] = f'{map_name}/bias'
            self.pth_to_tf_var_mapping[f'{bn_name}.weight'] = f'{map_name}/batch_norm/gamma'
            self.pth_to_tf_var_mapping[f'{bn_name}.bias'] = f'{map_name}/batch_norm/beta'
            self.pth_to_tf_var_mapping[f'{bn_name}.running_var'] = f'{map_name}/batch_norm/moving_variance'
            self.pth_to_tf_var_mapping[f'{bn_name}.running_mean'] = f'{map_name}/batch_norm/moving_mean'
        if downsample is not None:
            self.pth_to_tf_var_mapping[f'downsample.0.weight'] = (f'Shortcut/weight')
            self.pth_to_tf_var_mapping[f'downsample.1.weight'] = (f'Shortcut/batch_norm/gamma')
            self.pth_to_tf_var_mapping[f'downsample.1.bias'] = (f'Shortcut/batch_norm/beta')
            self.pth_to_tf_var_mapping[f'downsample.1.running_var'] = (f'Shortcut/batch_norm/moving_variance')
            self.pth_to_tf_var_mapping[f'downsample.1.running_mean'] = (f'Shortcut/batch_norm/moving_mean')

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        if self.stride == 2:
            out = F.pad(out, (0,1,0,1))
            out = self.conv2(out)
        elif self.stride == 1:
            out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class FPN(nn.Module):
    def __init__(self,
                 start_level=3,
                 length = 6,
                 fin=[64,64,128,256,512,512],
                 fout=512,
                 ):
        super(FPN, self).__init__()
        self.lconv_list = nn.ModuleList()
        self.fconv_list = nn.ModuleList()
        self.pth_to_tf_var_mapping = {}
        for i in range(start_level, length):
            self.lconv_list.append(nn.Conv2d(fin[i], fout, kernel_size=3, bias=True, padding=1))
            self.pth_to_tf_var_mapping[f'lconv_list.{i-start_level}.weight'] = f'lconv_{i}/weight'
            self.pth_to_tf_var_mapping[f'lconv_list.{i-start_level}.bias'] = f'lconv_{i}/bias'

            self.fconv_list.append(nn.Conv2d(fout, fout, kernel_size=3, bias=True, padding=1))
            self.pth_to_tf_var_mapping[f'fconv_list.{i-start_level}.weight'] = f'fconv_{i-start_level}/weight'
            self.pth_to_tf_var_mapping[f'fconv_list.{i-start_level}.bias'] = f'fconv_{i-start_level}/bias'

        self.start_level = start_level  

    def forward(self, inputs,):
        laterals = []
        for i in range(self.start_level, len(inputs)):        
            laterals.append(self.lconv_list[i-self.start_level](inputs[i]))
        flevel = len(laterals)
        for i in range(flevel-1, 0, -1):
            laterals[i-1] += F.interpolate(laterals[i], mode='nearest', scale_factor=2) + laterals[i-1]
        outputs = []
        for i in range(flevel):
            outputs.append(self.fconv_list[i](laterals[i]))
        return outputs

class DFuse(nn.Module):
    def __init__(self,
                 length=3,
                 fin=512,
                 fout=512,
                 fuse=True):
        super(DFuse, self).__init__()
        self.fuse_conv_list = nn.ModuleList()
        self.length = length
        self.fuse = fuse
        self.pth_to_tf_var_mapping = {}
        for i in range(length):
            fuse_conv = nn.Conv2d(fin, fout, kernel_size=3, padding=1, bias=True)
            self.fuse_conv_list.append(fuse_conv)
            self.pth_to_tf_var_mapping[f'fuse_conv_list.{i}.weight'] = f'fuse_conv_{i}/weight'            
            self.pth_to_tf_var_mapping[f'fuse_conv_list.{i}.bias'] = f'fuse_conv_{i}/bias'            

    def forward(self, inputs):
        assert len(inputs) == self.length, 'input length must be equal with self.length' 
        for i in range(len(inputs)-1):        
            for j in range(0, len(inputs)-1-i):            
                inputs[j] = F.avg_pool2d(inputs[j], kernel_size=2, stride=2)   

        for i in range(len(inputs)):
            inputs[i] = self.fuse_conv_list[i](inputs[i])

        if self.fuse:
            for i in range(len(inputs)-1):
                inputs[i] = inputs[i] + inputs[-1]     
        return inputs

class CodeHead(nn.Module):
    def __init__(
        self,
        in_planes,
        latent_size,
        norm_layer=nn.BatchNorm2d):
        super().__init__()  
        self.fc = nn.Linear(in_planes, latent_size, bias=True)
        self.norm = norm_layer(latent_size)

        self.pth_to_tf_var_mapping = {}

        self.pth_to_tf_var_mapping[f'fc.weight'] = f'weight'            
        self.pth_to_tf_var_mapping[f'fc.bias'] = f'bias'            
        self.pth_to_tf_var_mapping[f'norm.weight'] = f'batch_norm/gamma'
        self.pth_to_tf_var_mapping[f'norm.bias'] = f'batch_norm/beta'
        self.pth_to_tf_var_mapping[f'norm.running_var'] = f'batch_norm/moving_variance'
        self.pth_to_tf_var_mapping[f'norm.running_mean'] = f'batch_norm/moving_mean'

    def forward(
        self,
        input):
        if len(input.shape) > 2:
            input = input.view(input.shape[0], -1)
        latent = self.fc(input)
        latent = latent[..., None, None]
        latent = self.norm(latent) 

        return latent

@BACKBONES.register_module()
class ResNetOfficial(nn.Module):
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2, 1)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3, 1)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }
    def __init__(
        self,
        depth = 18,
        # block: Type[Union[BasicBlock, Bottleneck]],
        # layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        filter_max: int = 512, 
        with_fpn = True,
        with_ds_fuse = True,
        multi_level = True,
        frozen = True,
        norm_eval = True,
        pretrained = None,
    ) -> None:
        super(ResNetOfficial, self).__init__()
        block, layers = self.arch_settings[depth]
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.frozen = frozen
        self.norm_eval = norm_eval

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.pth_to_tf_var_mapping = {}
        
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=0,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.pth_to_tf_var_mapping['conv1.weight'] = ('Conv0/weight')
        self.pth_to_tf_var_mapping['bn1.weight'] = ('Conv0/batch_norm/gamma')
        self.pth_to_tf_var_mapping['bn1.bias'] = ('Conv0/batch_norm/beta')
        self.pth_to_tf_var_mapping['bn1.running_mean'] = ('Conv0/batch_norm/moving_mean')
        self.pth_to_tf_var_mapping['bn1.running_var'] = ('Conv0/batch_norm/moving_variance')
        

        self.layer1 = self._make_layer(block, 64, layers[0])
             
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.layer5 = self._make_layer(block, 512, layers[4], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        
        for layer_idx,layer in enumerate(layers):
            layer_idx += 1
            llayer = getattr(self, f'layer{layer_idx}')
            blocks = [block for block in llayer.children()]
            for block_idx, block in enumerate(blocks):
                pth_pattern = f'layer{layer_idx}.{block_idx}'
                if layer_idx <=  4:
                    tf_pattern = f'stage{layer_idx}_unit{block_idx+1}'
                else:
                    tf_pattern = f'stage{layer_idx}'
                for key, val in block.pth_to_tf_var_mapping.items():
                    self.pth_to_tf_var_mapping[f'{pth_pattern}.{key}'] = (f'{tf_pattern}/{val}') 

        self.with_fpn = with_fpn
        self.with_ds_fuse = with_ds_fuse
        self.multi_level = multi_level
        if self.with_fpn:
            self.fpn = FPN(fout=512)
            for key, val in self.fpn.pth_to_tf_var_mapping.items():
                self.pth_to_tf_var_mapping[f'fpn.{key}'] = (f'fpn/{val}')            

        # self.dfuse = DFuse(length=3, fin=512, fout=512, fuse=self.with_ds_fuse)
        # for key, val in self.dfuse.pth_to_tf_var_mapping.items():
        #     self.pth_to_tf_var_mapping[f'dfuse.{key}'] = (f'{val}')
        
        if self.multi_level:
            max_length = 1024
            self.dsize = [max_length] * 8 + [max_length // 2] * 2 + [max_length // 4] * 2 + [max_length // 8] * 2
            self.low_level = CodeHead(in_planes=2048*4*4, latent_size=sum(self.dsize[:4])) 
            self.mid_level = CodeHead(in_planes=2048*4*4, latent_size=sum(self.dsize[4:8])) 
            self.high_level = CodeHead(in_planes=2048*4*4, latent_size=sum(self.dsize[8:])) 
            
            level_mapping = dict(low_level='LowLevel',
                                 mid_level='MediaLevel',
                                 high_level='HighLevel')
            for level_key, level_val in level_mapping.items():
                level_block = getattr(self, level_key)
                for key, val in level_block.pth_to_tf_var_mapping.items():
                    self.pth_to_tf_var_mapping[f'{level_key}.{key}'] = (f'{level_val}/{val}')
        self.init_weights(pretrained=pretrained, frozen=frozen) 
        # pth_var_keys = list(self.pth_to_tf_var_mapping.keys())
        # weights_keys = list(self.state_dict().keys())
        # self.avBgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

        # # Zero-initialize the last BN in each residual branch,
        # # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # if zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, Bottleneck):
        #             nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
        #         elif isinstance(m, BasicBlock):
        #             nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion or True:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def init_weights(self, pretrained=None, frozen=True):
        # super(ResNetEncoder, self).init_weights(pretrained)

        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
            if frozen:
                logger.info('Froze backbone weights!')
                for name, param in self.named_parameters():
                    param.requires_grad = False        
        elif pretrained is None:
            print('Random Initialize Weights!')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
            if frozen:
                print('Froze backbone weights!')
                for name, param in self.named_parameters():
                    param.requires_grad = False
            # use default initializer or customized initializer in subclasses
        else:
            raise TypeError('pretrained must be a str or None.'
                            f' But received {type(pretrained)}.')

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        init_inputs= x
        x = F.pad(x,pad=(2,3,2,3))
        x = self.conv1(x)
        conv1 = x
        x = self.bn1(x)
        bn1 = x
        x = self.relu(x)
        relu1 = x
        x = F.pad(x, pad=(0,1,0,1), value=-np.inf)
        x = self.maxpool(x)
        down1 = x
        res1 = x

        x = self.layer1(x)
        res2 = x

        # layer1_blocks = [block for block in self.layer1.children()]
        # for idx, bk in enumerate(layer1_blocks):
        #     x = bk(x)
        #     if idx == 0:
        #         res1 = x
        x = self.layer2(x)
        res3 = x
        x = self.layer3(x)
        res4 = x
        x = self.layer4(x)
        res5 = x
        x = self.layer5(x)
        res6 = x
       
        # inputs = (res1, res2, res3, res4, res5, res6)
        # if self.with_fpn:
        #     inputs = self.fpn(inputs)
        # else:
        #     inputs = (res4, res5, res6)

        # inputs = self.dfuse(inputs)
        # res4, res5, res6 = inputs
        if self.multi_level:
            latent_w0 = self.low_level(res6)
            latent_w0 = latent_w0.reshape(-1, 4, self.dsize[0])            
 
            latent_w1 = self.mid_level(res6)
            latent_w1 = latent_w1.reshape(-1, 4, self.dsize[4])            

            latent_w2 = self.high_level(res6)
            latent_w20 = latent_w2[:, :sum(self.dsize[8:10])].reshape(-1, 2, self.dsize[8])
            latent_w21 = latent_w2[:, sum(self.dsize[8:10]):sum(self.dsize[8:12])].reshape(-1, 2, self.dsize[10])
            latent_w22 = latent_w2[:, sum(self.dsize[8:12]):].reshape(-1, 2, self.dsize[12])

            # tile tensor
            latent_w20 = latent_w20.repeat(1, 1, self.dsize[0]//self.dsize[8]) 
            latent_w21 = latent_w21.repeat(1, 1, self.dsize[0]//self.dsize[10]) 
            latent_w22 = latent_w22.repeat(1, 1, self.dsize[0]//self.dsize[12]) 
            latent_w2 = torch.cat([latent_w20, latent_w21, latent_w22], dim=1)

            # group adain
            latent_w = torch.cat([latent_w0, latent_w1, latent_w2], dim=1)
        else:
            latent_w = init_inputs
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)
        # print(res1.mean(), res2.mean(), res3.mean(), res4.mean(), res5.mean(), res6.mean())
        if True:
            return init_inputs, conv1, bn1, relu1, down1, res1, res2, res3, res4, res5, res6, latent_w
        else:
            return latent_w

    def forward(self, x: Tensor) -> Tensor:
        
        init_inputs, conv1, bn1, relu1, down1, res1, res2, res3, res4, res5, res6, latent_w =  self._forward_impl(x)
        res6_gp = F.adaptive_avg_pool2d(res6, (1,1))
        neck_f = torch.flatten(res6_gp, 1) 
        return neck_f

    def train(self, mode=True):
        super(ResNetOfficial, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
ResNet = ResNetOfficial
def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2, 1], pretrained, progress,
                   **kwargs)


def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3, 1], pretrained, progress,
                   **kwargs)

if __name__ == '__main__':
    # model = resnet18(with_ds_fuse=False).cuda()
    model = resnet50(with_ds_fuse=False,
                     with_fpn=False,
                     filter_max=2048).cuda()
    data = torch.randn(2,3,256,256).cuda()
    output = model(data)
