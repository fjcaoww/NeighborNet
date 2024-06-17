
import torch.nn as nn
import torch
import torchvision
from torchvision import models
from net.deformable_att import DeformableTransformer
from net.position_encoding import build_position_encoding
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_


class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x


class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)


class UpBlock(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose"):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """
        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)
        if self.training:
            down_x = torch.cat([down_x, down_x], dim=0)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x
    

class GenerateNumber(nn.Module):    # return a 0/1 matrix, with shape: [c,h,w]
    def __init__(self, num_point, in_channels, out_channels=1):
        super(GenerateNumber, self).__init__()
        self.num_point = num_point
        self.conv = nn.Conv2d(in_channels, num_point, kernel_size=1)
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.conv.weight.data)
        constant_(self.conv.bias.data, 0.)
        xavier_uniform_(self.conv1.weight.data)
        constant_(self.conv1.bias.data, 0.)

    def forward(self, x):
        number = self.conv(x)
        number = torch.sigmoid(number)
        # threshold
        number_threshold = self.conv1(x)
        number_threshold = torch.sigmoid(number_threshold)
        number_cat = torch.cat([number, number_threshold], dim=1)
        return number_cat


class resunet(nn.Module):

    def __init__(self, n_classes=2, sample=4):
        super().__init__()
        resnet = torchvision.models.resnet.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        down_blocks = []
        up_blocks = []
        self.inc = list(resnet.children())[0]
        self.inbn = list(resnet.children())[1]
        self.inrelu = list(resnet.children())[2]
        self.input_pool = list(resnet.children())[3]

        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)

        self.distance = [1.0, 1.0, 1.0, 1.0]
        num_point = 16
        num_head = 4    # points = num_point * num_head
        # sample = 4
        en_trans = 1.5    # encoder transformer
        self.dfmatt1 = DeformableTransformer(d_model=64, nhead=num_head, num_encoder_layers=1, dim_feedforward=64, enc_n_points=num_point, num_feature_levels=en_trans)
        self.dfmatt2 = DeformableTransformer(d_model=128, nhead=num_head, num_encoder_layers=1, dim_feedforward=128, enc_n_points=num_point, num_feature_levels=en_trans)
        self.dfmatt3 = DeformableTransformer(d_model=256, nhead=num_head, num_encoder_layers=1, dim_feedforward=256, enc_n_points=num_point, num_feature_levels=en_trans)
        self.dfmatt4 = DeformableTransformer(d_model=512, nhead=num_head, num_encoder_layers=1, dim_feedforward=512, enc_n_points=num_point, num_feature_levels=en_trans)
        self.dfmatt5 = DeformableTransformer(d_model=256, nhead=num_head, num_encoder_layers=1, dim_feedforward=256, enc_n_points=num_point)
        self.dfmatt6 = DeformableTransformer(d_model=128, nhead=num_head, num_encoder_layers=1, dim_feedforward=128, enc_n_points=num_point)
        self.dfmatt7 = DeformableTransformer(d_model=64, nhead=num_head, num_encoder_layers=1, dim_feedforward=64, enc_n_points=num_point)

        self.position_embed1 = build_position_encoding(mode='v2', hidden_dim=64)
        self.position_embed2 = build_position_encoding(mode='v2', hidden_dim=128)
        self.position_embed3 = build_position_encoding(mode='v2', hidden_dim=256)
        self.position_embed4 = build_position_encoding(mode='v2', hidden_dim=512)
        self.position_embed1up = build_position_encoding(mode='v2', hidden_dim=256)
        self.position_embed2up = build_position_encoding(mode='v2', hidden_dim=128)
        self.position_embed3up = build_position_encoding(mode='v2', hidden_dim=64)

        self.gen_num1 = GenerateNumber(num_point*num_head, in_channels=64)
        self.gen_num2 = GenerateNumber(num_point*num_head, in_channels=128)
        self.gen_num3 = GenerateNumber(num_point*num_head, in_channels=256)
        self.gen_num4 = GenerateNumber(num_point*num_head, in_channels=512)
        self.gen_num5 = GenerateNumber(num_point*num_head, in_channels=256)
        self.gen_num6 = GenerateNumber(num_point*num_head, in_channels=128)
        self.gen_num7 = GenerateNumber(num_point*num_head, in_channels=64)

        use_bf = 1
        self.use_bf = use_bf
        if self.use_bf:
            self.dfmatt4bf = DeformableTransformer(d_model=512, nhead=num_head, num_encoder_layers=1, dim_feedforward=512, num_feature_levels=sample, enc_n_points=num_point)
            self.position_embed4bf = build_position_encoding(mode='v2', hidden_dim=512)
            self.gen_num4bf = GenerateNumber(num_point*num_head*sample, in_channels=512)

        up_blocks.append(UpBlock(512, 256))
        up_blocks.append(UpBlock(256, 128))
        up_blocks.append(UpBlock(128, 64))
        self.up_blocks = nn.ModuleList(up_blocks)

        self.out = nn.Conv2d(64, n_classes, kernel_size=1, stride=1)
        self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)

    def posi_mask(self, x, dim, up=False, bf=False):
        x_fea = []
        x_posemb = []
        masks = []

        if up == True:
            if dim == 64:
                x_posemb.append(self.position_embed3up(x))
            elif dim == 128:
                x_posemb.append(self.position_embed2up(x))
            elif dim == 256:
                x_posemb.append(self.position_embed1up(x))
            else:
                print('ERROR!')
        elif bf == True:
            if dim == 512:
                x_posemb.append(self.position_embed4bf(x))
            else:
                print('ERROR!')
        else:
            if dim == 64:
                x_posemb.append(self.position_embed1(x))
            elif dim == 128:
                x_posemb.append(self.position_embed2(x))
            elif dim == 256:
                x_posemb.append(self.position_embed3(x))
            elif dim == 512:
                x_posemb.append(self.position_embed4(x))
            else:
                print('ERROR!')

        if bf == True:
            bs = x.shape[0]
            x_posemb_new = []
            for s in range(0, bs):
                x_fea.append(x[s:s+1, :, :, :])
                masks.append(torch.zeros((1, x.shape[2], x.shape[3]), dtype=torch.bool).cuda())
                x_posemb_new.append(x_posemb[0][s:s+1, :, :, :])
            return x_fea, masks, x_posemb_new

        x_fea.append(x)
        masks.append(torch.zeros((x.shape[0], x.shape[2], x.shape[3]), dtype=torch.bool).cuda())
        return x_fea, masks, x_posemb

    def forward(self, x):
        x0 = self.inc(x)
        x0 = self.inbn(x0)
        x0 = self.inrelu(x0)

        x1 = self.input_pool(x0)
        x1 = self.down_blocks[0](x1)
        mask_num1 = self.gen_num1(x1)
        x1_res = x1
        x1, masks1, x_posemb1 = self.posi_mask(x1, dim=64)
        x1 = self.dfmatt1(mask_num1, self.distance[0], x1, masks1, x_posemb1)
        x1 = x1 + x1_res

        x2 = self.down_blocks[1](x1)
        mask_num2 = self.gen_num2(x2)
        x2_res = x2
        x2, masks2, x_posemb2 = self.posi_mask(x2, dim=128)
        x2 = self.dfmatt2(mask_num2, self.distance[1], x2, masks2, x_posemb2)
        x2 = x2 + x2_res

        x3 = self.down_blocks[2](x2)
        mask_num3 = self.gen_num3(x3)
        x3_res = x3
        x3, masks3, x_posemb3 = self.posi_mask(x3, dim=256)
        x3 = self.dfmatt3(mask_num3, self.distance[2], x3, masks3, x_posemb3)
        x3 = x3 + x3_res

        x4 = self.down_blocks[3](x3)
        # first deformable transformer
        mask_num4 = self.gen_num4(x4)
        x4_res = x4
        x4, masks4, x_posemb4 = self.posi_mask(x4, dim=512)
        x4 = self.dfmatt4(mask_num4, self.distance[3], x4, masks4, x_posemb4)
        x4 = x4 + x4_res
        # second cross image transformer
        if self.training:
            old_x4 = x4
            mask_num4bf = self.gen_num4bf(x4)
            x4_res = x4
            x4, masks4bf, x_posemb4bf = self.posi_mask(x4, dim=512, bf=True)
            x4 = self.dfmatt4bf(mask_num4bf, self.distance[3], x4, masks4bf, x_posemb4bf) + x4_res
            x4 = torch.cat([old_x4, x4], dim=0)

        # ############################################# decoder #########################################
        xu1 = self.up_blocks[0](x4, x3)
        mask_num5 = self.gen_num5(xu1)
        xu1_res = xu1
        xu1, masks5, x_posemb5 = self.posi_mask(xu1, dim=256, up=True)
        xu1, preds_ds5 = self.dfmatt5(mask_num5, self.distance[2], xu1, masks5, x_posemb5)
        xu1 = xu1 + xu1_res

        xu2 = self.up_blocks[1](xu1, x2)
        mask_num6 = self.gen_num6(xu2)
        xu2_res = xu2
        xu2, masks6, x_posemb6 = self.posi_mask(xu2, dim=128, up=True)
        xu2, preds_ds6 = self.dfmatt6(mask_num6, self.distance[1], xu2, masks6, x_posemb6)
        xu2 = xu2 + xu2_res

        xu3 = self.up_blocks[2](xu2, x1)
        mask_num7 = self.gen_num7(xu3)
        xu3_res = xu3
        xu3, masks7, x_posemb7 = self.posi_mask(xu3, dim=64, up=True)
        xu3, preds_ds7 = self.dfmatt7(mask_num7, self.distance[0], xu3, masks7, x_posemb7)
        xu3 = xu3 + xu3_res

        x = self.out(xu3)
        x = self.upsample4(x)

        return [x, preds_ds7, preds_ds6, preds_ds5]


if __name__ == '__main__':
    input1 = torch.randn(8, 3, 512, 512).cuda()
    net = resunet(sample=8).cuda()
    # net.eval()
    # [x, xu3, xu2, xu1] = net(input1)
    # print(x.shape, xu3.shape, xu2.shape, xu1.shape)

    x = net(input1)
    print(x.shape)







