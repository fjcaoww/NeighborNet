import torch
import torch.nn as nn
import torch.nn.functional as F
from net.deformable_att import DeformableTransformer
from net.position_encoding import build_position_encoding

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class GenerateNumber(nn.Module):
    def __init__(self, in_channels, intermediate_channels=64, out_channels=1):
        super(GenerateNumber, self).__init__()
        GNlist = []
        GNlist.append(nn.Sequential(
            nn.Conv2d(in_channels, intermediate_channels, kernel_size=1),
            nn.ReLU(inplace=True)))
        GNlist.append(nn.Sequential(
            nn.Conv2d(in_channels, intermediate_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)))
        GNlist.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, intermediate_channels, kernel_size=1),
            nn.ReLU(inplace=True)))
        self.GNmodule = nn.ModuleList(GNlist)

        self.conv_out = nn.Conv2d(intermediate_channels*3, out_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        max_number = 10
        xsize = x.size()[2:]
        pixel_embedding = []
        for i in range(len(self.GNmodule) - 1):
            xx = self.GNmodule[i](x)
            pixel_embedding.append(xx)
        pixel_embedding.append(F.interpolate(self.GNmodule[-1](x), xsize, mode='bilinear', align_corners=True))
        pixel_embedding = torch.cat(pixel_embedding, dim=1)

        pixel_embedding = self.conv_out(pixel_embedding)
        number = torch.sigmoid(pixel_embedding)
        number = number * max_number
        number = number.int()
        return number


class DFMAtt(nn.Module):
    def __init__(self, in_ch, out_ch, k):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=True)
        self.k = k
        self.out_ch = out_ch
        offset_list = []
        for x in range(k):
            conv = nn.Conv2d(in_ch, 2, 1, 1, 0,bias=True)
            offset_list.append(conv)
        self.offset_conv = nn.ModuleList(offset_list)
        self.weight_conv = nn.Sequential(nn.Conv2d(in_ch, k, 1, 1, 0, bias=True), nn.Softmax(1))

        self.dropout = nn.Dropout(0.1)
        self.norm = nn.BatchNorm2d(out_ch)
        self.conv_out = nn.Conv2d(out_ch, in_ch, 1, 1, 0, bias=True)

    def forward(self, input):
        b, c, h, w = input.size()
        proj_feat = self.conv(input)
        offsets = []
        for x in range(self.k):
            flow = self.offset_conv[x](input)
            offsets.append(flow)
        # tem = self.weight_conv(input)
        offsetweights = torch.repeat_interleave(self.weight_conv(input), self.out_ch, 1)

        feats = []
        for x in range(self.k):
            flow = offsets[x]
            flow = flow.permute(0, 2, 3, 1)
            grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
            grid = torch.stack((grid_x, grid_y), 2).float()
            grid.requires_grad = False
            grid = grid.type_as(proj_feat)    # shape is [H,W,2]=[64,64,2], generate the coordinate of the input feature
            vgrid = grid + flow     # the index of the input feature + offset
            vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
            vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
            vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
            feat = F.grid_sample(proj_feat, vgrid_scaled, mode='bilinear', padding_mode='zeros', align_corners=True)
            feats.append(feat)

        feat = torch.cat(feats, 1)*offsetweights
        feat = sum(torch.split(feat, self.out_ch, 1))

        feat = proj_feat + self.dropout(feat)
        feat = self.norm(feat)

        feat = self.conv_out(feat)
        return feat


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # self.gn1 = GenerateNumber(128, 64, 1)

        # self.dfmatt1 = DFMAtt(128, 64, 4)
        # self.dfmatt2 = DFMAtt(256, 64, 4)
        # self.dfmatt3 = DFMAtt(512, 64, 4)
        # self.dfmatt4 = DFMAtt(1024, 64, 4)

        # self.dfmatt1 = DeformableTransformer(d_model=128, dim_feedforward=128, enc_n_points=4)
        # self.dfmatt2 = DeformableTransformer(d_model=256, dim_feedforward=256, enc_n_points=4)
        # self.dfmatt3 = DeformableTransformer(d_model=512, dim_feedforward=512, enc_n_points=4)
        self.dfmatt4 = DeformableTransformer(d_model=512, nhead=8, num_encoder_layers=6, dim_feedforward=1024, enc_n_points=4)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        # self.position_embed1 = build_position_encoding(mode='v2', hidden_dim=128)
        # self.position_embed2 = build_position_encoding(mode='v2', hidden_dim=256)
        self.position_embed3 = build_position_encoding(mode='v2', hidden_dim=512)
        # self.position_embed4 = build_position_encoding(mode='v2', hidden_dim=1024)

    def posi_mask(self, x, dim):
        x_fea = []
        x_posemb = []
        masks = []
        x_fea.append(x)
        masks.append(torch.zeros((x.shape[0], x.shape[2], x.shape[3]), dtype=torch.bool).cuda())
        # masks = None
        if dim == 128:
            x_posemb.append(self.position_embed1(x))
        elif dim == 256:
            x_posemb.append(self.position_embed2(x))
        elif dim == 512:
            x_posemb.append(self.position_embed3(x))
        elif dim == 1024:
            x_posemb.append(self.position_embed4(x))
        else:
            print('ERROR!')

        return x_fea, masks, x_posemb

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        # x2, masks, x_posemb = self.posi_mask(x2, dim=128)
        # x2 = self.dfmatt1(x2, masks, x_posemb)

        x3 = self.down2(x2)
        # x3, masks, x_posemb = self.posi_mask(x3, dim=256)
        # x3 = self.dfmatt2(x3, masks, x_posemb)

        x4 = self.down3(x3)
        # x4, masks, x_posemb = self.posi_mask(x4, dim=512)
        # x4 = self.dfmatt3(x4, masks, x_posemb)

        x5 = self.down4(x4)
        x5, masks, x_posemb = self.posi_mask(x5, dim=512)
        x5 = self.dfmatt4(x5, masks, x_posemb)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


if __name__ == '__main__':
    input2 = torch.randn(4, 3, 512, 512).cuda()
    net = UNet(3, 2).cuda()
    mask = net(input2)
    # print(net)
    print(mask.shape)
