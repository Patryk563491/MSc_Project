import torch
import torch.nn as nn
import torch.nn.functional as F

import math

##########################################################################################################
# DEFINE MODEL

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)



class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)



class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()

        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffD = x2.size()[2] - x1.size()[2]
        diffH = x2.size()[3] - x1.size()[3]
        diffW = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffW // 2, diffW - diffW // 2, diffH // 2, diffH - diffH // 2, diffD // 2, diffD - diffD // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)



class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)

class PositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding3D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None, persistent=False)

    def forward(self, tensor):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, z, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device, dtype=self.inv_freq.dtype)
        pos_y = torch.arange(y, device=tensor.device, dtype=self.inv_freq.dtype)
        pos_z = torch.arange(z, device=tensor.device, dtype=self.inv_freq.dtype)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1).unsqueeze(1)
        emb_y = get_emb(sin_inp_y).unsqueeze(1)
        emb_z = get_emb(sin_inp_z)
        emb = torch.zeros(
            (x, y, z, self.channels * 3),
            device=tensor.device,
            dtype=tensor.dtype,
        )
        emb[:, :, :, : self.channels] = emb_x
        emb[:, :, :, self.channels : 2 * self.channels] = emb_y
        emb[:, :, :, 2 * self.channels :] = emb_z

        self.cached_penc = emb[None, :, :, :, :orig_ch].repeat(batch_size, 1, 1, 1, 1)
        return self.cached_penc


class PositionalEncodingPermute3D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y, z) instead of (batchsize, x, y, z, ch)
        """
        super(PositionalEncodingPermute3D, self).__init__()
        self.penc = PositionalEncoding3D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 4, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 4, 1, 2, 3)

    @property
    def org_channels(self):
        return self.penc.org_channels
    


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.positional_encoding = PositionalEncodingPermute3D(channels)

    def forward(self, q, k, v):
        """
        Args:
            q (torch.Tensor): Query tensor of shape (b, c, h, w, d)
            k (torch.Tensor): Key tensor of shape (b, c, h, w, d)
            v (torch.Tensor): Value tensor of shape (b, c, h, w, d)
        Returns:
            attn_output (torch.Tensor): Output tensor after applying self-attention (b, c, h, w, d)
        """
        # Apply positional encoding
        q = self.positional_encoding(q)
        k = self.positional_encoding(k)
        v = self.positional_encoding(v)
        
        # Extract dimensions
        b, c, h, w, d = q.shape
        
        # Flatten the spatial dimensions
        q_flat = q.view(b, c, -1)  # (b, c, h*w*d)
        k_flat = k.view(b, c, -1)  # (b, c, h*w*d)
        v_flat = v.view(b, c, -1)  # (b, c, h*w*d)

        # Transpose key for matrix multiplication
        k_flat = k_flat.transpose(1, 2)  # shape: (b, h*w*d, c)
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(q_flat, k_flat)  # (b, h*w*d, h*w*d)
        attn_scores = attn_scores / math.sqrt(c)  # Scale by the square root of the channel size
        attn_weights = F.softmax(attn_scores, dim=-1)  # Softmax to get attention weights
        
        # Weight the values by attention weights
        attn_output = torch.matmul(attn_weights, v_flat)  # (b, h*w*d, c)
        attn_output = attn_output.transpose(-2, -1)  # (b, c, h*w*d)
        
        # Unflatten the output
        attn_output = attn_output.reshape(b, c, h, w, d)  # (b, c, h, w, d)
        
        return attn_output



class CrossAttention(nn.Module):
    def __init__(self, channels_s, channels_y):
        super(CrossAttention, self).__init__()

        # Initialize positional encoding for S and Y tensors
        self.positional_encoding_s = PositionalEncodingPermute3D(channels_s)
        self.positional_encoding_y = PositionalEncodingPermute3D(channels_y)

        # Process Y tensor: 1x1x1 convolution (halve channels) + BN + ReLU
        self.conv_y = nn.Sequential(
            nn.Conv3d(channels_y, channels_y // 2, kernel_size=1),
            nn.BatchNorm3d(channels_y // 2),
            nn.ReLU(inplace=True)
        )
        
        # Process S tensor: 1x1x1 convolution (do not change channels) + BN + ReLU + 3D MaxPool (reduce dimensions to fit Y tensor)
        self.conv_s = nn.Sequential(
            nn.Conv3d(channels_s, channels_s, kernel_size=1),
            nn.BatchNorm3d(channels_s),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2)
        )

        self.self_attention = SelfAttention(channels_y // 2) # Self attention module used in cross attention

        # Sequential block for processing Z tensor: 3x3x3 conv + BN + Sigmoid activation + Upsample
        self.process_z = nn.Sequential(
            nn.Conv3d(channels_y // 2, channels_s, kernel_size=1),
            nn.BatchNorm3d(channels_s),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        )

    def forward(self, S, Y):
        """
        Args:
            S (torch.Tensor): Skip connection tensor of shape (b, c, 2h, 2w, 2d)
            Y (torch.Tensor): Deep feature map tensor of shape (b, 2c, h, w, d)
        Returns:
            out (torch.Tensor): Hadamard product of Z and S with shape (b, c, 2h, 2w, 2d)
        """

        # Apply positional encoding
        S = self.positional_encoding_s(S)
        Y = self.positional_encoding_y(Y)

        # Process Y and S tensors
        Y_processed = self.conv_y(Y)  # Shape: (b, c, h, w, d)
        S_processed = self.conv_s(S)  # Shape: (b, c, h, w, d)

        # Apply cross attention using Y as query/key and S as value 
        attention_output = self.self_attention(Y_processed, Y_processed, S_processed)
        
        # Process Z tensor
        Z = self.process_z(attention_output)
        
        # Hadamard product with S
        out = Z * S
        
        return out



class TransformerUp(nn.Module):
    def __init__(self, channels_s, channels_y):
        super(TransformerUp, self).__init__()
        self.cross_attention = CrossAttention(channels_s, channels_y)
        
        # Sequential block for processing the Y tensor
        self.process_y = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(channels_y, channels_y, kernel_size=3, padding=1),
            nn.BatchNorm3d(channels_y),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels_y, channels_s, kernel_size=1),
            nn.BatchNorm3d(channels_s),
            nn.ReLU(inplace=True)
        )

        self.double_conv = DoubleConv(channels_s * 2, channels_s)

    def forward(self, S, Y):
        """
        Args:
            S (torch.Tensor): Skip connection tensor of shape (b, c, 2h, 2w, 2d)
            Y (torch.Tensor): Deep feature map tensor of shape (b, 2c, h, w, d)
        Returns:
            out (torch.Tensor): Output tensor after concatenating cross-attention output and upsampled Y
        """
        # Apply cross-attention to get Z
        Z = self.cross_attention(S, Y)

        # Process Y: Upsample, then 3x3x3 conv, and reduce channels
        Y_processed = self.process_y(Y)  # Shape: (b, c, 2h, 2w, 2d)

        # Concatenate Z and Y_processed along the channel dimension
        out = torch.cat([Z, Y_processed], dim=1)  # Shape: (b, 2c, 2h, 2w, 2d)

        # Apply DoubleConv to the concatenated tensor
        out = self.double_conv(out)  # Shape: (b, c, 2h, 2w, 2d)

        return out



class U_Transformer(nn.Module):
    def __init__(self, in_channels = 1, num_classes=2, feat_sizes=(16, 32, 64, 128, 256)):
        super(U_Transformer, self).__init__()
        self.num_classes = num_classes
        self.feat_sizes = feat_sizes

        # Encoder (Downsampling Path)
        self.inc = DoubleConv(in_channels, feat_sizes[0])  # Initial conv block
        self.down1 = Down(feat_sizes[0], feat_sizes[1])
        self.down2 = Down(feat_sizes[1], feat_sizes[2])
        self.down3 = Down(feat_sizes[2], feat_sizes[3])
        self.down4 = Down(feat_sizes[3], feat_sizes[4])

        # Self-Attention instead of Bottleneck
        self.self_attention = SelfAttention(feat_sizes[4])

        # Decoder (Upsampling Path with Transformer)
        self.up1 = TransformerUp(feat_sizes[3], feat_sizes[4])  # feat_sizes[4] = channels_y, feat_sizes[3] = channels_s
        self.up2 = TransformerUp(feat_sizes[2], feat_sizes[3])
        self.up3 = TransformerUp(feat_sizes[1], feat_sizes[2])
        self.up4 = TransformerUp(feat_sizes[0], feat_sizes[1])

        # Output Convolution: Generate `num_classes` output channels
        self.outc = OutConv(feat_sizes[0], num_classes)

    def forward(self, x):

        # Encoder pathway
        x1 = self.inc(x)      # Shape: (1, 16, 128, 128, 400)
        x2 = self.down1(x1)   # Shape: (1, 32, 64, 64, 200)
        x3 = self.down2(x2)   # Shape: (1, 64, 32, 32, 100)
        x4 = self.down3(x3)   # Shape: (1, 128, 16, 16, 50)
        x5 = self.down4(x4)   # Shape: (1, 256, 8, 8, 25)

        # Apply Self-Attention at the deepest level
        x5 = self.self_attention(x5, x5, x5)

        # Decoder pathway
        x = self.up1(x4, x5)  # Shape: (1, 128, 16, 16, 50)
        x = self.up2(x3, x)   # Shape: (1, 64, 32, 32, 100)
        x = self.up3(x2, x)   # Shape: (1, 32, 64, 64, 200)
        x = self.up4(x1, x)   # Shape: (1, 16, 128, 128, 400)

        # Output
        output = self.outc(x)  # Shape: (1, num_classes, 128, 128, 400)
        
        return output
    