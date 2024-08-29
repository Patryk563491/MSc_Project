import math

import torch
import torch.nn as nn
import torch.nn.functional as F

##########################################################################################################
# DEFINE U-NET TRANSFORMER MODEL

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
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffH = x2.size()[2] - x1.size()[2]
        diffW = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffW // 2, diffW - diffW // 2, diffH // 2, diffH - diffH // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None, persistent=False)

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device, dtype=self.inv_freq.dtype)
        pos_y = torch.arange(y, device=tensor.device, dtype=self.inv_freq.dtype)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1)
        emb_y = get_emb(sin_inp_y)
        emb = torch.zeros(
            (x, y, self.channels * 2),
            device=tensor.device,
            dtype=tensor.dtype,
        )
        emb[:, :, : self.channels] = emb_x
        emb[:, :, self.channels :] = emb_y

        self.cached_penc = emb[None, :, :, :orig_ch].repeat(batch_size, 1, 1, 1)
        return self.cached_penc


class PositionalEncodingPermute2D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y) instead of (batchsize, x, y, ch)
        """
        super(PositionalEncodingPermute2D, self).__init__()
        self.penc = PositionalEncoding2D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 3, 1, 2)

    @property
    def org_channels(self):
        return self.penc.org_channels


class MultiHeadAttention(nn.Module):
    def __init__(self, channels, num_heads=1):
        super(MultiHeadAttention, self).__init__()
        assert channels % num_heads == 0, "Channels must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.positional_encoding = PositionalEncodingPermute2D(channels)

        # Define linear layers for Q, K, V
        self.q_linear = nn.Linear(channels, channels)
        self.k_linear = nn.Linear(channels, channels)
        self.v_linear = nn.Linear(channels, channels)
        
        # Output linear layer
        self.out_linear = nn.Linear(channels, channels)
        
    def forward(self, q, k, v):
        """
        Args:
            q (torch.Tensor): Query tensor of shape (b, c, h, w)
            k (torch.Tensor): Key tensor of shape (b, c, h, w)
            v (torch.Tensor): Value tensor of shape (b, c, h, w)
        Returns:
            attn_output (torch.Tensor): Output tensor after applying multi-head attention (b, c, h, w)
        """
        # Apply positional encoding
        q = self.positional_encoding(q)
        k = self.positional_encoding(k)
        v = self.positional_encoding(v)
        
        b, c, h, w = q.shape
        
        # Flatten spatial dimensions and apply linear transformations
        q = q.view(b, c, -1).transpose(1, 2)  # Shape: (b, h*w, c)
        k = k.view(b, c, -1).transpose(1, 2)  # Shape: (b, h*w, c)
        v = v.view(b, c, -1).transpose(1, 2)  # Shape: (b, h*w, c)
        
        # Apply linear transformations
        q = self.q_linear(q)  # Shape: (b, h*w, c)
        k = self.k_linear(k)  # Shape: (b, h*w, c)
        v = self.v_linear(v)  # Shape: (b, h*w, c)

        # Reshape for multi-head attention: (b, num_heads, h*w, head_dim)
        q = q.view(b, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (b, num_heads, h*w, head_dim)
        k = k.view(b, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (b, num_heads, h*w, head_dim)
        v = v.view(b, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (b, num_heads, h*w, head_dim)
        
        # Transpose key for matrix multiplication
        k = k.transpose(-2, -1)  # Shape: (b, num_heads, head_dim, h*w)
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k)  # Shape: (b, num_heads, h*w, h*w)
        attn_scores = attn_scores / math.sqrt(self.head_dim)  # Scale by the square root of the head dimension
        attn_weights = F.softmax(attn_scores, dim=-1)  # Softmax to get attention weights
        
        # Weight the values by attention weights
        attn_output = torch.matmul(attn_weights, v)  # Shape: (b, num_heads, h*w, head_dim)
        
        # Concatenate heads and apply final linear transformation
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(b, -1, c)
        attn_output = self.out_linear(attn_output).reshape(b, c, h, w)
        
        return attn_output
    


class CrossAttention(nn.Module):
    def __init__(self, channels_s, channels_y, num_heads=1):
        super(CrossAttention, self).__init__()

        # Initialize positional encoding for S and Y tensors
        self.positional_encoding_s = PositionalEncodingPermute2D(channels_s)
        self.positional_encoding_y = PositionalEncodingPermute2D(channels_y)

        # Process Y tensor: 1x1 convolution (halve channels) + BN + ReLU
        self.conv_y = nn.Sequential(
            nn.Conv2d(channels_y, channels_y // 2, kernel_size=1),
            nn.BatchNorm2d(channels_y // 2),
            nn.ReLU(inplace=True)
        )
        
        # Process S tensor: 1x1 convolution (do not change channels) + BN + ReLU + 2D MaxPool (reduce dimensions to fit Y tensor)
        self.conv_s = nn.Sequential(
            nn.Conv2d(channels_s, channels_s, kernel_size=1),
            nn.BatchNorm2d(channels_s),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

        self.multi_head_attention = MultiHeadAttention(channels_y // 2, num_heads)  # Multi-head attention module

        # Sequential block for processing Z tensor: 3x3 conv + BN + Sigmoid activation + Upsample
        self.process_z = nn.Sequential(
            nn.Conv2d(channels_y // 2, channels_s, kernel_size=1),
            nn.BatchNorm2d(channels_s),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

    def forward(self, S, Y):
        """
        Args:
            S (torch.Tensor): Skip connection tensor of shape (b, c, 2h, 2w)
            Y (torch.Tensor): Deep feature map tensor of shape (b, 2c, h, w)
        Returns:
            out (torch.Tensor): Hadamard product of Z and S with shape (b, c, 2h, 2w)
        """

        # Apply positional encoding
        S = self.positional_encoding_s(S)
        Y = self.positional_encoding_y(Y)

        # Process Y and S tensors
        Y_processed = self.conv_y(Y)  # Shape: (b, c, h, w)
        S_processed = self.conv_s(S)  # Shape: (b, c, h, w)

        # Apply cross attention using Y as query/key and S as value 
        attention_output = self.multi_head_attention(Y_processed, Y_processed, S_processed)
        
        # Process Z tensor
        Z = self.process_z(attention_output)
        
        # Hadamard product with S
        out = Z * S
        
        return out


class TransformerUp(nn.Module):
    def __init__(self, channels_s, channels_y, num_heads=1):
        super(TransformerUp, self).__init__()
        self.cross_attention = CrossAttention(channels_s, channels_y, num_heads)
        
        # Sequential block for processing the Y tensor
        self.process_y = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(channels_y, channels_y, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels_y),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_y, channels_s, kernel_size=1),
            nn.BatchNorm2d(channels_s),
            nn.ReLU(inplace=True)
        )

        self.double_conv = DoubleConv(channels_s * 2, channels_s)

    def forward(self, S, Y):
        """
        Args:
            S (torch.Tensor): Skip connection tensor of shape (b, c, 2h, 2w)
            Y (torch.Tensor): Deep feature map tensor of shape (b, 2c, h, w)
        Returns:
            out (torch.Tensor): Output tensor after concatenating cross-attention output and upsampled Y
        """
        # Apply cross-attention to get Z
        Z = self.cross_attention(S, Y)

        # Process Y: Upsample, then 3x3 conv, and reduce channels
        Y_processed = self.process_y(Y)  # Shape: (b, c, 2h, 2w)

        # Concatenate Z and Y_processed along the channel dimension
        out = torch.cat([Z, Y_processed], dim=1)  # Shape: (b, 2c, 2h, 2w)

        # Apply DoubleConv to the concatenated tensor
        out = self.double_conv(out)  # Shape: (b, c, 2h, 2w)

        return out


class U_Transformer(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, feat_sizes=(16, 32, 64, 128, 256), num_heads=1):
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
        self.self_attention = MultiHeadAttention(feat_sizes[4], num_heads)

        # Decoder (Upsampling Path with Transformer)
        self.up1 = TransformerUp(feat_sizes[3], feat_sizes[4], num_heads)  # feat_sizes[4] = channels_y, feat_sizes[3] = channels_s
        self.up2 = TransformerUp(feat_sizes[2], feat_sizes[3], num_heads)
        self.up3 = TransformerUp(feat_sizes[1], feat_sizes[2], num_heads)
        self.up4 = TransformerUp(feat_sizes[0], feat_sizes[1], num_heads)

        # Output Convolution: Generate `num_classes` output channels
        self.outc = OutConv(feat_sizes[0], num_classes)

    def forward(self, x):

        # Encoder pathway
        x1 = self.inc(x)      # Shape: (1, 16, 128, 128)
        x2 = self.down1(x1)   # Shape: (1, 32, 64, 64)
        x3 = self.down2(x2)   # Shape: (1, 64, 32, 32)
        x4 = self.down3(x3)   # Shape: (1, 128, 16, 16)
        x5 = self.down4(x4)   # Shape: (1, 256, 8, 8)

        # Apply Self-Attention at the deepest level
        x5 = self.self_attention(x5, x5, x5)

        # Decoder pathway
        x = self.up1(x4, x5)  # Shape: (1, 128, 16, 16)
        x = self.up2(x3, x)   # Shape: (1, 64, 32, 32)
        x = self.up3(x2, x)   # Shape: (1, 32, 64, 64)
        x = self.up4(x1, x)   # Shape: (1, 16, 128, 128)

        # Output
        output = self.outc(x)  # Shape: (1, num_classes, 128, 128)
        
        return output
    