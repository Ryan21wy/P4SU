import math
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class AttnBlock(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.attn = SelfAttnBlock(dim, num_heads)
        self.mlp = FeedForward(dim, 4 * dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = x + self.conv(x)
        x = x.permute(0, 2, 1)
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SelfAttnBlock(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.size()

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LinearLayer(nn.Module):

    def __init__(self, dim_in, dim_out, activation=None, norm=None, bias=True):
        super().__init__()

        if activation:
            self.act = activation
        else:
            self.act = nn.Identity()

        if norm:
            self.norm = norm
        else:
            self.norm = nn.Identity()

        self.linear = nn.Linear(dim_in, dim_out, bias=bias)

    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class Conv1DLayer(nn.Module):

    def __init__(self, dim_in, dim_out, ks=3, s=1, pad=1, activation=None, norm=None, bias=True):
        super().__init__()

        if activation:
            self.act = activation
        else:
            self.act = nn.Identity()

        if norm:
            self.norm = norm
        else:
            self.norm = nn.Identity()

        self.Conv1D = nn.Conv1d(dim_in, dim_out, kernel_size=ks, stride=s, padding=pad, bias=bias)

    def forward(self, x):
        x = self.Conv1D(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class SpecFCN(nn.Module):
    def __init__(self, num_end=14, num_bands=156, hidden_dim=128):
        super(SpecFCN, self).__init__()

        self.encoder = nn.Sequential(
            LinearLayer(num_bands, hidden_dim * 8, activation=nn.GELU(), norm=nn.LayerNorm(hidden_dim * 8)),
            LinearLayer(hidden_dim * 8, hidden_dim * 4, activation=nn.GELU(), norm=nn.LayerNorm(hidden_dim * 4)),
            LinearLayer(hidden_dim * 4, hidden_dim * 2, activation=nn.GELU(), norm=nn.LayerNorm(hidden_dim * 2)),
            LinearLayer(hidden_dim * 2, hidden_dim, activation=nn.GELU(), norm=nn.LayerNorm(hidden_dim)),
            nn.Dropout(0.5),
        )

        self.su = nn.Linear(hidden_dim, num_end)
        self.out_norm = nn.Softmax(1)

    def Encoder(self, x):
        x = self.encoder(x)
        x = self.su(x)
        return x

    def forward(self, img):
        enc = self.Encoder(img)
        pred = self.out_norm(enc)
        return pred


class SpecCNN(nn.Module):
    def __init__(self, num_end=14, num_bands=156, hidden_dim=128):
        super(SpecCNN, self).__init__()

        f_size = math.ceil(math.ceil(math.ceil(num_bands / 3) / 3) / 3)

        self.encoder = nn.Sequential(
            Conv1DLayer(1, hidden_dim // 4, 7, 3, 3, activation=nn.GELU()),
            Conv1DLayer(hidden_dim // 4, hidden_dim // 2, 7, 3, 3, activation=nn.GELU()),
            Conv1DLayer(hidden_dim // 2, hidden_dim, 7, 3, 3, activation=nn.GELU()),
            nn.Flatten(1),
            LinearLayer(hidden_dim * f_size, hidden_dim, activation=nn.GELU(), norm=nn.LayerNorm(hidden_dim)),
            nn.Dropout(0.5),
        )

        self.su = nn.Linear(hidden_dim, num_end)
        self.out_norm = nn.Softmax(1)

    def Encoder(self, x):
        x = self.encoder(x)
        x = self.su(x)
        return x

    def forward(self, img):
        enc = self.Encoder(img)
        pred = self.out_norm(enc)
        return pred


class SpecFormer(nn.Module):
    def __init__(self, num_end=14, num_bands=156, hidden_dim=128, num_head=8):
        super(SpecFormer, self).__init__()

        self.patch_embed = nn.Sequential(
            nn.Conv1d(1, hidden_dim // 4, 7, 5, 3),
            nn.GELU(),
            nn.Conv1d(hidden_dim // 4, hidden_dim, 7, 5, 3),
        )

        f_size = math.ceil(math.ceil(num_bands / 5) / 5)

        self.patch_norm = nn.LayerNorm(hidden_dim)
        # self.pos_embed = nn.Parameter(torch.zeros(1, f_size, hidden_dim))
        # nn.init.trunc_normal_(self.pos_embed, std=.02)

        self.blocks = nn.Sequential(
            AttnBlock(hidden_dim, num_head),
            AttnBlock(hidden_dim, num_head),
            nn.Flatten(1),
            LinearLayer(hidden_dim * f_size, hidden_dim, activation=nn.GELU(), norm=nn.LayerNorm(hidden_dim)),
            nn.Dropout(0.5),
        )

        self.su = nn.Linear(hidden_dim, num_end)
        self.out_norm = nn.Softmax(1)

    def Encoder(self, x):
        x = self.patch_embed(x.unsqueeze(1))
        x = self.patch_norm(x.permute(0, 2, 1))
        x = self.blocks(x)
        x = self.su(x)
        return x

    def forward(self, img):
        enc = self.Encoder(img)
        pred = self.out_norm(enc)
        return pred