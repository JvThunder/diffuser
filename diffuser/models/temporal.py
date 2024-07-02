import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
from einops import rearrange
import pdb
from torch.distributions import Bernoulli

from .helpers import (
    SinusoidalPosEmb,
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.InstanceNorm2d(dim, affine = True)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 128):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class GlobalMixing(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 128):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)

class FiLM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FiLM, self).__init__()
        self.gamma_transform = nn.Linear(in_channels, out_channels)
        self.beta_transform = nn.Linear(in_channels, out_channels)
        self.out_channels = out_channels

    def forward(self, x, t):
        # Transform t to produce gamma and beta
        gamma = self.gamma_transform(t)
        beta = self.beta_transform(t)
        # Reshape gamma and beta to match the spatial dimensions of x
        gamma = gamma.view(-1, self.out_channels, 1)  # Reshape to [batch, channels, 1]
        beta = beta.view(-1, self.out_channels, 1)  # Reshape to [batch, channels, 1]
        return gamma * x + beta

class ResidualTemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim, kernel_size, mish, film=False):
        super().__init__()
        self.film = film
        self.conv1 = Conv1dBlock(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.act = nn.Mish() if mish else nn.SiLU()
        self.conv2 = Conv1dBlock(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.film1 = FiLM(embed_dim, out_channels)
        self.film2 = FiLM(embed_dim, out_channels)
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
        if in_channels != out_channels else nn.Identity()

        self.time_mlp = nn.Sequential(
            self.act,
            nn.Linear(embed_dim, out_channels),
            Rearrange('batch t -> batch t 1'),
        )    
    
    def forward(self, x, t):
        if self.film:
            out = self.conv1(x)
            out = self.film1(out, t)  # Apply FiLM with `t`
            out = self.act(out)
            out = self.conv2(out)
            out = self.film2(out, t)  # Apply FiLM with `t` again
            return out + self.residual_conv(x)
        else:
            out = self.conv1(x) + self.time_mlp(t)
            out = self.conv2(out)
            return out + self.residual_conv(x)

class TemporalUnet(nn.Module):

    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        dim=128,
        dim_mults=(1, 2, 4, 8),
        returns_condition=False,
        condition_dropout=0.1,
        calc_energy=False,
        kernel_size=5,
        film=False,
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {in_out}')

        if calc_energy:
            mish = False
            act_fn = nn.SiLU()
        else:
            mish = True
            act_fn = nn.Mish()

        self.time_dim = dim
        self.returns_dim = dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            act_fn,
            nn.Linear(dim * 4, dim),
        )

        self.returns_condition = returns_condition
        self.condition_dropout = condition_dropout
        self.calc_energy = calc_energy

        self.cond_mlp = nn.Sequential(
                        nn.Linear(cond_dim, dim),
                        act_fn,
                        nn.Linear(dim, dim * 4),
                        act_fn,
                        nn.Linear(dim * 4, dim),
                    )

        self.returns_mlp = nn.Sequential(
                    nn.Linear(1, dim),
                    act_fn,
                    nn.Linear(dim, dim * 4),
                    act_fn,
                    nn.Linear(dim * 4, dim),
                )
        self.mask_dist = Bernoulli(probs=1-self.condition_dropout)
        embed_dim = 3*dim

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=embed_dim, kernel_size=kernel_size, mish=mish, film=film),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=embed_dim, kernel_size=kernel_size, mish=mish, film=film),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=embed_dim, kernel_size=kernel_size, mish=mish, film=film)
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=embed_dim, kernel_size=kernel_size, mish=mish, film=film)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=embed_dim, kernel_size=kernel_size, mish=mish, film=film),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=embed_dim, kernel_size=kernel_size, mish=mish, film=film),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=kernel_size, mish=mish),
            nn.Conv1d(dim, transition_dim, 1),
        )

    def forward(self, x, cond, time, returns, use_dropout=False, force_dropout=False):
        '''
            x : [ batch x horizon x transition ]
            returns : [batch x horizon]
        '''
        x = einops.rearrange(x, 'b h t -> b t h')
        t = self.time_mlp(time)

        if self.returns_condition:
            assert returns is not None
            returns_embed = self.returns_mlp(returns)
            if use_dropout:
                mask = self.mask_dist.sample(sample_shape=(returns_embed.size(0), 1)).to(returns_embed.device)
                returns_embed = mask*returns_embed
            if force_dropout:
                returns_embed = 0*returns_embed
            t = torch.cat([t, returns_embed], dim=-1)

        assert cond is not None  # Assuming 'cond' is another input
        cond_embed = self.cond_mlp(cond)  # Assuming you have a similar MLP for 'cond'
        t = torch.cat([t, cond_embed], dim=-1)

        h = []

        for resnet, resnet2, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')
        return x