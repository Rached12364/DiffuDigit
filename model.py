"""DiffuDigit — petit U-Net conditionné (temps + chiffre) pour un modèle de diffusion (DDPM)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

T = 400                      # nombre d'étapes de bruitage
NULL_CLASS = 10              # classe "vide" pour la Classifier-Free Guidance


def schedule(device="cpu"):
    """Coefficients du processus de diffusion (planning linéaire)."""
    beta = torch.linspace(1e-4, 0.02, T, device=device)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    return beta, alpha, alpha_bar


def time_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=1)


class Block(nn.Module):
    def __init__(self, c_in, c_out, emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, c_out)
        self.norm2 = nn.GroupNorm(8, c_out)
        self.emb = nn.Linear(emb_dim, c_out)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x, e):
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.emb(e)[:, :, None, None]
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    """28x28 -> 14x14 -> 7x7 -> 14x14 -> 28x28, conditionné par l'étape t et le chiffre demandé."""
    def __init__(self, ch=64, emb_dim=128):
        super().__init__()
        self.emb_dim = emb_dim
        self.t_mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.c_emb = nn.Embedding(11, emb_dim)          # 10 chiffres + classe vide
        self.inc = nn.Conv2d(1, ch, 3, padding=1)
        self.d1 = Block(ch, ch, emb_dim)
        self.d2 = Block(ch, ch * 2, emb_dim)
        self.mid = Block(ch * 2, ch * 2, emb_dim)
        self.u2 = Block(ch * 4, ch, emb_dim)
        self.u1 = Block(ch * 2, ch, emb_dim)
        self.out = nn.Conv2d(ch, 1, 1)

    def forward(self, x, t, c):
        e = self.t_mlp(time_embedding(t, self.emb_dim)) + self.c_emb(c)
        x0 = self.inc(x)
        h1 = self.d1(x0, e)                                  # 28
        h2 = self.d2(F.avg_pool2d(h1, 2), e)                 # 14
        m = self.mid(F.avg_pool2d(h2, 2), e)                 # 7
        u = F.interpolate(m, scale_factor=2)                 # 14
        u = self.u2(torch.cat([u, h2], 1), e)
        u = F.interpolate(u, scale_factor=2)                 # 28
        u = self.u1(torch.cat([u, h1], 1), e)
        return self.out(u)                                   # bruit prédit


@torch.no_grad()
def generate(model, digit, n=1, w=3.0, keep=(399, 300, 200, 100, 50, 0), device="cpu", seed=None):
    """Génère n images du chiffre demandé. Renvoie (images finales, {étape: images})."""
    if seed is not None:
        torch.manual_seed(seed)
    beta, alpha, alpha_bar = schedule(device)
    x = torch.randn(n, 1, 28, 28, device=device)
    c = torch.full((n,), int(digit), device=device, dtype=torch.long)
    c_null = torch.full((n,), NULL_CLASS, device=device, dtype=torch.long)
    frames = {}
    for i in reversed(range(T)):
        t = torch.full((n,), i, device=device, dtype=torch.long)
        eps_c = model(x, t, c)
        eps_u = model(x, t, c_null)
        eps = (1 + w) * eps_c - w * eps_u                    # Classifier-Free Guidance
        mean = (x - beta[i] / torch.sqrt(1 - alpha_bar[i]) * eps) / torch.sqrt(alpha[i])
        x = mean + (torch.sqrt(beta[i]) * torch.randn_like(x) if i > 0 else 0)
        if i in keep:
            frames[i] = x.clamp(-1, 1).cpu()
    return x.clamp(-1, 1).cpu(), frames
