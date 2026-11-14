"""Entraîne DiffuDigit sur MNIST.   python train.py --epochs 10"""
import argparse, os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from model import UNet, schedule, generate, T, NULL_CLASS

p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=10)
p.add_argument("--batch", type=int, default=128)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--resume", action="store_true", help="reprendre depuis diffudigit.pt")
p.add_argument("--fake", action="store_true", help="données aléatoires (test rapide sans téléchargement)")
args = p.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Appareil :", device)
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
if args.fake:
    data = datasets.FakeData(512, (1, 28, 28), 10, transform=tf)
else:
    data = datasets.MNIST("data", train=True, download=True, transform=tf)
loader = DataLoader(data, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)

model = UNet().to(device)
if args.resume and os.path.exists("diffudigit.pt"):
    model.load_state_dict(torch.load("diffudigit.pt", map_location=device))
    print("Reprise depuis diffudigit.pt")
opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
_, _, alpha_bar = schedule(device)
os.makedirs("samples", exist_ok=True)
print("Paramètres :", sum(p.numel() for p in model.parameters()) // 1000, "k")

for epoch in range(1, args.epochs + 1):
    model.train(); total = 0
    for step, (x0, y) in enumerate(loader):
        x0, y = x0.to(device), torch.as_tensor(y).to(device)
        y = torch.where(torch.rand_like(y, dtype=torch.float) < 0.1, torch.full_like(y, NULL_CLASS), y)  # 10 % sans condition
        t = torch.randint(0, T, (x0.size(0),), device=device)
        noise = torch.randn_like(x0)
        ab = alpha_bar[t][:, None, None, None]
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * noise        # bruitage direct à l'étape t
        loss = F.mse_loss(model(xt, t, y), noise)            # le modèle apprend à prédire le bruit
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item()
        if step % 100 == 0:
            print(f"époque {epoch}  pas {step}/{len(loader)}  perte {loss.item():.4f}")
    print(f"== époque {epoch} terminée, perte moyenne {total / len(loader):.4f}")
    torch.save(model.state_dict(), "diffudigit.pt")
    model.eval()
    grid = torch.cat([generate(model, d, n=4, w=2.0, device=device)[0] for d in range(10)])
    save_image(grid, f"samples/epoque_{epoch:02d}.png", nrow=4, normalize=True, value_range=(-1, 1))
print("Modèle enregistré : diffudigit.pt")
