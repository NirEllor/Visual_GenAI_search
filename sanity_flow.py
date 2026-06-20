import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# import your existing teacher
from models.denoiser import TeacherDenoiser


DIM = 256
N_SAMPLES = 50000
BATCH_SIZE = 512
EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------
# create synthetic target distribution
# --------------------------------------------------

torch.manual_seed(0)

mu = torch.randn(DIM) * 2.0

std = torch.exp(torch.randn(DIM) * 0.3)

x0_all = (
    torch.randn(N_SAMPLES, DIM)
    * std.unsqueeze(0)
    + mu.unsqueeze(0)
)

loader = torch.utils.data.DataLoader(
    x0_all,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
)

# --------------------------------------------------
# model
# --------------------------------------------------

model = TeacherDenoiser(latent_dim=DIM).to(DEVICE)

opt = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
)

# --------------------------------------------------
# train
# --------------------------------------------------

for epoch in range(EPOCHS):

    running = 0.0

    for x0 in loader:

        x0 = x0.to(DEVICE)

        x1 = torch.randn_like(x0)

        t = torch.rand(x0.shape[0], device=DEVICE)

        t_view = t[:, None]

        xt = (1.0 - t_view) * x0 + t_view * x1

        v_target = x1 - x0

        v_pred = model(xt, t)

        loss = F.mse_loss(v_pred, v_target)

        opt.zero_grad()
        loss.backward()
        opt.step()

        running += loss.item()

    print(
        f"epoch {epoch:03d} "
        f"loss={running/len(loader):.6f}"
    )

# --------------------------------------------------
# sample
# --------------------------------------------------

model.eval()

with torch.no_grad():
    z = torch.randn(10000, DIM, device=DEVICE)

    steps = 200
    dt = 1.0 / steps

    for step in range(steps):
        t_val = 1.0 - step * dt
        t = torch.full((z.shape[0],), t_val, device=DEVICE)

        v = model(z, t)
        z = z - dt * v

    gen = z.cpu()

# --------------------------------------------------
# compare statistics
# --------------------------------------------------

real_mean = x0_all.mean(0)
gen_mean = gen.mean(0)

real_std = x0_all.std(0)
gen_std = gen.std(0)

mean_err = (real_mean - gen_mean).abs().mean()

std_err = (real_std - gen_std).abs().mean()

print()
print("=" * 60)
print("RESULTS")
print("=" * 60)

print("mean error :", mean_err.item())
print("std error  :", std_err.item())

print("real mean norm :", real_mean.norm().item())
print("gen mean norm  :", gen_mean.norm().item())