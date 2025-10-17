import os
import torch

from utils import plot_variable_guidance_scale
from nn.networks import DiffusionNet, FlowMatchingNet, Unet, VisionTransformer
from torchvision.utils import save_image

if not os.path.isdir('./imgs'):
    os.mkdir('./imgs')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

img_size = 128
model = FlowMatchingNet(
    Unet(64, 128, 256, 512, 1024, embed_dim=64, residual_depth=3),
    #VisionTransformer(
    #    img_size=img_size, in_dim=1, depth_encoder=12,
    #    n_heads=16, patch_size=16, embed_dim=512,  # or 1024
    #    dropout_rate=0.1,
    #),
    #1000,
    device=device,
)
model.load_state_dict(torch.load('./params.pt'))
model.eval()
print('Model loaded.')

label = torch.tensor([1], dtype=torch.long, device=device)
if model.__class__.__name__ == 'FlowMatchingNet':
    timesteps = {5, 10, 30, 50, 100}
    n_steps = 50
else:
    timesteps = {500, 1000}
    n_steps = 1000

for n_steps in timesteps:
    print(f'Sampling at {n_steps} steps...')
    img = model.sample_img(img_size=img_size, label=label, n_steps=n_steps)
    img.clamp(0, 1)
    # img = img + 1 / 2
    save_image(img, f'./imgs/sample_img_{n_steps}.pdf')

labels = (0, 1)
guidance_scale = (0., 0.5, 1., 2., 3., 5., 10.)
print('Plotting guidance scales...')
plot_variable_guidance_scale(
    model, img_size=img_size, n_steps=n_steps, labels=labels, guidance_scale=guidance_scale, do_save=True, path='./imgs'
)
