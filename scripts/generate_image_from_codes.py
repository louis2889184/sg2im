import os
import re
import json
import torch
import argparse
import torch.nn.functional as F
from torchvision.utils import save_image

from dvae import DVAE

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='sg2im-models/vg128.pt')
parser.add_argument('--tokens_json', default='scene_graphs/figure_6_sheep.json')
parser.add_argument('--output_dir', default='outputs')
parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
args = parser.parse_args()

dim = 512
model = DVAE(n_embed=512)
ckpt = torch.load(args.checkpoint)

if 'model' in ckpt:
    ckpt = ckpt['model']

updated_checkpoint = {}

for k, v in ckpt.items():
    if k.startswith("module"):
        updated_checkpoint[k[7:]] = v
    else:
        updated_checkpoint[k] = v

model.load_state_dict(updated_checkpoint)

model.to(args.device)
model.eval()

with open(args.tokens_json, 'r') as f:
    tokens_list = json.load(f)

    if isinstance(tokens_list, str):
        tokens_list = [tokens_list]

inputs = [list(map(int, re.findall(r"\d+", tokens)))[:64] for tokens in tokens_list]

inputs = torch.LongTensor(inputs).to(args.device)
inputs = inputs.reshape(inputs.shape[0], 8, 8)
inputs = F.one_hot(inputs, num_classes=dim)
inputs = inputs.permute(0, 3, 1, 2).float()

    # Run the model forward
with torch.no_grad():
    imgs = model.decode(inputs)
    
img_path = os.path.join(args.output_dir, 'img.png')
imgs = imgs.clamp(-1, 1)

save_image(imgs, img_path, normalize=True, range=(-1, 1))
