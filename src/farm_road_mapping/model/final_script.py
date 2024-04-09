import os

import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from model import DeepLabv3


model_dir='model.pt'
output_dir='test outputs'
img_path='0_128_896.tif'

model=DeepLabv3()
checkpoint=torch.load(model_dir)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

img=cv2.imread(img_path)

device=('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

img= torch.tensor(img, dtype=torch.float32).unsqueeze(0).permute(0,3,1,2).to(device)
img=img.to(device)

with torch.no_grad():
    output=model(img)
    output=torch.argmax(output, dim=1)

output = output.squeeze(0).cpu().numpy()

out=Image.fromarray((output*255).astype(np.uint8))
save_path=os.path.join(output_dir, 'output.jpg')
out.save(save_path)