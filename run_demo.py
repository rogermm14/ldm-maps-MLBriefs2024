import os
import numpy as np
import argparse
import shutil
from PIL import Image
import torchvision.transforms as tf
import torch

from src.utils_maps import *

import warnings
warnings.filterwarnings("ignore")

# if you need to access a file next to the source code, use the variable ROOT
# for example:
#    torch.load(os.path.join(ROOT, 'weights.pth'))
ROOT = os.path.dirname(os.path.realpath(__file__))

def get_opt():
    parser = argparse.ArgumentParser(description='Conditional Diffusion MLBriefs 2024 Demo')
    parser.add_argument('--img_path', type=str, default='example_data/206_map.jpg', help='Input condition image path')
    parser.add_argument('--time_steps', type=int, default=1000, help='Diffusion time steps')
    args = parser.parse_args()
    return args


def run_demo(img_path, time_steps):
    
    print(f"Input image: {img_path}")
    print(f"Number of time steps: {time_steps}")
    im_size = 256
    n_outputs = 4
    out_dir="demo_output"
    
    # read input and force input size equal to (256, 256)
    assert os.path.exists(img_path), "Input image not found"
    input_rgb =  tf.functional.pil_to_tensor(Image.open(img_path))/255
    h, w = input_rgb.shape[1:]
    assert input_rgb.shape[0] == 3, "Input image is not rgb. 3 channels expected."
    
    augmentations = transforms.Compose(
        [
            transforms.CenterCrop((512, 512)),
            transforms.Resize(im_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    t_cond = augmentations(Image.open(img_path))
    if (h != im_size) or  (w != im_size):
        print("\nWarning: Input image was resized to (256, 256)\n")
    t_cond = torch.tile(t_cond.unsqueeze(0), (4, 1, 1, 1))


    # load model
    # run pipeline in inference (sample random noise and denoise)
    ckpt_path = "rogermm14/MLBriefs24_5_conditional_CA_encodedmask"
    pipeline = load_pipeline(ckpt_path)
    print("Loaded model successfully")
        
    # run diffusion model
    generator = torch.Generator(device=pipeline.device).manual_seed(0)
    output = pipeline(
        input_condition_imgs=t_cond,
        generator=generator,
        batch_size=n_outputs,
        num_inference_steps=time_steps,
        output_type="numpy",
        return_dict=False
    )
    images, cond_images = output

    # save output synthetic images
    os.makedirs(out_dir, exist_ok=True)
    img = Image.fromarray((cond_images[0]*255).astype(np.uint8))
    img.save(os.path.join(out_dir, "input.png"))
    for idx in range(n_outputs):
        img = Image.fromarray((images[idx]*255).astype(np.uint8))
        out_path = os.path.join(out_dir, f"output_{idx:02}.png")
        img.save(out_path)
        assert os.path.exists(out_path), "Output image not found"
        print(f"Saved {out_path}")
    # save input as well
    #shutil.copy(img_path, os.path.join(out_dir, f"input.png"))
        
    print(f"Done")

if __name__ == "__main__":

    args = get_opt()
    run_demo(args.img_path, args.time_steps)

