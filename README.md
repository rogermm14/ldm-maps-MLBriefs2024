# Conditional LDM for Aerial Imagery

Check our paper *Latent Diffusion Approaches for Conditional Generation of Aerial Imagery: A Study* (2025), published at the [MLBriefs 2024](https://mlbriefs.com/) workshop of the Image Processing On Line journal ([IPOL](https://www.ipol.im/)).

This repository allows to explore a conditional latent diffusion model for the generation of aerial images from an input map.

We used the public `pix2pix-maps` dataset. Available [here](https://www.kaggle.com/datasets/alincijov/pix2pix-maps).

Based on [zyinghua/uncond-image-generation-ldm](https://github.com/zyinghua/uncond-image-generation-ldm) and [huggingface/diffusers](https://github.com/huggingface/diffusers).

If you find this code or work helpful, please cite:
```
@article{mari2025latent,
  title={Latent Diffusion Approaches for Conditional Generation of Aerial Imagery: A Study},
  author={Mar{\'\i}, Roger and Redondo, Rafael},
  journal={Image Processing On Line},
  year={2025}
}
```

---

<img src="example_data/teaser.png" alt="Conditional LDM for Aerial Imagery" width="750"/>
<strong>Figure 1:</strong> Left to right: Real aerial image, conditional map input to the diffusion model and 2 different synthetic output samples

---

## Installation

Use the script `setup_ldm-mlbriefs24_venv.sh` to install the necessary conda environment and train the LDM on your own dataset.

You can also check our [online demo](https://ipolcore.ipol.im/demo/clientApp/demo.html?id=77777000505).

