import numpy as np
import os
import torch
from torchvision import transforms
from PIL import Image
import random
import argparse

from diffusers import UNet2DModel, VQModel
from src.pipeline import *
import json
import datetime


def parse_args(input_path=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that HF Datasets can understand."
        ),
    )
    parser.add_argument(
        "--val_dataset_name",
        type=str,
        default=None,
        help=(
            "Validation dataset (optional)."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--model_config_name_or_path",
        type=str,
        default=None,
        help="The config of the UNet model to train, leave as None to use standard DDPM configuration.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ddpm-model-256",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        default=False,
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=16, help="The number of images to generate for evaluation."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "The number of subprocesses to use for data loading. 0 means that the data will be loaded in the main"
            " process."
        ),
    )
    parser.add_argument("--num_epochs", type=int, default=150)
    parser.add_argument("--save_images_epochs", type=int, default=10, help="How often to save images during training.")
    parser.add_argument(
        "--save_model_epochs", type=int, default=10, help="How often to save the model during training."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.95, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-6, help="Weight decay magnitude for the Adam optimizer."
    )
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer.")
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Whether to use Exponential Moving Average for the final model weights.",
    )
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0, help="The inverse gamma value for the EMA decay.")
    parser.add_argument("--ema_power", type=float, default=3 / 4, help="The power value for the EMA decay.")
    parser.add_argument("--ema_max_decay", type=float, default=0.9999, help="The maximum decay magnitude for EMA.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--hub_private_repo", action="store_true", help="Whether or not to create a private repository."
    )
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
        help=(
            "Whether to use [tensorboard](https://www.tensorflow.org/tensorboard) or [wandb](https://www.wandb.ai)"
            " for experiment tracking and logging of model metrics and model checkpoints"
        ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="epsilon",
        choices=["epsilon", "sample"],
        help="Whether the model should predict the 'epsilon'/noise error or directly the reconstructed image 'x0'.",
    )
    parser.add_argument("--ddpm_num_steps", type=int, default=1000)
    parser.add_argument("--ddpm_num_inference_steps", type=int, default=1000)
    parser.add_argument("--ddpm_beta_schedule", type=str, default="linear")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=5,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--acc_seed",
        type=int,
        default=None,
        help="A seed to reproduce the training. If not set, the seed will be random.",
    )
    parser.add_argument(
        "--train_data_files",
        type=str,
        default=None,
        help=(
            "The files of the training data. The files must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--encode_cond", action="store_true", help="Whether or not to encode the condition image."
    )
    parser.add_argument(
        "--unconditional", action="store_true", help="Whether or not to condition the diffusion model."
    )
    parser.add_argument(
        "--crossattention", action="store_true", help="Whether or not to inject the condition image in crossattention layers."
    )

    if input_path is not None:
        args = load_args_from_json(input_path, parser)
    else:
        args = parser.parse_args()

        env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if env_local_rank != -1 and env_local_rank != args.local_rank:
            args.local_rank = env_local_rank

        if args.dataset_name is None and args.train_data_files is None and args.train_data_dir is None:
            raise ValueError("You must specify either a dataset name from the hub or a train data directory.")

        date_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        exp_id = f"{date_id}_" + args.output_dir.split("/")[-1]
        args.output_dir = args.output_dir.replace(args.output_dir.split("/")[-1], exp_id)
        args.logging_dir = args.output_dir

        save_args_to_json(args, os.path.join(args.output_dir, "args.json"))
    
    return args

def save_args_to_json(args, output_path):
    assert os.path.splitext(output_path)[-1] == ".json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

def load_args_from_json(input_path, parser):
    assert os.path.exists(input_path)
    assert os.path.splitext(input_path)[-1] == ".json"
    with open(input_path, 'r') as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
    args = parser.parse_args(args=[],namespace=t_args)
    return args

def get_images(image):
    image = np.array(image)
    h, w, c = image.shape
    s=w//2
    condition = image[:,s:,:]
    image = image[:,:s,:]
    image = Image.fromarray(image)
    condition = Image.fromarray(condition)
    return image, condition

def parse_maps(batch):
    
    resolution = 256
    augmentations = transforms.Compose(
        [
            transforms.CenterCrop((512, 512)),
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    angle = random.choice([0, 90, 180, 270])

    batch_size = len(batch['image'])
    out = {"image": [], "condition": []}
    for i in range(batch_size):
        image, condition = get_images(batch["image"][i])
        t_img = augmentations(image)
        t_img = transforms.functional.rotate(t_img, angle)
        out["image"].append(t_img)
        t_cond = augmentations(condition)
        t_cond = transforms.functional.rotate(t_cond, angle)
        out["condition"].append(t_cond)
    return out

def parse_maps_val(batch):
    
    resolution = 256
    augmentations = transforms.Compose(
        [
            transforms.CenterCrop((512, 512)),
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    batch_size = len(batch['image'])
    out = {"image": [], "condition": []}
    for i in range(batch_size):
        image, condition = get_images(batch["image"][i])
        t_img = augmentations(image)
        out["image"].append(t_img)
        t_cond = augmentations(condition)
        out["condition"].append(t_cond)
    return out

class CondLatentDiffusionPipeline_maps(LatentDiffusionPipelineBase):
    def __init__(
            self,
            vae: VQModel,
            scheduler: Union[
                DDIMScheduler,
                DDPMScheduler,
                DPMSolverMultistepScheduler,
                EulerAncestralDiscreteScheduler,
                EulerDiscreteScheduler,
                LMSDiscreteScheduler,
                PNDMScheduler,
            ],
            unet: Union[
                UNet2DModel,
                UNet2DConditionModel,
            ],
            args: argparse.Namespace,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            unet=unet,
            scheduler=scheduler,
        )

        self.args = args
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        latent_resolution = int(self.unet.config.sample_size)
        self.downsample_cond_img = transforms.Resize(latent_resolution, interpolation=transforms.InterpolationMode.BILINEAR)

    @torch.no_grad()
    def __call__(
            self,
            batch_size: int = 1,  # default to generate a single image
            input_condition_imgs: Optional[torch.FloatTensor] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: Optional[int] = 50,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            nodule_features: Optional[dict] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            eta: Optional[float] = 0.0,
            **kwargs,
    ) -> Union[Tuple, ImagePipelineOutput]:

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

        latents = self.prepare_latents(batch_size, 3, height, width,
                                       self.unet.dtype, self.device, generator, latents)

        self.scheduler.set_timesteps(num_inference_steps)

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        if not self.args.unconditional:
            assert input_condition_imgs.shape[0] == batch_size
            clean_condition = input_condition_imgs.to(latents.device)
            if self.args.encode_cond:
                cond_latents = self.vae.encode(clean_condition).latents
                cond_latents = cond_latents * 0.18215
            else:
                cond_latents = self.downsample_cond_img(clean_condition)


        for t in self.progress_bar(self.scheduler.timesteps):
            latents = self.scheduler.scale_model_input(latents, t)

            model_input = latents if self.args.unconditional else torch.cat((latents, cond_latents), 1)

            if self.args.crossattention:
                hidden_states = cond_latents.view(batch_size, 1, -1)
                noise_pred = self.unet(
                    model_input,
                    t,
                    encoder_hidden_states = hidden_states,
                ).sample
            else:
                noise_pred = self.unet(
                    model_input,
                    t,
                ).sample

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs
            ).prev_sample

        # scale and decode the image latents with vae
        image = self.decode_latents(latents)
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            cond_images = None if self.args.unconditional else clean_condition.permute(0, 2, 3, 1).cpu().numpy()
            return (image, cond_images)

        return ImagePipelineOutput(images=image)

def load_pipeline(model_path, verbose=True):

    if verbose:
        print("Loading Diffusion pipeline from:")
        print(f"    - {model_path}\n")

    args = parse_args(input_path=os.path.join(model_path, "args.json"))

    vae = VQModel.from_pretrained("CompVis/ldm-celebahq-256", subfolder="vqvae")
    vae.requires_grad_(False)
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    vae.cuda()
    if verbose:
        print("VQ-VAE loaded")
    if not args.unconditional and args.crossattention:
        unet = UNet2DConditionModel.from_pretrained(model_path, subfolder=f"unet")
    else:
        unet = UNet2DModel.from_pretrained(model_path, subfolder=f"unet")
    unet.cuda()
    if verbose:
        print("U-Net model loaded")

    scheduler_config_path = model_path + "/scheduler/scheduler_config.json" 
    noise_scheduler = DDPMScheduler.from_config(scheduler_config_path)
    
    pipeline = CondLatentDiffusionPipeline_maps(
        vae=vae,
        unet=unet,
        scheduler=noise_scheduler,
        args=args)
    print("Diffusion pipeline is ready\n")
    
    return pipeline