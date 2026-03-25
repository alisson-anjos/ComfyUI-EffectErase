import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import copy
import gc

import folder_paths
import subprocess

# Allow importing from EffectErase
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.join(current_dir, "EffectErase")
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)

req_file = os.path.join(current_dir, "requirements.txt")

try:
    from huggingface_hub import snapshot_download, hf_hub_download
    from diffsynth import ModelManager, WanRemovePipeline
except ImportError as e:
    print(f"[EffectErase] WARNING: Missing dependencies. Attempting to install... Error: {e}")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])
        from huggingface_hub import snapshot_download, hf_hub_download
        from diffsynth import ModelManager, WanRemovePipeline
    except Exception as ie:
        msg = f"[EffectErase] Failed to install or import dependencies! Please run manually: {sys.executable} -m pip install -r {req_file}\nError: {ie}"
        print(msg)
        raise RuntimeError(msg)

# --- GLOBAL CACHE ---
GLOBAL_CACHE = {
    "pipe": None,
    "current_lora": None,
    "current_lora_strength": None,
    "current_dtype": None,
    "current_vram": None,
}

def crop_square_from_pil(mask_img: Image.Image, fg_bg_img: Image.Image, target_size: int = 224):
    mask_np = np.array(mask_img)
    if mask_np.ndim == 3:
        mask_np = mask_np.max(axis=-1)
    mask_np = (mask_np > 0).astype(np.uint8)

    img_np = np.array(fg_bg_img.convert("RGB"))
    h, w = mask_np.shape

    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0:
        # Fallback if mask is empty
        xs = np.array([w // 2])
        ys = np.array([h // 2])

    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    bw, bh = x1 - x0, y1 - y0

    side = max(bw, bh)
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0

    sx0 = int(np.floor(cx - side / 2))
    sy0 = int(np.floor(cy - side / 2))
    sx1 = sx0 + side
    sy1 = sy0 + side

    pad_left = max(0, -sx0)
    pad_top = max(0, -sy0)
    pad_right = max(0, sx1 - w)
    pad_bottom = max(0, sy1 - h)

    if pad_left or pad_top or pad_right or pad_bottom:
        img_np = np.pad(
            img_np,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        mask_np = np.pad(
            mask_np,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=0,
        )
        sx0 += pad_left
        sx1 += pad_left
        sy0 += pad_top
        sy1 += pad_top

    crop_img = img_np[sy0:sy1, sx0:sx1]
    crop_mask = mask_np[sy0:sy1, sx0:sx1][..., None]
    crop_img = crop_img * crop_mask

    crop_t = torch.from_numpy(crop_img).permute(2, 0, 1).float().unsqueeze(0)  # [1,3,S,S]
    crop_t = F.interpolate(
        crop_t,
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
    )[0]
    crop_t = crop_t / 255.0
    crop_t = crop_t * 2.0 - 1.0  # [-1,1]

    return crop_t


class EffectEraseObjectRemoval:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_fg_bg": ("IMAGE",),
                "video_mask": ("MASK",),
                "remove_prompt": ("STRING", {"multiline": True, "default": "Remove the specified object and all related effects, then restore a clean background."}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "细节模糊不清，字幕，作品，画作，画面，静止，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，杂乱的背景，三条腿，背景人很多，倒着走"}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 20.0, "step": 0.5}),
                "sigma_shift": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 15.0, "step": 0.5}),
                "accel_lora": (["none"] + folder_paths.get_filename_list("loras"), ),
                "accel_lora_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05}),
                "dtype": (["bfloat16", "fp16"], {"default": "bfloat16"}),
                "vram_mode": (["low_vram", "high_vram"], {"default": "low_vram"}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "tiled": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("removed_video",)
    FUNCTION = "process"
    CATEGORY = "EffectErase"

    def download_models_if_needed(self):
        models_dir = folder_paths.models_dir
        effect_erase_dir = os.path.join(models_dir, "EffectErase")
        os.makedirs(effect_erase_dir, exist_ok=True)
        
        wan_path = os.path.join(effect_erase_dir, "Wan2.1-Fun-1.3B-InP")
        if not os.path.exists(wan_path):
            print("[EffectErase] Downloading Wan-AI/Wan2.1-Fun-1.3B-InP...")
            snapshot_download(repo_id="alibaba-pai/Wan2.1-Fun-1.3B-InP", local_dir=wan_path)
            
        lora_ckpt = os.path.join(effect_erase_dir, "EffectErase.ckpt")
        if not os.path.exists(lora_ckpt):
            print("[EffectErase] Downloading EffectErase.ckpt...")
            hf_hub_download(repo_id="FudanCVL/EffectErase", filename="EffectErase.ckpt", local_dir=effect_erase_dir)
            
        return wan_path, lora_ckpt

    def process(self, video_fg_bg, video_mask, remove_prompt, negative_prompt, num_inference_steps, cfg, sigma_shift, accel_lora, accel_lora_strength, dtype, vram_mode, keep_model_loaded, seed, tiled):
        global GLOBAL_CACHE
        
        if GLOBAL_CACHE.get("pipe") is not None and \
           GLOBAL_CACHE.get("current_lora") == accel_lora and \
           GLOBAL_CACHE.get("current_lora_strength") == accel_lora_strength and \
           GLOBAL_CACHE.get("current_dtype") == dtype and \
           GLOBAL_CACHE.get("current_vram") == vram_mode and keep_model_loaded:
            print("[EffectErase] Reusing cached model pipeline...")
            pipe = GLOBAL_CACHE["pipe"]
        else:
            if GLOBAL_CACHE.get("pipe") is not None:
                print("[EffectErase] Purging old model pipeline to load new config...")
                GLOBAL_CACHE["pipe"] = None
                gc.collect()
                torch.cuda.empty_cache()
            
            wan_path, lora_ckpt = self.download_models_if_needed()

            text_encoder_path = os.path.join(wan_path, "models_t5_umt5-xxl-enc-bf16.pth")
            vae_path = os.path.join(wan_path, "Wan2.1_VAE.pth")
            dit_path = os.path.join(wan_path, "diffusion_pytorch_model.safetensors")
            image_encoder_path = os.path.join(wan_path, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")

            dtype_map = {
                "fp16": torch.float16,
                "bfloat16": torch.bfloat16,
            }
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_manager = ModelManager(device=device)
            model_manager.load_models(
                [
                    dit_path,
                    text_encoder_path,
                    vae_path,
                    image_encoder_path,
                ],
                torch_dtype=dtype_map[dtype],
            )
            model_manager.load_lora_v2(lora_ckpt, lora_alpha=1.0)
            
            if accel_lora != "none":
                accel_lora_path = folder_paths.get_full_path("loras", accel_lora)
                print(f"[EffectErase] Loading acceleration LoRA from file: {accel_lora_path} with strength {accel_lora_strength}")
                model_manager.load_lora_v2(accel_lora_path, lora_alpha=accel_lora_strength)

            pipe = WanRemovePipeline.from_model_manager(model_manager)
            if vram_mode == "low_vram":
                pipe.enable_vram_management(num_persistent_param_in_dit=6 * 10**9)

        fg_bg_first_np = (video_fg_bg[0].cpu().numpy() * 255).astype(np.uint8)
        fg_bg_first_pil = Image.fromarray(fg_bg_first_np)

        if len(video_mask.shape) == 3: # [T, H, W]
            mask_first_np = (video_mask[0].cpu().numpy() * 255).astype(np.uint8)
        else:
            mask_first_np = (video_mask[0, ..., 0].cpu().numpy() * 255).astype(np.uint8)
        mask_first_pil = Image.fromarray(mask_first_np, mode="L")

        fg_first_img = crop_square_from_pil(mask_first_pil, fg_bg_first_pil, target_size=224)

        T, H, W, C = video_fg_bg.shape
        fg_bg_imgs_tensor = video_fg_bg.permute(3, 0, 1, 2).float()
        fg_bg_imgs_tensor = fg_bg_imgs_tensor * 2.0 - 1.0 # [0, 1] to [-1, 1]

        if len(video_mask.shape) == 3:
            mask_tensor = video_mask.unsqueeze(1).repeat(1, 3, 1, 1) # [T, 3, H, W]
        else:
            mask_tensor = video_mask.repeat(1, 1, 1, 3).permute(0, 3, 1, 2)
        
        mask_imgs_tensor = mask_tensor.permute(1, 0, 2, 3).float() # [3, T, H, W]
        mask_imgs_tensor = (mask_imgs_tensor > 0.5).float()
        mask_imgs_tensor = mask_imgs_tensor * 2.0 - 1.0

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mask_imgs_tensor = mask_imgs_tensor.to(device)
        fg_bg_imgs_tensor = fg_bg_imgs_tensor.to(device)
        fg_first_img = fg_first_img.to(device)

        print("[EffectErase] Generating video inference...")
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype_map[dtype]):
            remove_video_frames, _ = pipe(
                video_mask=mask_imgs_tensor,
                video_fg_bg=fg_bg_imgs_tensor,
                video_bg=None,
                task="remove",
                fg_first_img=fg_first_img,
                prompt_remove=remove_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                cfg_scale=cfg,
                sigma_shift=sigma_shift,
                seed=seed,
                tiled=tiled,
                height=H,
                width=W,
                num_frames=T,
            )

        out_tensor = torch.from_numpy(np.array(remove_video_frames)).float() / 255.0
        
        if keep_model_loaded:
            GLOBAL_CACHE["pipe"] = pipe
            GLOBAL_CACHE["current_lora"] = accel_lora
            GLOBAL_CACHE["current_lora_strength"] = accel_lora_strength
            GLOBAL_CACHE["current_dtype"] = dtype
            GLOBAL_CACHE["current_vram"] = vram_mode
        else:
            GLOBAL_CACHE["pipe"] = None
            pipe = None
            gc.collect()
            torch.cuda.empty_cache()

        return (out_tensor,)

NODE_CLASS_MAPPINGS = {
    "EffectEraseNode": EffectEraseObjectRemoval
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EffectEraseNode": "EffectErase Object Removal"
}
