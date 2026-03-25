# ComfyUI-EffectErase

A custom node for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that implements **Joint Video Object Removal and Insertion** using the Wan2.1 1.3B Video DiT architecture. 

This repository allows you to perform advanced, temporally consistent Video-to-Video inpainting and object removal directly inside your ComfyUI workflows without complicated setups.

## Features
- **Object Removal**: Erase moving objects from videos by providing an input video and a mask.
- **Auto-Dependencies**: Automatically installs `modelscope`, `peft`, and HuggingFace dependencies at startup.
- **Model Auto-Download**: Automatically fetches the required Wan2.1 model and EffectErase LoRAs on the first generation and saves them to `models/EffectErase/`.
- **Acceleration LoRA Support**: Supports attaching native Wan2.1 acceleration LoRAs to speed up inference times.
- **FlowMatch Scheduler Configuration**: Exposes `sigma_shift`, CFG, inference steps, and seed controls.

## Usage
Simply plug the **`EffectEraseObjectRemoval`** node into your ComfyUI workflow.
1. Provide a `video_fg_bg` (Image Tensor representing video frames).
2. Provide a `video_mask` (Target Object Mask).
3. Connect the output to a Save Video or Video Combine node.

## Acknowledgements & Credits

This node is just a wrapper to bring the state-of-the-art research into the ComfyUI ecosystem. All credit for the mathematical implementation, training, and neural architecture modifications belongs entirely to the original authors.

* **EffectErase**: Developed by FudanCVL. [Official Repository](https://github.com/FudanCVL/EffectErase)
* **DiffSynth-Studio**: The inference framework used to execute the custom modified WanRemovePipeline natively. [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
* **Wan2.1**: The underlying Video Diffusion Transformer baseline.

If you use this node in your research or projects, please consider starring and citing the original [EffectErase repository](https://github.com/FudanCVL/EffectErase).
