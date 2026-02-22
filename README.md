# ai-images

A Google Colab notebook for generating AI images using [Stable Diffusion 3.5 Medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) via the Hugging Face [diffusers](https://github.com/huggingface/diffusers) library.

## Features

- Text-to-image generation using Stable Diffusion 3.5 Medium
- Memory-optimized for free-tier GPU (T4) in Google Colab
- Configurable prompt, image size, inference steps, and guidance scale

## Prerequisites

- A [Hugging Face](https://huggingface.co/) account and access token
- Google Colab (free tier with T4 GPU is sufficient)

## Getting Started

1. Open the notebook in Google Colab by clicking the badge below:

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sujithkumarmp/ai-images/blob/main/diffusers_generator.ipynb)

2. Run the first cell to install the required dependencies:
   ```
   pip install -U diffusers transformers accelerate torch
   ```

3. Log in to Hugging Face with your access token when prompted.

4. Update the `prompt` variable with your desired image description.

5. Run the generation cell — the output image will be saved as `output_image.png`.

## Dependencies

| Package        | Purpose                              |
|----------------|--------------------------------------|
| `diffusers`    | Stable Diffusion pipeline            |
| `transformers` | Model components & tokenizers        |
| `accelerate`   | Memory optimization / CPU offloading |
| `torch`        | Deep learning backend                |

## Example Output

Prompt: *"A steampunk robotic owl perched on a crystal branch, digital art, 8k"*

![Output Image](output_image.png)

## Configuration

| Parameter            | Default | Description                         |
|----------------------|---------|-------------------------------------|
| `num_inference_steps`| 28      | Number of denoising steps           |
| `guidance_scale`     | 7.0     | How closely to follow the prompt    |
| `width` / `height`   | 768     | Output image resolution in pixels   |

> **Note:** Using a resolution of 768×768 is recommended for T4 GPUs (16 GB VRAM) to avoid out-of-memory errors. Increase to 1024×1024 if a higher-memory GPU is available (e.g., V100 or A100 with 16 GB+ VRAM).

## License

This project is open source. See the repository for details.
