# Image-Enhancer

Image-Enhancer is a simple command-line tool that uses the Stable Diffusion x4 upscaler model from Hugging Face to upscale images (from local path or URL) by 4×.

Author: ArihantSharma  
License: MIT

---

## Features

- Upscales any image (local file or image URL) by 4×
- Based on stabilityai/stable-diffusion-x4-upscaler
- Works with MPS (Mac), CUDA (NVIDIA), or CPU
- Enables memory optimizations (attention slicing, VAE tiling)
- Simple command-line interface

---

## Installation

1. Clone the repo:

    ```bash
    git clone https://github.com/ArihantSharma/Image-Enhancer.git
    cd Image-Enhancer
2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
Requires Python 3.9+

---

## Usage

Run:

    python enhancer.py

It will ask:

    Enter image URL / path  (q=quit):

Provide a valid image URL or local file path.  
The enhanced image will be saved as `<filename>_x4.png`.

Type `q` or `quit` to exit.

---

## requirements.txt

Contents of `requirements.txt`:

    diffusers==0.27.2
    transformers>=4.40.0
    torch>=2.3.0
    pillow
    requests

---

## Notes for macOS M1/M2 users

To avoid memory errors, set:

    export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8
    export PYTORCH_ENABLE_SDPA=0

---

## Optional: Create Executable

Install PyInstaller:

    pip install pyinstaller

Build a standalone binary:

    pyinstaller --onefile enhancer.py

The binary will be in the `dist/` directory.

---

## License

MIT License. See `LICENSE` file for details.

---

## Credits

- https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler
- https://github.com/huggingface/diffusers
