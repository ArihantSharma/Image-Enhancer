
from diffusers import StableDiffusionUpscalePipeline
from PIL import Image
import requests, torch, os, urllib.parse, sys

def load_image(src: str) -> Image.Image:
    """
    Accepts a local path or an http/https URL and returns a PIL Image.
    """
    if urllib.parse.urlparse(src).scheme in ("http", "https"):
        return Image.open(requests.get(src, stream=True).raw).convert("RGB")
    else:
        return Image.open(os.path.expanduser(src)).convert("RGB")

def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.backends.cuda.is_available() else "cpu"
    pipe   = StableDiffusionUpscalePipeline.from_pretrained(
        "stabilityai/stable-diffusion-x4-upscaler",
        torch_dtype=torch.float32
    ).to(device)

    pipe.enable_attention_slicing()
    pipe.enable_vae_tiling()
    print("🖼  Stable Diffusion 4× Upscaler ready.\n")

    while True:
        src = input("Enter image URL / path  (q=quit): ").strip()
        if src.lower() in {"q", "quit", "exit"}:
            print("Bye!")
            break

        try:
            img = load_image(src)
        except Exception as e:
            print(f"❌  Couldn’t open image: {e}\n")
            continue

        try:
            result = pipe(
                prompt="anime manga cover",    # change or parametrize if you like
                image=img,
                num_inference_steps=20,
                guidance_scale=1.0
            ).images[0]
        except Exception as e:
            print(f"⚠️  Upscaling failed: {e}\n")
            continue

        base = os.path.splitext(os.path.basename(src))[0] or "output"
        out  = f"{base}_x4.png"
        result.save(out)
        print(f"✅  Saved → {out}\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Goodbye!")
        sys.exit(0)
