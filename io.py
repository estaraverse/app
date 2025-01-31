# Install required libraries
!pip install -q diffusers transformers torch pillow gspread google-auth google-auth-oauthlib google-auth-httplib2 opencv-python-headless

# Import necessary libraries
from diffusers import StableDiffusionXLInpaintPipeline
import torch
from PIL import Image
import cv2
import numpy as np
from google.colab import auth
import os
import gspread
from google.auth import default
from google.colab import drive

# Authenticate and mount Google Drive
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

# Force remount Google Drive
if os.path.exists('/content/drive'):
    drive.flush_and_unmount()
    print('Google Drive unmounted.')
drive.mount('/content/drive', force_remount=True)
print('Google Drive mounted successfully!')

# Define paths
INPUT_DIR = os.path.join('/content/drive/My Drive/input-logos')
OUTPUT_DIR = os.path.join('/content/drive/My Drive/output-logos')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Check input directory
if not os.path.exists(INPUT_DIR):
    print(f"Error: Input directory '{INPUT_DIR}' not found.")
    os.makedirs(INPUT_DIR, exist_ok=True)
    print(f"Created input directory. Add images and restart.")
else:
    # Initialize SDXL pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    ).to(device)

    # Aesthetic configuration
    BASE_PROMPT = (
        "Minimalist logo with maximum 2 focal points, "
        "very dark blue (near black) background, "
        "neon green/orange/lilac/purple strokes in bright tones, "
        "clean modern design, high contrast, vector-style artwork"
    )

    def apply_lens_effect(image, distortion=0.3):
        """Applies barrel distortion for lens effect using OpenCV"""
        img = np.array(image)
        h, w = img.shape[:2]
        
        # Generate distortion maps
        map_x = np.zeros((h, w), np.float32)
        map_y = np.zeros((h, w), np.float32)
        scale = 1.5  # Curvature intensity
        
        for y in range(h):
            for x in range(w):
                nx = 2*(x - w/2)/w
                ny = 2*(y - h/2)/h
                r = np.sqrt(nx**2 + ny**2)
                theta = 1.0 / (1.0 + distortion*r**2)
                
                map_x[y, x] = (theta * nx * scale + 1) * w/2
                map_y[y, x] = (theta * ny * scale + 1) * h/2
        
        return Image.fromarray(cv2.remap(img, map_x, map_y, cv2.INTER_CUBIC))

    def process_logos(input_dir, output_dir):
        for filename in os.listdir(input_dir):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                input_path = os.path.join(input_dir, filename)
                try:
                    print(f"Processing: {filename}")
                    init_image = Image.open(input_path).convert("RGB").resize((1024, 1024))
                    mask_image = Image.new("RGB", init_image.size, "white")
                    
                    for i in range(4):
                        variation_prompt = f"{BASE_PROMPT} - Variation {i+1} with unique geometric composition"
                        result = pipe(
                            prompt=variation_prompt,
                            image=init_image,
                            mask_image=mask_image,
                            num_inference_steps=40,
                            strength=0.85,
                            guidance_scale=9.0
                        ).images[0]
                        
                        # Apply lens effect
                        curved_result = apply_lens_effect(result.resize((2048, 2048)))
                        
                        # Save output
                        output_filename = f"{os.path.splitext(filename)[0]}_variant_{i+1}.png"
                        output_path = os.path.join(output_dir, output_filename)
                        curved_result.save(output_path)
                        print(f"Saved: {output_path}")
                        
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")

    # Start processing
    process_logos(INPUT_DIR, OUTPUT_DIR)