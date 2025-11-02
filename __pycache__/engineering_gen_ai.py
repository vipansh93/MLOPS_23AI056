# ================================================
# IntelliScale AI - Engineering Image Generator
# Author: Vivek Rajak
# ================================================

import torch
from PIL import Image
import cv2
import numpy as np
import os
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

# ==========================================================
# 1️⃣ SETUP PATHS AND CONFIG
# ==========================================================
os.makedirs("models/sam", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Downloaded SAM checkpoint location
SAM_CHECKPOINT = "models/sam/sam_vit_h_4b8939.pth"  # Change if using vit_b or vit_l
MODEL_TYPE = "vit_h"

# Your fine-tuned ControlNet model (or use pretrained)
CONTROLNET_MODEL_PATH = "lllyasviel/sd-controlnet-canny"

# Input test image
INPUT_IMAGE_PATH = "data/test/test_image.png"  # Change this to your test image path

# ==========================================================
# 2️⃣ LOAD SAM MODEL
# ==========================================================
print("🔹 Loading SAM model...")
sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
mask_generator = SamAutomaticMaskGenerator(sam)
print("✅ SAM model loaded successfully!")

# ==========================================================
# 3️⃣ SEGMENT OBJECTS IN IMAGE
# ==========================================================
print("🔹 Running SAM segmentation...")
image_bgr = cv2.imread(INPUT_IMAGE_PATH)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
masks = mask_generator.generate(image_rgb)

print(f"✅ {len(masks)} objects detected in image.")

# Save segmented overlay image
annotated = image_rgb.copy()
for m in masks:
    color = np.random.randint(0, 255, (1, 3), dtype=np.uint8)
    annotated[m['segmentation']] = color
Image.fromarray(annotated).save("outputs/segmented.png")
print("💾 Saved segmentation as outputs/segmented.png")

# ==========================================================
# 4️⃣ LOAD CONTROLNET FOR IMAGE GENERATION
# ==========================================================
print("🔹 Loading ControlNet...")
controlnet = ControlNetModel.from_pretrained(CONTROLNET_MODEL_PATH, torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
)
pipe.to("cuda" if torch.cuda.is_available() else "cpu")
print("✅ ControlNet model loaded!")

# ==========================================================
# 5️⃣ GENERATE ENGINEERING VIEWS (ZOOM, ROTATE, ETC.)
# ==========================================================
print("🔹 Generating engineering image variants...")

prompts = [
    "engineering drawing top view, zoomed in, clean technical style",
    "engineering drawing wide angle perspective, realistic shading",
    "exploded mechanical diagram, all components visible, dimension labels",
    "isometric engineering view, detailed, sharp edges",
    "side view technical projection, white background"
]

input_pil = Image.open(INPUT_IMAGE_PATH).convert("RGB")

for i, prompt in enumerate(prompts):
    print(f"🧩 Generating variant {i+1} -> {prompt}")
    result = pipe(
        prompt=prompt,
        image=input_pil,
        num_inference_steps=50,
        guidance_scale=7.5
    )
    result.images[0].save(f"outputs/variant_{i+1}.png")
    print(f"💾 Saved outputs/variant_{i+1}.png")

print("🎯 All engineering image variants generated successfully!")
