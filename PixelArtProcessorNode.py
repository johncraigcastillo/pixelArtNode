import numpy as np
from PIL import Image, ImageEnhance
import torch
from typing import List

class ImagePalette:
    def __init__(self, name: str, image: Image.Image):
        self.name = name
        self.image = image

class PixelArtProcessorNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),  # Batch of images
                "palette_image": ("IMAGE",),  # Palette image
                "factor": ("INT", {"default": 8, "min": 1, "max": 16}),  # Customizable downscale/upscale factor
                "dither": (["None", "FloydSteinberg"],),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
                "colors": ("INT", {"default": 0, "min": 0, "max": 256}),  # Set 0 to use all colors from palette
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_images"
    CATEGORY = "Image Processing"

    def process_images(self, images, palette_image, factor, dither, contrast, saturation, colors):
        processed_images = []

        # Convert dither option to corresponding PIL constant
        dither_option = Image.NONE if dither == "None" else Image.FLOYDSTEINBERG

        # Convert palette image from Tensor to numpy array and create ImagePalette object
        palette_np = palette_image.cpu().detach().numpy()
        palette_np = (palette_np * 255).clip(0, 255).astype('uint8')

        # Convert palette numpy array to a PIL Image and ensure it's in RGB
        if len(palette_np.shape) == 4 and palette_np.shape[0] == 1:
            palette_np = palette_np[0]
        palette_pil = Image.fromarray(palette_np).convert('P', palette=Image.ADAPTIVE)

        # Create the ImagePalette object
        palette = ImagePalette("palette", palette_pil)

        # Extract number of colors if not specified
        if colors == 0:
            colors = len(palette_pil.getcolors(maxcolors=256)) or 256

        # Loop through each image in the batch
        for image in images:
            # Convert the image from Tensor to numpy array
            image_np = image.cpu().detach().numpy()

            # Convert the image to uint8 format by scaling to 0-255 and changing dtype
            image_np = (image_np * 255).clip(0, 255).astype('uint8')

            # Remove any extra dimensions if they exist (e.g., if the image is 1xHxWx3)
            if len(image_np.shape) == 4 and image_np.shape[0] == 1:
                image_np = image_np[0]

            # Convert the numpy array to a PIL Image
            target_pil_image = Image.fromarray(image_np).convert('RGB')
            og_width, og_height = target_pil_image.size

            # Downscale the target image using the factor
            new_width = max(1, og_width // factor)
            new_height = max(1, og_height // factor)
            img = target_pil_image.resize((new_width, new_height), Image.NEAREST)

            # Adjust contrast
            if contrast != 1.0:
                contrast_enhancer = ImageEnhance.Contrast(img)
                img = contrast_enhancer.enhance(contrast)

            # Adjust saturation
            if saturation != 1.0:
                saturation_enhancer = ImageEnhance.Color(img)
                img = saturation_enhancer.enhance(saturation)

            # Apply dithering option and palette quantization
            img = img.quantize(palette=palette.image, dither=dither_option)

            # Apply forced quantization to reduce the number of colors if specified
            if colors > 0:
                img = img.quantize(colors=colors).convert('RGB')

            # Resize back to the original size of the target image
            img = img.resize((og_width, og_height), Image.NEAREST)

            # Convert back to numpy array and normalize back to 0-1 range for ComfyUI
            output_image_np = np.array(img).astype(np.float32) / 255.0

            # Convert back to torch tensor
            output_image_tensor = torch.from_numpy(output_image_np)

            # Append processed image to list
            processed_images.append(output_image_tensor)

        # Stack images to create a batch and return it
        processed_images_tensor = torch.stack(processed_images)

        return (processed_images_tensor,)

NODE_CLASS_MAPPINGS = {
    "PixelArtProcessorNode": PixelArtProcessorNode,
}
