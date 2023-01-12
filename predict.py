# Simple Cog interface for ESRGAN models
# https://github.com/replicate/cog/blob/main/docs/python.md

import torch
from cog import BasePredictor, Input, Path
from PIL import Image

from esrgan_model import esrgan_upscale, load_model

MODEL_FILES = {
    "4x_UniversalUpscalerV2-Neutral_115000_swaG": "upscalers/4x_UniversalUpscalerV2-Neutral_115000_swaG.pth",
    "4x_UniversalUpscalerV2-Sharp_101000_G": "upscalers/4x_UniversalUpscalerV2-Sharp_101000_G.pth",
    "4x_UniversalUpscalerV2-Sharper_103000_G": "upscalers/4x_UniversalUpscalerV2-Sharper_103000_G.pth",
}


class Predictor(BasePredictor):
    """Predictor for upscalers"""

    def setup(self):
        """Load all models into memory"""
        self.models = {}

        for model_name, model_path in MODEL_FILES.items():
            model = load_model(model_path)
            self.models[model_name] = model

    def predict(
        self,
        image: Path = Input(description="Input image"),
        model_name: str = Input(
            description="Model to use for upscaling",
            default="4x_UniversalUpscalerV2-Neutral_115000_swaG",
            choices=list(MODEL_FILES.keys()),
        ),
    ) -> Path:
        """Upscale a single image"""

        img = Image.open(image)
        model = self.models[model_name]
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model.to(device)

        upscaled_img = esrgan_upscale(model, img)
        upscaled_img.save("output.png")

        return Path("output.png")
