import time
import os
from typing import List
import json

from tqdm.auto import tqdm
import torch
from cog import BasePredictor, Input, Path
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler,
)
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from PIL import Image
from transformers import CLIPFeatureExtractor
import shutil
import subprocess
from diffusers.utils import load_image

SAFETY_MODEL_CACHE = "diffusers-cache"
SAFETY_MODEL_ID = "CompVis/stable-diffusion-safety-checker"

DEFAULT_HEIGHT = 512
DEFAULT_WIDTH = 512
DEFAULT_SCHEDULER = "DDIM"
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_NUM_INFERENCE_STEPS = 50
DEFAULT_STRENGTH = 0.8

# grab instance_prompt from weights,
# unless empty string or not existent

DEFAULT_PROMPT = "a photo of an astronaut riding a horse on mars"


class KerrasDPM:
    def from_config(config):
        return DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True)


SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "HeunDiscrete": HeunDiscreteScheduler,
    "KerrasDPM": KerrasDPM,
    "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler,
    "K_EULER": EulerDiscreteScheduler,
    "KLMS": LMSDiscreteScheduler,
    "PNDM": PNDMScheduler,
    "UniPCMultistep": UniPCMultistepScheduler,
}


class Predictor(BasePredictor):
    def setup(self):
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            SAFETY_MODEL_ID,
            cache_dir=SAFETY_MODEL_CACHE,
            torch_dtype=torch.float16,
            local_files_only=True,
        ).to("cuda")
        self.feature_extractor = CLIPFeatureExtractor.from_pretrained(
            "openai/clip-vit-base-patch32", cache_dir=SAFETY_MODEL_CACHE
        )
        self.url = None


    def download_tar_weights(self, url):
        """Download the model weights from the given URL"""
        print("Downloading weights...")

        if os.path.exists("weights"):
            shutil.rmtree("weights")
        os.makedirs("weights")
        subprocess.check_output(["script/get_weights.sh", url], stderr=subprocess.STDOUT)

    def download_zip_weights_python(self, url):
        """Download the model weights from the given URL"""
        print("Downloading weights...")
   
        if os.path.exists("weights"):
            shutil.rmtree("weights")
        os.makedirs("weights")

        import zipfile
        from io import BytesIO
        import urllib.request

        url = urllib.request.urlopen(url)
        with zipfile.ZipFile(BytesIO(url.read())) as zf:
            zf.extractall("weights")

    def load_weights(self, url):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading Safety pipeline...")

        if url == self.url:
            return

        start_time = time.time()
        self.download_zip_weights_python(url)
        print("Downloaded weights in {:.2f} seconds".format(time.time() - start_time))

        start_time = time.time()
        print("Loading SD pipeline...")
        self.txt2img_pipe = StableDiffusionPipeline.from_pretrained(
            "weights",
            safety_checker=self.safety_checker,
            feature_extractor=self.feature_extractor,
            torch_dtype=torch.float16,
        ).to("cuda")

        self.img2img_pipe = StableDiffusionImg2ImgPipeline(
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            tokenizer=self.txt2img_pipe.tokenizer,
            unet=self.txt2img_pipe.unet,
            scheduler=self.txt2img_pipe.scheduler,
            safety_checker=self.txt2img_pipe.safety_checker,
            feature_extractor=self.txt2img_pipe.feature_extractor,
        ).to("cuda")
        print("Loaded pipelines in {:.2f} seconds".format(time.time() - start_time))

        self.txt2img_pipe.set_progress_bar_config(disable=True)
        self.img2img_pipe.set_progress_bar_config(disable=True)
        self.url = url

    def generate_images(self, images, output_dir):
        with torch.autocast("cuda"), torch.inference_mode():
            for info in tqdm(images, desc="Generating samples"):
                inputs = info.get("input") or info.get("inputs")
                name = info["name"]
                print(name)

                num_outputs = int(inputs.get("num_outputs", 1))

                kwargs = {
                    "prompt": [inputs["prompt"]] * num_outputs,
                    "num_inference_steps": int(inputs.get("num_inference_steps", DEFAULT_NUM_INFERENCE_STEPS)),
                    "guidance_scale": float(inputs.get("guidance_scale", DEFAULT_GUIDANCE_SCALE)),
                }

                image = inputs.get("image")
                if image is not None:
                    kwargs['image'] = load_image(image)
                    kwargs['strength'] = float(inputs.get('strength', DEFAULT_STRENGTH))
                    pipeline = self.img2img_pipe
                else:
                    pipeline = self.txt2img_pipe
                    kwargs["width"] = int(inputs.get("width", DEFAULT_WIDTH))
                    kwargs["height"] = int(inputs.get("height", DEFAULT_HEIGHT))

                negative_prompt = inputs.get("negative_prompt")
                if negative_prompt is not None:
                    kwargs["negative_prompt"] = [negative_prompt] * num_outputs

                scheduler = inputs.get("scheduler", DEFAULT_SCHEDULER)
                pipeline.scheduler = SCHEDULERS[scheduler].from_config(pipeline.scheduler.config)

                if bool(inputs.get("disable_safety_check", False)):
                    pipeline.safety_checker = None
                else:
                    pipeline.safety_checker = self.safety_checker

                seed = int(inputs.get("seed", int.from_bytes(os.urandom(2), "big")))
                generator = torch.Generator("cuda").manual_seed(seed)
                output = pipeline(
                    generator=generator,
                    **kwargs,
                )

                for i, image in enumerate(output.images):
                    if output.nsfw_content_detected and output.nsfw_content_detected[i]:
                        print("skipping nsfw detected for", inputs)
                        continue
                    image.save(os.path.join(output_dir, f"{name}-{i}.png"))


    @torch.inference_mode()
    def predict(
        self,
        images: str = Input(
            description="JSON input",
        ),
        weights: str = Input(
            description="URL to weights",
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""

        weights = weights.replace("https://replicate.delivery/pbxt/", "https://storage.googleapis.com/replicate-files/")

        images_json = json.loads(images)
    
        if weights is None:
            raise ValueError("No weights provided")
        self.load_weights(weights)

        cog_generated_images = "cog_generated_images"
        if os.path.exists(cog_generated_images):
            shutil.rmtree(cog_generated_images)
        os.makedirs(cog_generated_images)

        self.generate_images(images_json, cog_generated_images)

        directory = Path(cog_generated_images)

        results = []
        for file_path in directory.rglob("*"):
            print(file_path)
            results.append(file_path)
        return results
