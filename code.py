!pip install diffusers torch transformers gradio

from diffusers import StableDiffusionPipeline
import torch
import gradio as gr

# Load model
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipe.to("cuda")

# Function to generate image
def generate_image(prompt):
    image = pipe(prompt).images[0]
    return image

# Launch public web app
gr.Interface(fn=generate_image, inputs="text", outputs="image").launch(share=True)
