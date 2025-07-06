import torch
import diffusers
from PIL import ImageDraw, ImageFont
import streamlit as st 

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

#This will save the first time the model is loaded and the next time use the save loaded model
@st.cache_resource
def load_model():
    
    MODEL_NAME = "CompVis/stable-diffusion-v1-4"
    pipeline = diffusers.AutoPipelineForText2Image.from_pretrained(
        MODEL_NAME, torch_dtype=dtype, 
    )
        
    OUTPUT_DIR = "rschroll/maya_model_v1_lora"
    
    pipeline.load_lora_weights(
        OUTPUT_DIR, # Directory containing weights file
    
        weight_name="pytorch_lora_weights.safetensors",
    )
    
    return pipeline.to(device)

def generate_images(prompt, pipeline, n):
    images = pipeline([prompt]*n).images
    return images

def add_text_to_image(image, text, text_color="white", outline_color="black",
                      font_size=50, border_width=2, font_path="arial.ttf"):
    # Initialization
    font = ImageFont.truetype(font_path, size=font_size)
    draw = ImageDraw.Draw(image)
    width, height = image.size

    # Calculate the size of the text
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Calculate the position at which to draw the text to center it
    x = (width - text_width) / 2
    y = (height - text_height) / 2

    # Draw text
    draw.text((x, y), text, font=font, fill=text_color,
              stroke_width=border_width, stroke_fill=outline_color)

def generate_memes(prompt, text, pipeline, n):
    images = generate_images(prompt, pipeline, n)
    for img in images:
        add_text_to_image(img, text)
    return images

def main():
    st.title("Diffusion Modal Image Generator")
    with st.sidebar:
        num_images = st.number_input(
            "Number of Images:",
            min_value = 1,
            max_value = 10
        )
        prompt = st.text_area("Text to Generate the desired Image Prompt:")
        text = st.text_area("Text to Display on the Image:")
        generate = st.button("Generate Images")
    if generate:
        if not prompt:
            st.error("Please Enter the Prompt")
        elif not text:
            st.error("Please Enter the Text")
        else:
            with st.spinner("Generating images..."):
                pipeline = load_model()
                images = generate_memes(prompt, text, pipeline, num_images)
                st.subheader("Generated images")
                for im in images:
                    st.image(im)

if __name__ == "__main__":
    main()
    