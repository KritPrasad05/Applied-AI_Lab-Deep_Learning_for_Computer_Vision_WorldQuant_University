# Import needed libraries
import streamlit as st

import torchvision
from medigan import Generators
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid
# Task 5.4.2: Fill in the app.py file with imports and model_ids list.

# Define the GAN models available in the app
model_ids = [
    "00001_DCGAN_MMG_CALC_ROI",
    "00002_DCGAN_MMG_MASS_ROI",
    "00003_CYCLEGAN_MMG_DENSITY_FULL",
    "00004_PIX2PIX_MMG_MASSES_W_MASKS",
    "00019_PGGAN_CHEST_XRAY"# New model
]


def main():
    st.title("MEDIGAN Medical Image Data Generator")

    # Add dropdown widget for model selection to the sidebar
    model_id = st.sidebar.selectbox("Select Model ID", model_ids)

    # Add number image selector to the sidebar
    # Task 5.4.3: Add values for the keyword arguments min_value= and max_value= in the st.sidebar.number_input function call in the main function.
    num_images = st.sidebar.number_input(
        "Number of Images", min_value = 1, max_value = 7, value=1, step=1
    )


    # Add generate button to the sidebar
    if st.sidebar.button("Generate Images"):
        # Task 5.4.12: Add the parameters to the generate_images function call in the st.sidebar.button function call in the main function.
        generate_images(num_images, model_id)


# Task 5.4.9: Copy the torch_images function from this notebook to app.py.
def torch_images(num_images, model_id):
    generators = Generators()
    dataloader = generators.get_as_torch_dataloader(
        model_id=model_id,
        install_dependencies=True,
        num_samples=num_images,
        prefetch_factor=None,
    )

    images = []
    for batch_idx, data_dict in enumerate(dataloader):
        image_list = []
        for i in data_dict:
            if "sample" in i:
                sample = data_dict.get("sample")
                if sample.dim() == 4:
                    sample = sample.squeeze(0).permute(2, 0, 1)

                sample = to_pil_image(sample).convert("RGB")
                # Convert the image to a PyTorch tensor
                transform = torchvision.transforms.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                    ]
                )

                # Apply the transform to your PIL image
                sample = transform(sample)
                image_list.append(sample)

            # Preprocess the mask
            if "mask" in i:
                mask = data_dict.get("mask")
                if mask.dim() == 4:
                    mask = mask.squeeze(0).permute(2, 0, 1)
                mask = to_pil_image(mask).convert("RGB")
                mask = transform(mask)
                image_list.append(mask)

        # Organize the grid to have 'sample' images per row
        Grid = make_grid(image_list, nrow=2)

        # Change Grid tensor to be a consistent shape
        # The Grid tensor has shape [1, 128, 128, 1] in some models
        if Grid.dim() == 4:
            # Remove the singleton batch dimension
            Grid = Grid.squeeze(0)
            if Grid.size(-1) == 1:
                # Remove the singleton channel dimension (assuming grayscale)
                Grid = Grid.squeeze(-1)
            else:
                raise ValueError("Expected a single channel (grayscale) image.")

        # Convert the tensor grid to a PIL Image for display
        img = torchvision.transforms.ToPILImage()(Grid)
        images.append(img)
    return images


def generate_images(num_images, model_id):
    st.subheader("Generated Images:")
    # Task 5.4.11: Add the parameters to the torch_images function call in the generate_images function.
    images = torch_images(num_images, model_id)
    for i in range(len(images)):
        # Display generated images in the web app
        st.image(
            images[i],
            caption=f"Generated Image {i+1} (Model ID: {model_id})",
            use_container_width=True,
        )


if __name__ == "__main__":
    # Task 5.4.4: Add a call to the main function in the main block.
    main()
