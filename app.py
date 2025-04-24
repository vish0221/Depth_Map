import glob
import gradio as gr
import matplotlib
import numpy as np
from PIL import Image
import torch
import tempfile
from gradio_imageslider import ImageSlider
from depth_anything_v2.dpt import DepthAnythingV2

# CSS for styling the Gradio interface
css = """
#img-display-container {
    max-height: 100vh;
}
#img-display-input {
    max-height: 80vh;
}
#img-display-output {
    max-height: 80vh;
}
#download {
    height: 62px;
}
"""

# Determine the device to use (GPU or CPU)
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# Model configurations
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

# Select the encoder
encoder = 'vitl'
model = DepthAnythingV2(**model_configs[encoder])

# Load the model state
try:
    state_dict = torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(DEVICE).eval()
    print(f"Model loaded successfully on device: {DEVICE}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)  # Exit if the model cannot be loaded

# Gradio interface title and description
title = "# Depth Anything V2"
description = """Official demo for **Depth Anything V2**.
Please refer to our [paper](https://arxiv.org/abs/2406.09414), [project page](https://depth-anything-v2.github.io), or [github](https://github.com/DepthAnything/Depth-Anything-V2) for more details."""

# Function to predict depth
def predict_depth(image):
    try:
        # Ensure the image is in the correct format (HWC)
        if image.ndim == 3 and image.shape[2] == 3:
            depth = model.infer_image(image[:, :, ::-1])  # Convert BGR to RGB
            return depth
        else:
            print("Input image is not in the expected format.")
            return None
    except Exception as e:
        print(f"Error during depth prediction: {e}")
        return None

# Gradio Blocks interface
with gr.Blocks(css=css) as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    gr.Markdown("### Depth Prediction demo")

    with gr.Row():
        input_image = gr.Image(label="Input Image", type='numpy', elem_id='img-display-input')
        depth_image_slider = ImageSlider(label="Depth Map with Slider View", elem_id='img-display-output', position=0.5)
    
    submit = gr.Button(value="Compute Depth")
    gray_depth_file = gr.File(label="Grayscale depth map", elem_id="download")
    raw_file = gr.File(label="16-bit raw output (can be considered as disparity)", elem_id="download")

    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    def on_submit(image):
        try:
            original_image = image.copy()
            depth = predict_depth(image)  # Directly use the image

            if depth is None:
                print("Depth prediction failed.")
                return None, None, None  # Handle prediction error

            # Process depth output
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0  # Normalize depth
            depth = depth.astype(np.uint8)
            colored_depth = (cmap(depth)[:, :, :3] * 255).astype(np.uint8)

            # Create images for output
            gray_depth = Image.fromarray(depth)
            tmp_gray_depth = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            gray_depth.save(tmp_gray_depth.name)

            colored_depth_image = Image.fromarray(colored_depth)
            tmp_colored_depth = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            colored_depth_image.save(tmp_colored_depth.name)

            # Save raw 16-bit depth
            raw_depth = (predict_depth(original_image) * 256.0).astype(np.uint16)
            raw_depth_image = Image.fromarray(raw_depth)
            tmp_raw_depth = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            raw_depth_image.save(tmp_raw_depth.name)

            return original_image, tmp_colored_depth.name, tmp_gray_depth.name, tmp_raw_depth.name
        except Exception as e:
            print(f"Error in submission function: {e}")
            return None, None, None, None

    submit.click(fn=on_submit, inputs=[input_image], outputs=[input_image, depth_image_slider, gray_depth_file, raw_file])

# Launch the demo
if __name__ == "__main__":
    demo.launch()

