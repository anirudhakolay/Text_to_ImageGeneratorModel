# First, make sure you have all the necessary libraries installed:
# pip install torch diffusers accelerate gradio

import torch
from diffusers import StableDiffusionPipeline
import gradio as gr

# --- 1. Load the AI Model ---
# We load the model only once when the script starts to avoid reloading it
# for every user request, which would be very slow.
print("Loading Stable Diffusion model... This may take a moment.")

# --- QUALITY UPGRADE: Using a more powerful and accurate model ---
# This model produces much higher quality images than the "tiny" version.
model_id = "stabilityai/stable-diffusion-2-1-base"

# We use float16 for memory efficiency.
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16
)

# Check if a CUDA-enabled GPU is available and move the model to the GPU
# for much faster image generation.
if torch.cuda.is_available():
    print("GPU detected. Moving model to CUDA.")
    pipe = pipe.to("cuda")
else:
    print("Warning: CUDA not available. This model will be very slow on a CPU.")
    # For CPU, it's better to work with full precision.
    pipe.to(torch.float32)


# --- 2. Define the Image Generation Function ---
# This is the core function that takes user inputs and returns an image.
def generate_image(prompt, negative_prompt, steps):
    """
    Takes a text prompt, a negative prompt, and the number of inference steps,
    and uses the Stable Diffusion model to generate an image.
    """
    print(f"Received prompt: '{prompt}'")
    print(f"Negative prompt: '{negative_prompt}'")
    print(f"Inference steps: {steps}")
    
    # The 'pipe' object is called like a function. It handles the whole
    # diffusion process. We access the 'images' list and take the first one.
    try:
        # Pass the negative_prompt and num_inference_steps to the pipeline
        image = pipe(
            prompt=prompt, 
            negative_prompt=negative_prompt,
            num_inference_steps=steps
        ).images[0]
        
        print("Image generated successfully.")
        return image
    except Exception as e:
        print(f"An error occurred during image generation: {e}")
        return None


# --- 3. Create the Gradio Web Interface ---
# We use gr.Blocks() for more control over the layout.
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎨 AI Text-to-Image Generator (High Quality) 🎨
        This version uses a powerful model for more accurate and detailed images.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=4):
            # The Textbox is the input field for the user's prompt.
            prompt_input = gr.Textbox(
                label="Your Prompt", 
                placeholder="e.g., A majestic lion wearing a crown, photorealistic",
                lines=2
            )
            negative_prompt_input = gr.Textbox(
                label="Negative Prompt (what to avoid)",
                placeholder="e.g., blurry, ugly, deformed, bad anatomy, extra limbs",
                lines=2
            )
        with gr.Column(scale=1):
             # The button that the user will click to start the generation.
            submit_button = gr.Button("Generate Image", variant="primary", scale=1)

    # Add a slider to control speed vs. quality
    steps_slider = gr.Slider(
        minimum=15, 
        maximum=50, 
        step=1, 
        value=25, 
        label="Inference Steps",
        info="Fewer steps are faster. More steps can improve detail. 25 is a good balance."
    )
    
    # The Image component will display the output from our function.
    output_image = gr.Image(label="Generated Image")

    # --- 4. Connect the Components ---
    # This is the event listener. When the 'submit_button' is clicked,
    # it calls the 'generate_image' function with the inputs
    # and puts the result into 'output_image'.
    submit_button.click(
        fn=generate_image, 
        inputs=[prompt_input, negative_prompt_input, steps_slider], 
        outputs=output_image
    )

# --- 5. Launch the Application ---
# The launch() method starts a local web server.
print("Launching Gradio interface... Open the local URL in your browser.")
demo.launch()