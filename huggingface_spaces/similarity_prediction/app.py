import base64
import io
import os
import pathlib
import traceback
import zipfile

import gradio as gr
import numpy as np
from fastai.vision.all import PILImage, load_learner
from PIL import Image

# Unzip images on startup
with zipfile.ZipFile("celeb_samples_all.zip", "r") as zip_ref:
    zip_ref.extractall(".")

CELEB_IMAGES_DIR = "celeb_samples_all" if os.path.exists("celeb_samples_all") else "."
MODEL_PATH = "best_resnet18_celeb_model_8630_classes.pkl"

#  Load model
learn = load_learner(MODEL_PATH)


# Print logs as soon as they are logged
def log(*args, **kwargs):
    print(*args, **kwargs, flush=True)


def _decode_data_url(data_url: str) -> Image.Image:
    """Decode data:image/... base64 string to PIL Image."""
    header, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def predict(image_input):
    """
    Returns a list of tuples for the gallery: [(image, "caption"), (image, "caption")]
    """
    log("Predict called; input type:", type(image_input))
    try:
        if learn is None:
            return []

        #  Input handling
        if isinstance(image_input, str) and image_input.startswith("data:image"):
            img = _decode_data_url(image_input)
        elif isinstance(image_input, Image.Image):
            img = image_input
        elif image_input is None:
            return None
        else:
            return []

        #  Inference
        log("Running prediction...")

        # Convert PIL image to numpy array to avoid 'default_collate' error
        fastai_img = PILImage.create(img)

        pred_class, pred_idx, probs = learn.predict(fastai_img)

        #  Get top 5 matches
        top_k = 5
        top_indices = probs.argsort(descending=True)[:top_k]

        gallery_items = []

        for idx in top_indices:
            name = learn.dls.vocab[idx]
            score = float(probs[idx])

            # Construct path to the reference image, e.g., celeb_samples/Brad_Pitt.jpg
            ref_path = os.path.join(CELEB_IMAGES_DIR, f"{name}.jpg")

            caption = f"{name} ({score * 100:.1f}%)"

            # Load the celebrity image
            if os.path.exists(ref_path):
                celeb_img = Image.open(ref_path)
                gallery_items.append((celeb_img, caption))
            else:
                # If image is missing, just show the name as caption with image replaced by black square
                placeholder = Image.new("RGB", (224, 224), color="gray")
                gallery_items.append((placeholder, f"{caption} (Image Missing)"))

        log(f"Found {len(gallery_items)} matches.")
        return gallery_items

    except Exception as e:
        log("ERROR:", e)
        traceback.print_exc()
        return []


#  UI setup
with gr.Blocks() as iface:
    gr.Markdown("# Celebrity Lookalike Finder")
    gr.Markdown("Upload a photo to see who you look like.")

    with gr.Row():
        with gr.Column(scale=1):
            image_uploader = gr.Image(type="pil", label="Your photo")
            ui_button = gr.Button("Find lookalike", variant="primary")

        with gr.Column(scale=2):
            result_gallery = gr.Gallery(
                label="Top matches",
                columns=5,
                rows=1,
                height="auto",
                object_fit="contain",
            )

    # API endpoint (hidden)
    api_textbox = gr.Textbox(label="API input (data URL)", visible=False)

    #  Connect components
    ui_button.click(fn=predict, inputs=image_uploader, outputs=result_gallery)

    api_textbox.submit(fn=predict, inputs=api_textbox, outputs=result_gallery)

    gr.Examples(
        examples=[["example_face.jpeg"]],
        inputs=image_uploader,
        outputs=result_gallery,
        fn=predict,
    )

if __name__ == "__main__":
    iface.queue()
    port = int(os.environ.get("PORT", 7860))
    iface.launch(server_name="0.0.0.0", server_port=port, share=False, ssr_mode=False)
