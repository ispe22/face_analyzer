import base64
import io
import os
import traceback

import gradio as gr
from fastai.vision.all import load_learner
from PIL import Image

# Load model
learn = load_learner("model_18.pkl")


# Print logs as soon as they are logged
def log(*args, **kwargs):
    print(*args, **kwargs, flush=True)


def _decode_data_url(data_url: str) -> Image.Image:
    """Decode data:image/... base64 string to PIL Image."""
    header, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def predict(image_input):
    """Predicts age from an image (URL or PIL) and returns a formatted string."""
    log("Predict called; input type:", type(image_input))
    try:
        if learn is None:
            return []

        # Input handling
        if isinstance(image_input, str) and image_input.startswith("data:image"):
            img = _decode_data_url(image_input)
        elif isinstance(image_input, Image.Image):
            img = image_input
        elif image_input is None:
            return None
        else:
            return []

        log("Running prediction...")
        pred, *_ = learn.predict(img)

        try:
            if hasattr(pred, "__len__") and not isinstance(pred, str):
                candidate = pred[0]
            else:
                candidate = pred
            age_val = float(candidate)
            out = f"Predicted age: {age_val:.1f} years"
        except Exception:
            out = str(pred)

        log("Result:", out)
        return out

    except Exception as e:
        log("ERROR:", e)
        traceback.print_exc()
        return f"Error processing image: {e}"


# UI setup
with gr.Blocks() as iface:
    gr.Markdown("# Face age predictor")
    gr.Markdown(
        "Upload an image using the button below, or use the API endpoint for programmatic access."
    )

    with gr.Row():
        with gr.Column():
            image_uploader = gr.Image(type="pil", label="Upload a face image")
            ui_button = gr.Button("Predict Age from Image", variant="primary")

        with gr.Column():
            result_textbox = gr.Textbox(label="Predicted age")

    api_textbox = gr.Textbox(label="API input (Data URL)", visible=False)

    ui_button.click(fn=predict, inputs=image_uploader, outputs=result_textbox)

    api_textbox.submit(fn=predict, inputs=api_textbox, outputs=result_textbox)

    gr.Examples(
        [["example_face.jpeg"]],
        inputs=image_uploader,
        outputs=result_textbox,
        fn=predict,
    )


if __name__ == "__main__":
    iface.queue()
    port = int(os.environ.get("PORT", 7860))
    iface.launch(server_name="0.0.0.0", server_port=port, share=False, ssr_mode=False)
