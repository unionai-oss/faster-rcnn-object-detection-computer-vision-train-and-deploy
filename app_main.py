# main.py

import time

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.transforms import functional as F

# Load model from artifact or fallback path
try:
    from union_runtime import get_input

    model_path = get_input("downloaded-model")
except:
    model_path = "frccn_fine_tuned_model.pth"

# Load the model
model = torch.load(model_path, map_location="cpu", weights_only=False)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

labels_map = {1: "union", 2: "flyte"}


def detect_objects(frame: np.ndarray) -> np.ndarray:
    start = time.time()

    pil_img = Image.fromarray(frame).convert("RGB").resize((320, 240))
    img_tensor = F.to_tensor(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)

    boxes = outputs[0]["boxes"].cpu()
    scores = outputs[0]["scores"].cpu()
    labels = outputs[0]["labels"].cpu()

    threshold = 0.5
    selected = scores > threshold
    boxes = boxes[selected]
    scores = scores[selected]
    labels = labels[selected]

    draw = ImageDraw.Draw(pil_img)
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.tolist()
        draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
        draw.text(
            (x1, y1),
            f"{labels_map.get(label.item(), label.item())}: {score:.2f}",
            fill="white",
        )

    # Overlay inference time and device info
    end = time.time()
    inference_time = (end - start) * 1000  # ms
    debug_text = f"{device.type.upper()} | {inference_time:.1f} ms"
    draw.rectangle([0, 0, 200, 20], fill=(0, 0, 0, 128))  # semi-transparent background
    draw.text((5, 2), debug_text, fill="white")

    return np.array(pil_img)


# Create Gradio app with upload option
demo = gr.Interface(
    fn=detect_objects,
    inputs=gr.Image(type="numpy", label="Upload Image"),
    outputs=gr.Image(type="numpy", label="Detection Result"),
    title="Union Faster RCNN Object Detection",
    description="Upload an image to run Faster RCNN object detection.",
)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080)

# union deploy apps app.py frccn-object-detection-gradio
