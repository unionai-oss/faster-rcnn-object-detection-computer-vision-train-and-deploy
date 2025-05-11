import cv2
import numpy as np
import requests
import torch
from flytekit.types.file import FlyteFile
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional as F
from union import Artifact, UnionRemote
from io import BytesIO

# Define labels map
labels_map = {1: "Union Sticker",
               2: "Flyte Sticker"}


# ---------------------------------
# Draw bounding boxes on images
# ---------------------------------
def draw_boxes(image, boxes, labels, scores, labels_map, threshold=0.75):
    draw = ImageDraw.Draw(image, "RGBA")

    font_url = "https://github.com/google/fonts/raw/refs/heads/main/apache/ultra/Ultra-Regular.ttf"
    response = requests.get(font_url)
    font = ImageFont.truetype(BytesIO(response.content), size=20)

    # font = ImageFont.truetype(urlopen(truetype_url), size=20)
    # font = ImageFont.load_default() # default font in pil

    colors = {
        0: (255, 173, 10, 200),  # Class 0 color
        1: (28, 140, 252, 200),  # Class 1 color
    }
    colors_fill = {
        0: (255, 173, 10, 100),  # Class 0 fill color
        1: (28, 140, 252, 100),  # Class 1 fill color
    }

    for box, label, score in zip(boxes, labels, scores):
        if score > threshold:  # adjust threshold as needed
            color = colors.get(label, (0, 255, 0, 200))
            fill_color = colors_fill.get(label, (0, 255, 0, 100))
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline=color, width=3)
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], fill=fill_color)
            label_text = f"{labels_map[label]}: {score:.2f}"
            text_size = font.getbbox(label_text)
            draw.rectangle(
                [(box[0], box[1] - text_size[1]), (box[0] + text_size[0], box[1])],
                fill=color,
            )
            draw.text(
                (box[0], box[1] - text_size[1]), label_text, fill="white", font=font
            )

    return image


# --------------------------------------------------
# Load the fine-tuned SSD model from Union Artifact
# --------------------------------------------------
FRCCNFineTunedModel = Artifact(name="frccn_fine_tuned_model")
query = FRCCNFineTunedModel.query(
    project="default",
    domain="development",
    # version="anmrqcq8pfbnlp42j2vp/n3/0/o0"  # Optional: specify version
)
remote = UnionRemote()
artifact = remote.get_artifact(query=query)
model_file: FlyteFile = artifact.get(as_type=FlyteFile)
model = torch.load(model_file.download(), map_location="cpu", weights_only=False)

model.eval()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)


# ------------------------------------
# create video writer
# ------------------------------------

# Video path and properties
video_path = "dataset/swag/videos/union_sticker_video.mp4"
video = cv2.VideoCapture(video_path)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
frames_per_second = video.get(cv2.CAP_PROP_FPS)
num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))


# Initialize video writer
video_writer = cv2.VideoWriter(
    "out.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps=float(frames_per_second),
    frameSize=(width, height),
    isColor=True,
)


def run_inference_video(video, model, device, labels_map):
    while True:
        hasFrame, frame = video.read()
        if not hasFrame:
            break

        # Convert frame to PIL image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image_tensor = F.to_tensor(image).unsqueeze(0).to(device)

        # Run inference
        with torch.no_grad():
            outputs = model(image_tensor)

        # Get the boxes, labels, and scores
        boxes = outputs[0]["boxes"].cpu().numpy()
        labels = outputs[0]["labels"].cpu().numpy()
        scores = outputs[0]["scores"].cpu().numpy()

        # Draw the boxes on the image
        image_with_boxes = draw_boxes(image, boxes, labels, scores, labels_map)

        # Convert back to OpenCV image format
        result_frame = cv2.cvtColor(np.array(image_with_boxes), cv2.COLOR_RGB2BGR)

        yield result_frame


# Run inference and write video
for frame in run_inference_video(video, model, device, labels_map):
    video_writer.write(frame)

# Release resources
video.release()
video_writer.release()
