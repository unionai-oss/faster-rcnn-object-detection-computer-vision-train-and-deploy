import torch
import cv2
import time
from torchvision.transforms import functional as F
from huggingface_hub import hf_hub_download
from union import Artifact, UnionRemote
from flytekit.types.file import FlyteFile

# --------------------------------------------------
# Load the fine-tuned SSD model from Union Artifact
# --------------------------------------------------
SSDFineTunedModel = Artifact(name="frccn_fine_tuned_model")
query = SSDFineTunedModel.query(
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

# --------------------------------------------------
# Function to process a single frame and draw bounding boxes
# --------------------------------------------------
def process_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_tensor = F.to_tensor(frame_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(image_tensor)

    boxes = prediction[0]['boxes'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()

    for i, box in enumerate(boxes):
        if scores[i] > 0.3:
            x_min, y_min, x_max, y_max = box
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
            label = f"Class {labels[i]}: {scores[i]:.2f}"
            cv2.putText(frame, label, (int(x_min), int(y_min) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame

# --------------------------------------------------
# Run feed with frame skipping option for efficiency
# --------------------------------------------------
def run_video_feed(skip_frames=5):
    frame_skip = skip_frames
    frame_count = 0
    last_processed_frame = None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        if frame_count % frame_skip == 0:
            last_processed_frame = process_frame(frame)
            fps_text = f"FPS: {fps:.2f}"
            cv2.putText(last_processed_frame, fps_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        if last_processed_frame is not None:
            cv2.imshow('Object Detection SSD', last_processed_frame)

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the video feed function
if __name__ == "__main__":
    run_video_feed()
