# %%
import matplotlib.patches as patches
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torchvision
from flytekit import task, workflow, ImageSpec, Resources, current_context, Deck, Secret
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.models.detection.faster_rcnn import \
    FasterRCNN_ResNet50_FPN_Weights, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
from torchvision.ops import box_iou
from torchvision.transforms import transforms as T
from flytekit.types.directory import FlyteDirectory
from flytekit.types.file import FlyteFile
import base64
from textwrap import dedent
from pathlib import Path

from datasets import load_dataset
import os
import requests
from dotenv import load_dotenv
from containers import container_image

from tasks.helpers import image_to_base64, collate_fn, dataset_dataloader


load_dotenv()

# %% ------------------------------
# donwload model - task
# --------------------------------
@task(container_image=container_image,
    cache=True,
    cache_version="1.333",
    requests=Resources(cpu="2", mem="2Gi"))
def download_model() -> torch.nn.Module:

    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights, weights_only=True
    )

    return model

# %% ------------------------------
# train model - task
# --------------------------------
@task(container_image=container_image,
    requests=Resources(cpu="2", mem="8Gi", gpu="1"))
def train_model(model: torch.nn.Module, dataset_dir: FlyteDirectory, num_epochs: int, num_classes: int) -> torch.nn.Module:

    # TODO: make from dict
    num_classes = num_classes  # number of classes + background (TODO: add one for the background class automatically)
    num_epochs = num_epochs
    best_mean_iou = 0
    model_dir = "models"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_dir.download()

    os.makedirs(model_dir, exist_ok=True)

    local_dataset_dir = dataset_dir.path  # Use the local path for FlyteDirectory

    data_loader = dataset_dataloader(root=local_dataset_dir, annFile="train.json")
    test_data_loader = dataset_dataloader(root=local_dataset_dir, annFile="train.json")

    # Modify the model to add a new classification head based on the number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = (
        torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
        )
    )

    model.to(device)

    # Define optimizer and learning rate
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Function to filter out and correct invalid boxes
    def filter_and_correct_boxes(targets):
        filtered_targets = []
        for target in targets:
            boxes = target["boxes"]
            labels = target["labels"]
            valid_indices = []
            for i, box in enumerate(boxes):
                if box[2] > box[0] and box[3] > box[1]:
                    valid_indices.append(i)
                else:
                    print(f"Invalid box found and removed: {box}")
            filtered_boxes = boxes[valid_indices]
            filtered_labels = labels[valid_indices]
            filtered_targets.append(
                {"boxes": filtered_boxes, "labels": filtered_labels}
            )
        return filtered_targets

    # Function to evaluate the model
    def evaluate_model(model, data_loader):
        model.eval()
        iou_list, loss_list = [], []
        correct_predictions, total_predictions = 0, 0
        with torch.no_grad():
            for images, targets in data_loader:
                images = [image.to(device) for image in images]
                targets = [
                    {
                        "boxes": torch.tensor(
                            [obj["bbox"] for obj in t], dtype=torch.float32
                        ).to(device),
                        "labels": torch.tensor(
                            [obj["category_id"] for obj in t], dtype=torch.int64
                        ).to(device),
                    }
                    for t in targets
                ]
                for target in targets:
                    boxes = target["boxes"]
                    boxes[:, 2] += boxes[:, 0]
                    boxes[:, 3] += boxes[:, 1]
                    target["boxes"] = boxes

                targets = filter_and_correct_boxes(targets)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                outputs = model(images)

                for i, output in enumerate(outputs):
                    pred_boxes = output["boxes"]
                    true_boxes = targets[i]["boxes"]
                    if pred_boxes.size(0) == 0 or true_boxes.size(0) == 0:
                        continue
                    iou = box_iou(pred_boxes, true_boxes)
                    iou_list.append(iou.mean().item())

                    pred_labels = output["labels"]
                    true_labels = targets[i]["labels"]

                    # Ensure both tensors are the same size for comparison
                    min_size = min(len(pred_labels), len(true_labels))
                    correct_predictions += (
                        (pred_labels[:min_size] == true_labels[:min_size]).sum().item()
                    )
                    total_predictions += min_size

        mean_iou = sum(iou_list) / len(iou_list) if iou_list else 0
        accuracy = correct_predictions / total_predictions if total_predictions else 0
        print(f"Mean IoU: {mean_iou:.4f}, Accuracy: {accuracy:.4f}")
        return mean_iou, accuracy

    for epoch in range(num_epochs):
        model.train()
        for i, (images, targets) in enumerate(data_loader):
            images = [image.to(device) for image in images]
            targets = [
                {
                    "boxes": torch.tensor(
                        [obj["bbox"] for obj in t], dtype=torch.float32
                    ).to(device),
                    "labels": torch.tensor(
                        [obj["category_id"] for obj in t], dtype=torch.int64
                    ).to(device),
                }
                for t in targets
            ]
            for target in targets:
                boxes = target["boxes"]
                boxes[:, 2] += boxes[:, 0]
                boxes[:, 3] += boxes[:, 1]
                target["boxes"] = boxes

            targets = filter_and_correct_boxes(targets)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if i % 10 == 0:
                print(
                    f"Epoch [{epoch}/{num_epochs}], Step [{i}/{len(data_loader)}], Loss: {losses.item():.4f}"
                )

        lr_scheduler.step()

        mean_iou, accuracy = evaluate_model(model, test_data_loader)
        if mean_iou > best_mean_iou:
            best_mean_iou = mean_iou
            torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pth"))
            print("Best model saved")

    print("Training completed.")
    return model


# %% ------------------------------
# evaluate model - task
# ---------------------------------
@task(container_image=container_image,
      enable_deck=True,
      requests=Resources(cpu="2", mem="8Gi", gpu="1"))
def evaluate_model(model: torch.nn.Module, dataset_dir: FlyteDirectory, threshold: float = 0.75) -> str:
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    dataset_dir.download()
    local_dataset_dir = dataset_dir.path
    data_loader = dataset_dataloader(root=local_dataset_dir, 
                                     annFile="train.json", shuffle=False)

    model.to(device)
    model.eval()

    num_images = 9  # Number of images to display in the grid
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))  # Create a 3x3 grid
    axes = axes.flatten()  # Flatten the axes array for easier iteration

    iou_list, accuracy_list = [], []
    report = []  # To store the IoU and accuracy report for each image
    global_image_index = 0  # Global image counter across batches
    images_plotted = 0  # Counter for images plotted in the grid

    correct_predictions, total_predictions = 0, 0

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(data_loader):
            images = [image.to(device) for image in images]
            targets = [
                {
                    "boxes": torch.tensor(
                        [obj["bbox"] for obj in t], dtype=torch.float32
                    ).to(device),
                    "labels": torch.tensor(
                        [obj["category_id"] for obj in t], dtype=torch.int64
                    ).to(device),
                }
                for t in targets
            ]
            for target in targets:
                boxes = target["boxes"]
                boxes[:, 2] += boxes[:, 0]  # Convert width to x_max
                boxes[:, 3] += boxes[:, 1]  # Convert height to y_max
                target["boxes"] = boxes

            outputs = model(images)

            for i, output in enumerate(outputs):
                pred_boxes = output["boxes"]
                pred_scores = output["scores"]
                pred_labels = output["labels"]
                true_boxes = targets[i]["boxes"]
                true_labels = targets[i]["labels"]

                # Filter predictions by confidence threshold
                high_conf_indices = pred_scores > threshold
                pred_boxes = pred_boxes[high_conf_indices]
                pred_labels = pred_labels[high_conf_indices]

                # Get the global image index
                image_index = global_image_index + i

                if pred_boxes.size(0) == 0 or true_boxes.size(0) == 0:
                    report.append(f"Image {image_index}: No valid predictions or ground truths")
                    continue

                # Calculate IoU and match predictions to ground truth based on IoU
                iou = box_iou(pred_boxes, true_boxes)
                max_iou_indices = iou.argmax(dim=1)  # Find the best matching true box for each predicted box

                # Calculate accuracy based on matching boxes
                matched_true_labels = true_labels[max_iou_indices]  # Match true labels with best IoU
                correct_predictions += (pred_labels == matched_true_labels).sum().item()
                total_predictions += len(pred_labels)

                # Calculate mean IoU
                mean_iou = iou.max(dim=1)[0].mean().item()  # Get the highest IoU for each predicted box
                iou_list.append(mean_iou)

                # Append report for this image
                accuracy = correct_predictions / total_predictions if total_predictions else 0
                report.append(f"Image {image_index}: IoU = {mean_iou:.4f}, Accuracy = {accuracy:.4f}")

                # Plot images (limit to num_images)
                if images_plotted < num_images:
                    img = images[i].cpu().permute(1, 2, 0)  # Convert image to HWC format for plotting
                    ax = axes[images_plotted]  # Access the correct subplot

                    ax.imshow(img)
                    for j in range(len(pred_boxes)):
                        bbox = pred_boxes[j].cpu().numpy()
                        score = pred_scores[high_conf_indices][j].cpu().item()
                        label = pred_labels[j].cpu().item()

                        if score > threshold:  # Only display predictions with confidence score above threshold
                            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                                                     linewidth=2, edgecolor='r', facecolor='none')
                            ax.add_patch(rect)
                            ax.text(bbox[0], bbox[1], f"{label}: {score:.2f}", color="white", fontsize=8,
                                    bbox=dict(facecolor="red", alpha=0.5))
                    ax.axis('off')  # Hide axes
                    images_plotted += 1

            # Update global image index after processing the batch
            global_image_index += len(images)

            if images_plotted >= num_images:  # Break once we've plotted 9 images
                break

    # Compute overall metrics
    overall_iou = sum(iou_list) / len(iou_list) if iou_list else 0
    overall_accuracy = correct_predictions / total_predictions if total_predictions else 0

    # Save the image grid
    pred_boxes_imgs = "prediction_grid.png"
    plt.tight_layout()
    plt.savefig(pred_boxes_imgs)
    plt.close()

    train_image_base64 = image_to_base64(pred_boxes_imgs)

    # Prepare the report as text
    report_text = "\n".join(report)
    overall_report = dedent(f"""
    Overall Metrics:
    ----------------
    Mean IoU: {overall_iou:.4f}
    Mean Accuracy: {overall_accuracy:.4f}

    Per-Image Metrics:
    ------------------
    {report_text}
    """)

    # Display the report in FlyteDeck
    ctx = current_context()
    deck = Deck("Evaluation Results")
    html_report = dedent(f"""
    <div style="font-family: Arial, sans-serif; line-height: 1.6;">
       <h2 style="color: #2C3E50;">Predicted Bounding Boxes</h2>
        <img src="data:image/png;base64,{train_image_base64}" width="600">
    </div>               
    <div style="font-family: Arial, sans-serif; line-height: 1.6;">
        <h2 style="color: #2C3E50;">Evaluation Report</h2>
        <pre>{overall_report}</pre>
    </div>

    """)

    # Append HTML content to the deck
    deck.append(html_report)
    ctx.decks.insert(0, deck)

    return overall_report


# %% ------------------------------
# upload model to hub - task
# --------------------------------
@task(
    container_image=container_image,
    requests=Resources(cpu="2", mem="2Gi"),
    secret_requests=[Secret(group=None, key="hf_token")],
)
def upload_model_to_hub(model: torch.nn.Module, repo_name: str) -> str:
    from huggingface_hub import HfApi
    # Get the Flyte context and define the model path
    ctx = current_context()
    model_path = "best_model.pth"  # Save the model locally as "best_model.pth"

    # Save the model's state dictionary
    torch.save(model.state_dict(), model_path)

    # Set Hugging Face token from local environment or Flyte secrets
    hf_token = os.getenv("HF_TOKEN")
    if hf_token is None:
        # If HF_TOKEN is not found, attempt to get it from the Flyte secrets
        hf_token = ctx.secrets.get(key="hf_token")
        print("Using Hugging Face token from Flyte secrets.")
    else:
        print("Using Hugging Face token from environment variable.")

    # Create a new repository (if it doesn't exist) on Hugging Face Hub
    api = HfApi()
    api.create_repo(repo_name, token=hf_token, exist_ok=True)

    # Upload the model to the Hugging Face repository
    api.upload_file(
        path_or_fileobj=model_path,      # Path to the local file
        path_in_repo="pytorch_model.bin", # Destination path in the repo
        repo_id=repo_name,
        commit_message="Upload Faster R-CNN model",
        token=hf_token
    )

    return f"Model uploaded to Hugging Face Hub: https://huggingface.co/{repo_name}"