"""
this module contains the data loading and preprocessing functions
"""
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
    FasterRCNN_ResNet50_FPN_Weights
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

from tasks.helpers import image_to_base64, collate_fn, dataset_dataloader
from containers import image

load_dotenv()

# %% ------------------------------
# Download dataset - task
# --------------------------------
@task(container_image=image,
      enable_deck=True,
      cache=True,
      cache_version="1.333",
      requests=Resources(cpu="2", mem="2Gi")) 

def download_hf_dataset(repo_id: str = 'sagecodes/union_swag_coco',
                        local_dir: str = "dataset",
                        sub_folder: str = "swag") -> FlyteDirectory:
    
    from huggingface_hub import snapshot_download

    if local_dir:
        dataset_dir = os.path.join(local_dir)
        os.makedirs(dataset_dir, exist_ok=True)

    # Download the dataset repository
    repo_path = snapshot_download(repo_id=repo_id, 
                                  repo_type="dataset",
                                  local_dir=local_dir)
    if sub_folder:
        repo_path = os.path.join(repo_path, sub_folder)
        # use sub_folder to return a specific folder from the dataset

    print(f"Dataset downloaded to {repo_path}")

    print(f"Files in dataset directory: {os.listdir(repo_path)}")

    return FlyteDirectory(repo_path)

# %% ------------------------------
# visualize data - task
# --------------------------------
@task(container_image=image,
      enable_deck=True,
      requests=Resources(cpu="2", mem="4Gi"))
def verify_data_and_annotations(dataset_dir: FlyteDirectory) -> FlyteFile:
    
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    # Download the dataset locally from the FlyteDirectory
    dataset_dir.download()
    local_dataset_dir = dataset_dir.path
    
    # Load the dataset
    data_loader = dataset_dataloader(root=local_dataset_dir, annFile="train.json", shuffle=True)
    
    # Number of images to display
    num_images = 9
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))  # Create a 3x3 grid
    axes = axes.flatten()  # Flatten the axes array for easier iteration
    
    images_plotted = 0  # Counter for images plotted

    # Plot images along with annotations
    for batch_idx, (images, targets) in enumerate(data_loader):
        for i, image in enumerate(images):
            if images_plotted >= num_images:
                break  # Limit to 9 images
            
            # Plot the image
            img = image.cpu().permute(1, 2, 0)  # Convert image to HWC format for plotting
            ax = axes[images_plotted]  # Access the correct subplot
            ax.imshow(img)

            # Iterate over the list of annotations (objects) for the current image
            for annotation in targets[i]:
                # Extract the bounding box
                bbox = annotation['bbox']  # This is in [x_min, y_min, width, height] format
                
                # Convert [x_min, y_min, width, height] to [x_min, y_min, x_max, y_max]
                x_min, y_min, width, height = bbox
                x_max = x_min + width
                y_max = y_min + height

                # Draw the bounding box
                rect = patches.Rectangle((x_min, y_min), width, height,
                                         linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)

            # Increment image counter
            images_plotted += 1

        if images_plotted >= num_images:
            break  # Stop if we've plotted the desired number of images

    plt.tight_layout()

    # Save the grid of images and annotations
    output_img = "data_verification_grid.png"
    plt.savefig(output_img)
    plt.close()

    # Convert the image to base64 for display in FlyteDeck
    verification_image_base64 = image_to_base64(output_img)

    # Display the results in FlyteDeck
    ctx = current_context()
    deck = Deck("Data Verification")
    html_report = dedent(f"""
    <div style="font-family: Arial, sans-serif; line-height: 1.6;">
       <h2 style="color: #2C3E50;">Data Verification: Images and Annotations</h2>
        <img src="data:image/png;base64,{verification_image_base64}" width="600">
    </div>
    """)

    # Append the HTML content to the deck
    deck.append(html_report)
    ctx.decks.insert(0, deck)

    # Return the image file for further use in the workflow
    return FlyteFile(output_img)