import base64
import os

from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.transforms import transforms as T

# %% ------------------------------
# helper functions
# --------------------------------


# Convert images to base64 and embed in HTML
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def collate_fn(batch):
    return tuple(zip(*batch))


def dataset_dataloader(
    root: str,
    annFile: str,
    batch_size=2,
    shuffle=True,
    num_workers=0,
) -> DataLoader:
    # Define the transformations for the images
    transform = T.Compose([T.ToTensor()])

    annFile_path = os.path.join(str(root), annFile)
    # Load the dataset
    dataset = CocoDetection(root=root, annFile=annFile_path, transform=transform)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    return data_loader
