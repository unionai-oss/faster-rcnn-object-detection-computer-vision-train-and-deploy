
import os
from datetime import timedelta
from union import Artifact, ImageSpec, Resources
from union.app import App, Input, ScalingMetric
from flytekit.extras.accelerators import GPUAccelerator, L4

# Point to your object detection model artifact
FRCCNFineTunedModel = Artifact(name="frccn_fine_tuned_model")

image_spec = ImageSpec(
    name="union-serve-frccn-object-detector",
    packages=[
        "gradio==5.29.0",
        "torch==2.5.1",
        "torchvision==0.20.1",
        "union-runtime>=0.1.18",
        "opencv-python-headless",
    ],
    apt_packages=["ffmpeg", "libsm6", "libxext6"],
    cuda="11.8",
    builder="union",
)

gradio_app = App(
    name="frccn-object-detection-gradio",
    inputs=[
        Input(
            name="downloaded-model",
            value=FRCCNFineTunedModel.query(),
            download=True,
        )
    ],
    container_image=image_spec,
    port=8080,
    include=["./app_main.py"],  # Include your Streamlit code
    args=["python", "app_main.py"],
    limits=Resources(cpu="2", mem="8Gi", gpu="1"),
    requests=Resources(cpu="2", mem="8Gi", gpu="1"),
    accelerator=L4,
    min_replicas=0,
    max_replicas=1,
    scaledown_after=timedelta(minutes=2),
    scaling_metric=ScalingMetric.Concurrency(2),
)

# union deploy apps app.py frccn-object-detection-gradio
