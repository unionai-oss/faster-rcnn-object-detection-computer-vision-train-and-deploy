from union import workflow

from tasks.data import download_hf_dataset, verify_data_and_annotations
from tasks.model import download_model, evaluate_model, train_model, upload_model_to_hub


# %% ------------------------------
# Object Detection Workflow
# --------------------------------
@workflow
def faster_rcnn_train_workflow(
    epochs: int = 3, classes: int = 3, hf_repo_id: str = ""
) -> None:

    dataset_dir = download_hf_dataset(
        repo_id="sagecodes/union_flyte_swag_object_detection"
    )
    model_file = download_model()
    verify_data_and_annotations(dataset_dir=dataset_dir)
    trained_model = train_model(
        model_file=model_file,
        dataset_dir=dataset_dir,
        num_epochs=epochs,
        num_classes=classes,
    )
    evaluate_model(model=trained_model, dataset_dir=dataset_dir)
    # upload_model_to_hub(model=trained_model, repo_name=hf_repo_id) # uncomment to upload the model to Hugging Face Hub


# union run --remote workflows/train-frcnn-pipeline.py faster_rcnn_train_workflow --epochs 3
# union run --remote workflows/train-frcnn-pipeline.py faster_rcnn_train_workflow --epochs 3 --hf_repo_id "sagecodes/cv-object-rcnn"
