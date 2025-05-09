
from tasks.data import download_hf_dataset, verify_data_and_annotations
from tasks.model import download_model, train_model, evaluate_model, upload_model_to_hub
from union import workflow
import torch
# %% ------------------------------
# Object Detection Workflow
# --------------------------------
@workflow
def object_detection_workflow(hf_repo_id: str, epochs: int =3, classes:int =3) -> torch.nn.Module:
    
    dataset_dir = download_hf_dataset(repo_id="sagecodes/union_flyte_swag_object_detection")
    model = download_model()
    verify_data_and_annotations(dataset_dir=dataset_dir)
    trained_model = train_model(model=model, dataset_dir=dataset_dir, num_epochs=epochs, num_classes=classes)
    evaluate_model(model=trained_model, dataset_dir=dataset_dir)
    upload_model_to_hub(model=trained_model, repo_name=hf_repo_id)

    return model


# union run --remote workflows/wf_training_detection.py object_detection_workflow --epochs 3 --hf_repo_id "sagecodes/cv-object-mobilenet"