"""
This script downloads a union artifact from the Union platform.

"""
from union import Artifact, UnionRemote
from flytekit.types.file import FlyteFile
import torch

# --------------------------------------------------
# Download & save the fine-tuned model from Union Artifacts
# --------------------------------------------------
FRCCNFineTunedModel = Artifact(name="frccn_fine_tuned_model")

query = FRCCNFineTunedModel.query(
    project="default",
    domain="development",
    # version="anmrqcq8pfbnlp42j2vp/n3/0/o0"  # Optional: specify version. Will download the latest version if not specified
)
remote = UnionRemote()
artifact = remote.get_artifact(query=query)
model_file: FlyteFile = artifact.get(as_type=FlyteFile)
model = torch.load(model_file.download(), map_location="cpu", weights_only=False)

save_dir = "local_frccn_faster_rcnn_trained_union.pth"
torch.save(model, save_dir)
