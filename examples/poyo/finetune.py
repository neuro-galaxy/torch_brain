import torch
import torch_brain
from torch_brain.models import POYO
from torch_brain.registry import ModalitySpec, DataType


# User defines readout properties
readout_modality_name = "wheel"
readout_value_key = "wheel.vel"
readout_timestamp_key = "wheel.timestamps"

# Helper class method to create minimal dataset config
dataset_config = POYO.create_basic_dataset_config(
    dir_path=".",
    brainset="ibl",
    sessions=[
        "sub-CSH-ZAD-024_ses-8207abc6-6b23-4762-92b4-82e05bed5143-processed-only_behavior"
    ],
    readout_id=readout_modality_name,
    value_key=readout_value_key,
    timestamp_key=readout_timestamp_key,
)

# Define readout specification
# PS: we could probably automatize this using the dataset config
readout_spec = ModalitySpec(
    id=100,
    name=readout_modality_name,
    value_key=readout_value_key,
    timestamp_key=readout_timestamp_key,
    dim=1,
    type=DataType.CONTINUOUS,
    loss_fn=torch_brain.nn.loss.MSELoss(),
)

# Helper class method to load pre-trained model checkpoint
model = POYO.load_pretrained(
    checkpoint_path="poyo_mp.ckpt",
    readout_spec=readout_spec,
    skip_readout=True,
)

# Load model to device
# PS: we could probably automatically do this (by default)
device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
)
model.to(device)

# Helper method to bind dataset to model
model.set_datasets(
    dir_path=".",
    dataset_config=dataset_config,
)

# Helper method finetune model with provided Brainset
poyo_ft_r2_log, poyo_ft_loss_log, poyo_ft_train_outputs = model.finetune(
    num_epochs=2,
    epoch_to_unfreeze=10,
    data_loader_batch_size=16,
)
