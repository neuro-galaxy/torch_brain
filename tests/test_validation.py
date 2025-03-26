# from pathlib import Path

# import torch
# import pytest
# import lightning
# import util
# from torch.utils.data import DataLoader

# from torch_brain.data import Dataset
# from torch_brain.data.sampler import SequentialFixedWindowSampler
# from torch_brain.data.collate import collate
# from torch_brain.models import POYOPlus, POYOPlusTokenizer
# from torch_brain.taxonomy import decoder_registry
# from torch_brain.utils import train_wrapper, validation_wrapper

# import util

# DATA_ROOT = Path(util.get_data_paths()["processed_dir"]) / "processed"


# # this test only passes when only one gpu is visible
# def test_validation():
#     model = POYOPlus(
#         task_specs=decoder_registry,
#         backend_config="cpu",
#     )

#     tokenizer = POYOPlusTokenizer(
#         unit_tokenizer=model.unit_emb.tokenizer,
#         session_tokenizer=model.session_emb.tokenizer,
#         decoder_registry=decoder_registry,
#         latent_step=1.0 / 8,
#         num_latents_per_step=4,
#         batch_type=["stacked", "stacked", "stacked"],
#         eval=True,
#     )

#     ds = Dataset(
#         DATA_ROOT,
#         "valid",
#         [
#             {
#                 "selection": [
#                     {
#                         "dandiset": "mc_maze_small",
#                         "session": "jenkins_20090928_maze",
#                     }
#                 ],
#                 "config": {
#                     "multitask_readout": [
#                         {
#                             "decoder_id": "ARMVELOCITY2D",
#                             "weight": 2.0,
#                             "subtask_key": None,
#                             "metrics": [
#                                 {
#                                     "metric": "r2",
#                                     "task": "REACHING",
#                                 },
#                             ],
#                         }
#                     ],
#                 },
#             }
#         ],
#         transform=tokenizer,
#     )

#     model.unit_emb.initialize_vocab(ds.unit_ids)
#     model.session_emb.initialize_vocab(ds.session_ids)

#     sampler = SequentialFixedWindowSampler(
#         sampling_intervals=ds.get_sampling_intervals(),
#         window_length=1.0,
#     )
#     assert len(sampler) > 0

#     loader = DataLoader(ds, collate_fn=collate, batch_size=16, sampler=sampler)

#     device = "cpu"

#     model = model.to(device)

#     wrapper = train_wrapper.TrainWrapper(
#         model=model,
#         optimizer=None,
#         scheduler=None,
#     )
#     wrapper.to(device)

#     trainer = lightning.Trainer(accelerator=device)

#     validator = train_wrapper.CustomValidator(loader)
#     metrics_df = validator.on_validation_epoch_start(trainer, wrapper)

#     assert (
#         "val_mc_maze_small/jenkins_20090928_maze_armvelocity2d_r2"
#         == metrics_df.iloc[0].metric
#     )


# def avg_pool_naive(timestamps: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
#     unique_timestamps, indices = torch.unique(timestamps, return_inverse=True)
#     averages = torch.zeros((len(unique_timestamps), *values.shape[1:]))

#     for i in range(len(unique_timestamps)):
#         group_values = values[indices == i]

#         if group_values.dtype == torch.long:
#             averages[i] = torch.mode(group_values, dim=0).values
#         else:
#             averages[i] = torch.mean(group_values, dim=0)
#     return averages


# def test_avg_pool():
#     # Test cases with different input shapes and data types
#     test_cases = [
#         # case: ground truth, float32. Should meanpool.
#         (
#             torch.tensor([0, 0, 1, 1, 2]),
#             torch.tensor([[1.5], [2.5], [3.9], [4.0], [5.0]]),
#         ),
#         # case: ground truth, int64. Should modepool.
#         (torch.tensor([0, 0, 1, 1, 2]), torch.tensor([[1], [2], [3], [4], [5]])),
#         # case: pred tensors with 2 classes. Should meanpool.
#         (
#             torch.tensor([0, 0, 1, 1, 2]),
#             torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]),
#         ),
#     ]

#     for timestamps, values in test_cases:
#         # Fast implementation
#         fast_output = validation_wrapper.avg_pool(timestamps, values)
#         # Naive implementation
#         naive_output = avg_pool_naive(timestamps, values)
#         if values.dtype == torch.long:
#             fast_output = fast_output.long()
#             naive_output = naive_output.long()
#         # Compare the outputs
#         assert torch.allclose(
#             fast_output, naive_output, atol=1e-6
#         ), f"Fast and naive outputs do not match for inputs:\ntimestamps: {timestamps}\nvalues: {values}"


# def avg_pool_naive(timestamps: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
#     unique_timestamps, indices = torch.unique(timestamps, return_inverse=True)
#     averages = torch.zeros((len(unique_timestamps), *values.shape[1:]))

#     for i in range(len(unique_timestamps)):
#         group_values = values[indices == i]

#         if group_values.dtype == torch.long:
#             averages[i] = torch.mode(group_values, dim=0).values
#         else:
#             averages[i] = torch.mean(group_values, dim=0)
#     return averages


# def test_avg_pool():
#     # Test cases with different input shapes and data types
#     test_cases = [
#         # case: ground truth, float32. Should meanpool.
#         (
#             torch.tensor([0, 0, 1, 1, 2]),
#             torch.tensor([[1.5], [2.5], [3.9], [4.0], [5.0]]),
#         ),
#         # case: ground truth, int64. Should modepool.
#         (torch.tensor([0, 0, 1, 1, 2]), torch.tensor([[1], [2], [3], [4], [5]])),
#         # case: pred tensors with 2 classes. Should meanpool.
#         (
#             torch.tensor([0, 0, 1, 1, 2]),
#             torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]),
#         ),
#     ]

#     for timestamps, values in test_cases:
#         # Fast implementation
#         fast_output = validation_wrapper.avg_pool(timestamps, values)
#         # Naive implementation
#         naive_output = avg_pool_naive(timestamps, values)
#         if values.dtype == torch.long:
#             fast_output = fast_output.long()
#             naive_output = naive_output.long()
#         # Compare the outputs
#         assert torch.allclose(
#             fast_output, naive_output, atol=1e-6
#         ), f"Fast and naive outputs do not match for inputs:\ntimestamps: {timestamps}\nvalues: {values}"


# # def test_validation_willett():
# #     ds = Dataset(
# #         DATA_ROOT,
# #         "valid",
# #         OmegaConf.create([
# #             {
# #                 "selection": {
# #                     "dandiset": "willett_shenoy",
# #                     "session": "willett_shenoy_t5/t5.2019.05.08_single_letters",
# #                 },
# #                 "metrics": [{"output_key": "WRITING_CHARACTER"}],
# #             }
# #         ]),
# #     )
# #     batch_size = 16

# #     od = OrderedDict({x: 1 for x in ds.unit_names})
# #     vocab = torchtext.vocab.vocab(od, specials=["NA"])

# #     collate_fn = Collate(
# #         num_latents_per_step=16,  # This was tied in train_poyo_1.py
# #         step=1.0 / 8,
# #         sequence_length=1.0,
# #         unit_vocab=vocab,
# #         decoder_registry=decoder_registry,
# #         weight_registry=weight_registry,
# #     )

# #     model = PerceiverNM(
# #         unit_vocab=vocab,
# #         session_names=ds.session_names,
# #         num_latents=16,
# #         task_specs=decoder_registry
# #     )
# #     model = model.to('cuda')

# #     wrapper = train_wrapper.TrainWrapper(
# #         model=model,
# #         optimizer=None,
# #         scheduler=None,
# #     )
# #     wrapper.to('cuda')

# #     trainer = lightning.Trainer(accelerator="cuda")

# #     validator = train_wrapper.CustomValidator(ds, collate_fn)
# #     r2 = validator.on_validation_epoch_start(trainer, wrapper)
# #     assert "bce_writing_character" in r2.columns
