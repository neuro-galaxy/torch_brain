# from pathlib import Path

# import pytest
# import torch
# import util
# from torch.utils.data import DataLoader

# from torch_brain.data import Dataset
# from torch_brain.data.sampler import SequentialFixedWindowSampler
# from torch_brain.data.collate import collate
# from torch_brain.models import POYOPlus, POYOPlusTokenizer
# from torch_brain.nn import InfiniteVocabEmbedding
# from brainsets_utils.taxonomy import decoder_registry
# from torch_brain.utils import move_to

# DATA_ROOT = Path(util.get_data_paths()["processed_dir"]) / "processed"


# def test_load_real_data():
#     ds = Dataset(DATA_ROOT, "train", [{"selection": [{"dandiset": "mc_maze_small"}]}])
#     sampling_intervals = ds.session_dict["mc_maze_small/jenkins_20090928_maze"][
#         "sampling_intervals"
#     ]
#     assert (
#         ds.get(
#             "mc_maze_small/jenkins_20090928_maze",
#             sampling_intervals.start[0],
#             sampling_intervals.end[0],
#         ).start
#         >= 0
#     )


# def test_collate_data():
#     unit_emb = InfiniteVocabEmbedding(2)
#     sess_emb = InfiniteVocabEmbedding(2)

#     tokenizer = POYOPlusTokenizer(
#         unit_tokenizer=unit_emb.tokenizer,
#         session_tokenizer=sess_emb.tokenizer,
#         decoder_registry=decoder_registry,
#         latent_step=1.0 / 8,
#         num_latents_per_step=4,
#         batch_type=["stacked", "stacked", "stacked"],
#     )

#     ds = Dataset(
#         DATA_ROOT,
#         "train",
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
#                         }
#                     ],
#                 },
#             }
#         ],
#         transform=tokenizer,
#     )

#     unit_emb.initialize_vocab(ds.unit_ids)
#     sess_emb.initialize_vocab(ds.session_ids)

#     sampler = SequentialFixedWindowSampler(
#         sampling_intervals=ds.get_sampling_intervals(),
#         window_length=1.0,
#     )
#     assert len(sampler) > 0

#     train_loader = DataLoader(ds, collate_fn=collate, batch_size=16, sampler=sampler)
#     for data in train_loader:
#         assert data["spike_timestamps"].shape[0] == 16
#         break


# # def test_collate_data_willett():
# #     print("test_collate_data_willett")
# #     ds = Dataset(
# #         DATA_ROOT,
# #         "train",
# #         [
# #             {
# #                 "selection": {
# #                     "dandiset": "willett_shenoy",
# #                     "sortset": "willett_shenoy_t5/t5.2019.05.08",
# #                 },
# #                 "metrics": [{"output_key": "WRITING_CHARACTER"}],
# #             }
# #         ],
# #     )
# #     assert len(ds) > 0

# #     od = OrderedDict({x: 1 for x in ds.unit_names})
# #     vocab = torchtext.vocab.vocab(od, specials=["NA"])

# #     collate_fn = Collate(
# #         num_latents_per_step=128,  # This was tied in train_poyo_1.py
# #         step=1.0 / 8,
# #         sequence_length=128,
# #         unit_vocab=vocab,
# #         decoder_registry=decoder_registry,
# #         weight_registry=weight_registry,
# #     )
# #     train_loader = DataLoader(
# #         ds, collate_fn=collate_fn, batch_size=4, drop_last=True, shuffle=True
# #     )
# #     for i, data in enumerate(train_loader):
# #         print(i)
# #         assert data["spike_timestamps"].shape[0] == 4
# #         npt.assert_allclose(
# #             data["output_weights"]["WRITING_CHARACTER"].detach().cpu().numpy(),
# #             1.0,
# #         )


# # def test_collate_data_perich():
# #     ds = Dataset(
# #         DATA_ROOT,
# #         "train",
# #         [
# #             {
# #                 "selection": {
# #                     "dandiset": "perich_miller",
# #                     "sortset": "chewie_20161013",
# #                 },
# #                 "metrics": [{"output_key": "CURSOR2D"}],
# #             }
# #         ],
# #     )
# #     assert len(ds) > 0

# #     od = OrderedDict({x: 1 for x in ds.unit_names})
# #     vocab = torchtext.vocab.vocab(od, specials=["NA"])

# #     collate_fn = Collate(
# #         num_latents_per_step=128,  # This was tied in train_poyo_1.py
# #         step=1.0 / 8,
# #         sequence_length=128,
# #         unit_vocab=vocab,
# #         decoder_registry=decoder_registry,
# #         weight_registry=weight_registry,
# #     )
# #     train_loader = DataLoader(ds, collate_fn=collate_fn, batch_size=16)
# #     for data in train_loader:
# #         assert data["spike_timestamps"].shape[0] == 16
# #         npt.assert_allclose(
# #             data["output_weights"]["CURSOR2D"].detach().cpu().numpy().max(),
# #             50.0,
# #         )
# #         npt.assert_allclose(
# #             data["output_weights"]["CURSOR2D"].detach().cpu().numpy().min(),
# #             1.0,
# #         )
# #         break


# def test_collated_data_model():
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
#     )

#     ds = Dataset(
#         DATA_ROOT,
#         "train",
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

#     train_loader = DataLoader(ds, collate_fn=collate, batch_size=16, sampler=sampler)
#     device = "cpu"

#     model = model.to(device)
#     assert len(train_loader) > 0
#     for data in train_loader:
#         data = move_to(data, device)
#         output, loss, losses_taskwise = model(**data)

#         assert len(output) == 16
#         assert loss.shape == torch.Size([])  # it should be a scalar
#         assert len(losses_taskwise.keys()) == len(data["output_values"].keys())
#         break
