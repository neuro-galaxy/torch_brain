import torch
import torch

from lightning.pytorch.callbacks import Callback


class UnitEmbeddingGradientRescaling(Callback):
    def __init__(self, dataset):
        super().__init__()

        session_weights = {"NA": 1.0}
        # session_info = dataset.session_info_dict[session_id]
        for session_id, session_info in dataset.session_info_dict.items():
            assert len(session_info["config"]["multitask_readout"]) == 1
            unit_weight = session_info["config"]["multitask_readout"][0]["weight"]
            session_weights[session_id] = 1.0 / unit_weight

        unit_ids = dataset.unit_ids
        session_ids = dataset.session_ids

        unit_weight = torch.ones(1 + len(unit_ids))
        for unit_idx, unit_id in enumerate(unit_ids):
            for session_idx, session_id in enumerate(session_ids):
                if session_id in unit_id:
                    unit_weight[unit_idx + 1] = session_weights[session_id]
                    break

        session_weight = torch.ones(1 + len(session_ids))
        for session_idx, session_id in enumerate(session_ids):
            session_weight[session_idx + 1] = session_weights[session_id]

        self.unit_weight = unit_weight
        self.session_weight = session_weight

    def on_after_backward(self, trainer, pl_module):
        # assert len(self.unit_weight) == len(pl_module.model.unit_emb.weight)

        pl_module.model.unit_emb.weight.grad.mul_(
            self.unit_weight.view(-1, 1).to(pl_module.model.unit_emb.weight.device)
        )
        pl_module.model.session_emb.weight.grad.mul_(
            self.session_weight.view(-1, 1).to(
                pl_module.model.session_emb.weight.device
            )
        )
