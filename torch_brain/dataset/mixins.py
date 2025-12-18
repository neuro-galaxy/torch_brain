import numpy as np


class SpikingDatasetMixin:
    def get_unit_ids(self) -> list[str]:
        ans = [self.get_recording(rid).units.id for rid in self.recording_ids]
        return np.sort(np.concatenate(ans)).tolist()
