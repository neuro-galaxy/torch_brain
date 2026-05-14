import os

import numpy as np
import pytest
from omegaconf import OmegaConf

os.environ.setdefault("ROOT_DIR_BRAINTREEBANK", "/tmp")

from neuroprobe_eval.models.base_model import BaseModel
from neuroprobe_eval.models.cnn_model import CNNModel
from neuroprobe_eval.models.model_utils import to_int_list


class _DummyModel(BaseModel):
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        _ = (X_train, y_train, X_val, y_val)
        self.classes_ = np.array([0, 1], dtype=np.int64)
        return self

    def predict_proba(self, X):
        return np.tile(np.array([[0.25, 0.75]], dtype=np.float64), (len(X), 1))


def test_base_model_predict_requires_fit():
    model = _DummyModel()
    with pytest.raises(RuntimeError, match="Call fit"):
        model.predict(np.zeros((2, 3), dtype=np.float32))


def test_to_int_list_rejects_silent_float_coercion():
    with pytest.raises(TypeError, match="int"):
        to_int_list(3.5, [256])
    with pytest.raises(TypeError, match="int"):
        to_int_list([64, 32.1], [256])


def test_to_int_list_keeps_int_values_and_default_fallback():
    assert to_int_list([64, 128], [256]) == [64, 128]
    assert to_int_list(np.int64(32), [256]) == [32]
    assert to_int_list([], [256], allow_empty=False) == [256]
    assert to_int_list(None, [256]) == [256]


def test_cnn_model_rejects_spatial_inputs_too_small_for_pooling():
    cfg = OmegaConf.create({"name": "cnn", "device": "cpu", "hidden_dims": [64]})
    model = CNNModel(cfg)
    with pytest.raises(ValueError, match="too small"):
        model.build_model((2, 7, 7), n_classes=2, device="cpu")
