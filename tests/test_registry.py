import pytest
from torch_brain.registry import (
    DataType,
    ModalitySpec,
    register_modality,
    MODALITIY_REGISTRY,
)


def test_data_type_enum():
    """Test DataType enum values and members."""
    assert DataType.CONTINUOUS.value == 0
    assert DataType.BINARY.value == 1
    assert DataType.MULTINOMIAL.value == 2
    assert DataType.MULTILABEL.value == 3

    # Test all expected enum members exist
    expected_members = {"CONTINUOUS", "BINARY", "MULTINOMIAL", "MULTILABEL"}
    assert set(DataType.__members__.keys()) == expected_members


def test_modality_spec_creation():
    """Test ModalitySpec dataclass creation and attributes."""
    spec = ModalitySpec(
        id=0,
        dim=2,
        type=DataType.CONTINUOUS,
        loss_fn="mse",
        timestamp_key="test.timestamps",
        value_key="test.values",
        context_key="test.context",
    )

    assert spec.id == 0
    assert spec.dim == 2
    assert spec.type == DataType.CONTINUOUS
    assert spec.loss_fn == "mse"
    assert spec.timestamp_key == "test.timestamps"
    assert spec.value_key == "test.values"
    assert spec.context_key == "test.context"


def test_modality_spec_optional_context():
    """Test ModalitySpec creation without optional context_key."""
    spec = ModalitySpec(
        id=0,
        dim=2,
        type=DataType.CONTINUOUS,
        loss_fn="mse",
        timestamp_key="test.timestamps",
        value_key="test.values",
    )

    assert spec.context_key is None


@pytest.fixture
def clear_registry():
    """Fixture to clear the registry before and after each test."""
    MODALITIY_REGISTRY.clear()
    yield
    MODALITIY_REGISTRY.clear()


def test_register_modality(clear_registry):
    """Test successful modality registration."""
    modality_id = register_modality(
        "test_modality",
        dim=2,
        type=DataType.CONTINUOUS,
        loss_fn="mse",
        timestamp_key="test.timestamps",
        value_key="test.values",
    )

    assert modality_id == 0
    assert "test_modality" in MODALITIY_REGISTRY
    assert MODALITIY_REGISTRY["test_modality"].id == 0
    assert MODALITIY_REGISTRY["test_modality"].dim == 2


def test_register_duplicate_modality(clear_registry):
    """Test that registering a duplicate modality raises ValueError."""
    register_modality(
        "test_modality",
        dim=2,
        type=DataType.CONTINUOUS,
        loss_fn="mse",
        timestamp_key="test.timestamps",
        value_key="test.values",
    )

    with pytest.raises(
        ValueError, match="Modality test_modality already exists in registry"
    ):
        register_modality(
            "test_modality",
            dim=3,
            type=DataType.BINARY,
            loss_fn="bce",
            timestamp_key="other.timestamps",
            value_key="other.values",
        )


def test_register_multiple_modalities(clear_registry):
    """Test registering multiple modalities with correct ID assignment."""
    id1 = register_modality(
        "modality1",
        dim=2,
        type=DataType.CONTINUOUS,
        loss_fn="mse",
        timestamp_key="test1.timestamps",
        value_key="test1.values",
    )

    id2 = register_modality(
        "modality2",
        dim=3,
        type=DataType.BINARY,
        loss_fn="bce",
        timestamp_key="test2.timestamps",
        value_key="test2.values",
    )

    assert id1 == 0
    assert id2 == 1
    assert len(MODALITIY_REGISTRY) == 2
    assert MODALITIY_REGISTRY["modality1"].id == 0
    assert MODALITIY_REGISTRY["modality2"].id == 1