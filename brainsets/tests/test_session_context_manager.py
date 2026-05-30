import pytest
import tempfile

from brainsets_utils import DatasetBuilder
from brainsets_utils.taxonomy import Species


@pytest.fixture
def test_folderpath(request):
    tmpdir = tempfile.TemporaryDirectory()
    folderpath = tmpdir.name

    def finalizer():
        tmpdir.cleanup()

    request.addfinalizer(finalizer)
    return folderpath


def test_session_context_manager(test_folderpath):
    builder = DatasetBuilder(
        raw_folder_path=test_folderpath,
        processed_folder_path=test_folderpath,
        experiment_name="my_experiment",
        origin_version="unknown",
        derived_version="0.0.1",
        source="https://example.com",
        description="This is a description of the data.",
    )

    # not using builder.new_session() correctly will raise an error
    with pytest.raises(ValueError):
        session = builder.new_session()
        session.register_subject(id="alice", species=Species.HOMO_SAPIENS)
