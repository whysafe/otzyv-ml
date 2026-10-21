import pytest

from otzyv.model import train_full


@pytest.fixture(scope="session")
def classifier():
    return train_full("logreg")


@pytest.fixture(scope="session")
def model_file(classifier, tmp_path_factory):
    return classifier.save(tmp_path_factory.mktemp("model") / "model.joblib")
