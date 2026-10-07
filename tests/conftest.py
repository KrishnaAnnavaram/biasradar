import pytest

from biasradar.model import TfidfBiasModel
from biasradar.splits import make_splits
from biasradar.synthetic import make_corpus


@pytest.fixture(scope="session")
def corpus():
    return make_splits(make_corpus(n_biased=400, n_neutral=400, n_antistereo=60, seed=1), seed=1)


@pytest.fixture(scope="session")
def parts(corpus):
    return tuple(corpus[corpus["split"] == s].reset_index(drop=True) for s in ("train", "val", "test"))


@pytest.fixture(scope="session")
def model(parts):
    train, val, _ = parts
    return TfidfBiasModel(seed=0).fit(train, val)
