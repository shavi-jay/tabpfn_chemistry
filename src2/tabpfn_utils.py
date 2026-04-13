from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion


def create_tabpfn(random_state=0):
    return TabPFNRegressor.create_default_for_version(
        ModelVersion.V2,
        n_estimators=8,
        device='cuda',
        random_state=random_state,
        ignore_pretraining_limits=True,
    )