import tools.validation as valid

import gc
from types import SimpleNamespace

batch_sizes = [16, 32, 64, 128, 256, 512]

for batch_size in batch_sizes:
    config = SimpleNamespace(**{})
    config.seed = 123
    config.num_epochs = 50
    config.n_folds = 5
    config.lr = 1e-4

    config.batch_size = batch_size
    config.name = f"batch_size_{batch_size}"

    valid.cross_validation(config)

    # Clear config
    del config

    # Start garbage collection
    gc.collect()