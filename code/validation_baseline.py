import tools.validation as valid

from types import SimpleNamespace

config = SimpleNamespace(**{})
config.seed = 123
config.num_epochs = 50
config.n_folds = 5
config.lr = 1e-4
config.batch_size = 512
config.name = "baseline_220313"

valid.cross_validation(config)