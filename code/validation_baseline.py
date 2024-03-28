import tools.validation as valid

from types import SimpleNamespace

config = SimpleNamespace(**{})
config.seed = 123
config.num_epochs = 30
config.lr = 1e-4
config.batch_size = 128
config.n_folds = 5
config.name = "baseline_220328"

valid.cross_validation(config)
# valid.summarize(config.name)