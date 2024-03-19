import tools.training as train

from types import SimpleNamespace

config = SimpleNamespace(**{})
config.seed = 123
config.num_epochs = 30
config.lr = 1e-4
config.batch_size = 128
config.name = "baseline_220318"

train.training_run(config)