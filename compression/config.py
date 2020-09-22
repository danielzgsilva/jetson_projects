from torch import cuda

# This file contains the configuration parameters which will be used throughout your experiments
use_cuda = cuda.is_available()

n_classes = 100

n_epochs = 1000
batch_size = 100
workers = 2

learning_rate = 1e-5
weight_decay = 0.0#1e-7

model_id = 2
save_dir = './SavedModels/Run%d/' % model_id
