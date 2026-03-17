import torch

# Device setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model parameters
batch_size = 64
context_win = 256
max_iters = 5000
eval_interval = 1000
eval_iters = 200
learning_rate = 3e-4

max_epochs = 10
n_embd = 128
n_head = 8
n_layer = 8
dropout = 0.2
vocab_size = 100256 


train_file = r"D:\dataset\webcrawled_train.txt"
val_file = r"D:\dataset\webcrawled_val.txt"
save_dir =r"D:\min_gpt_model"
