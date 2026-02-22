import torch

# Hyperparameters
batch_size = 16  # How many batches per training step
context_length = 1024  # Length of the token chunk each batch
d_model = 1024  # The size of our model token embeddings
num_blocks = 24  # Number of transformer blocks
num_heads = 16  # Number of heads in Multi-head attention
learning_rate = 3e-4  # 0.001
dropout = 0.1  # Dropout rate
max_iters = 10000  # Total of training iterations <- Change this to smaller number for testing
eval_interval = 50  # How often to evaluate
eval_iters = 20  # Number of iterations to average for evaluation
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if it's available.
TORCH_SEED = 1337
torch.manual_seed(TORCH_SEED)