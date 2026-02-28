import pickle
from pprint import pprint

# ==== paths ====
checkpoint_path = "checkpoints/dfs+bfs-shared=False-encdec_rank=0-steps=10000.pkl"
# checkpoint_path = "checkpoints/dfs+bfs-shared=True-encdec_rank=2-steps=10000.pkl"

# ==== load ====
with open(checkpoint_path, "rb") as f:
    ckpt = pickle.load(f)

pprint(ckpt['params'].keys())