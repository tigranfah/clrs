import pickle

# ==== paths ====
# checkpoint_path = "checkpoints/dfs+bfs+dijkstra+floyd_warshall-shared=False-encdec_rank=0-steps=10000.pkl"
checkpoint_path = "checkpoints/dfs+bfs+dijkstra+floyd_warshall-shared=True-encdec_rank=2-steps=10000.pkl"

# ==== load ====
with open(checkpoint_path, "rb") as f:
    ckpt = pickle.load(f)

print(ckpt['params'].keys(), len(ckpt['params'].keys()))