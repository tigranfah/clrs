import pickle

# ==== paths ====
dijkstra_ckpt_path = "checkpoints/dijkstra_10k.pkl"
output_bfs_ckpt_path = "checkpoints/bfs_from_dijkstra_10k.pkl"

# ==== load ====
with open(dijkstra_ckpt_path, "rb") as f:
    ckpt = pickle.load(f)

# Some CLRS checkpoints store params under "params"
if "params" in ckpt:
    weights = ckpt["params"]
else:
    weights = ckpt

new_weights = {}

def keep(key):
    """Keys that should transfer directly."""
    if key.startswith("net/mpnn_aggr_clrs_processor"):
        return True

    keep_list = [
        "algo_0_A_enc_linear",
        "algo_0_adj_enc_linear",
        "algo_0_pos_enc_linear",
        "algo_0_s_enc_linear",
        "algo_0_pi_dec_linear",
        "algo_0_pi_dec_linear_1",
        "algo_0_pi_dec_linear_2",
        "algo_0_pi_dec_linear_3",
        "algo_0_pi_h_dec_linear",
        "algo_0_pi_h_dec_linear_1",
        "algo_0_pi_h_dec_linear_2",
        "algo_0_pi_h_dec_linear_3",
        "algo_0_pi_h_enc_linear",
    ]

    return any(k in key for k in keep_list)


for k, v in weights.items():

    # ===== processor + shared weights =====
    if keep(k):
        new_weights[k] = v
        continue

    # ===== map in_queue -> reach_h =====
    if "algo_0_in_queue_enc_linear" in k:
        new_key = k.replace("in_queue_enc", "reach_h_enc")
        new_weights[new_key] = v
        continue

    if "algo_0_in_queue_dec_linear" in k:
        new_key = k.replace("in_queue_dec", "reach_h_dec")
        new_weights[new_key] = v
        continue

    # everything else is discarded


# ==== wrap back if needed ====
if "params" in ckpt:
    ckpt["params"] = new_weights
    out = ckpt
else:
    out = new_weights

# ==== save ====
with open(output_bfs_ckpt_path, "wb") as f:
    pickle.dump(out, f)

print("Done. BFS-compatible checkpoint saved to:", output_bfs_ckpt_path)
print("Total keys:", len(new_weights))
