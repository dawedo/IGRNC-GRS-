import torch
import dgl
import torch.nn as nn
from torch.utils.data import DataLoader
from model_GRU3 import HeteroGCN, NeuralCollaborativeFiltering, EXPLAINER, CustomDataset
from dataset_Final_Negative import GDataset
import copy
import time
import os
import numpy as np
import random

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def create_data_splits(dataset, val_ratio=0.2, random_seed=42):
    set_seed(random_seed)
    training_samples = dataset.generate_training_samples(random_seed=random_seed)
    n_val         = int(len(training_samples) * val_ratio)
    val_samples   = training_samples[:n_val]
    train_samples = training_samples[n_val:]
    def _t(samps, idx, dtype):
        return torch.tensor([s[idx] for s in samps], dtype=dtype)
    return (
        (_t(train_samples, 0, torch.long),
         _t(train_samples, 1, torch.long),
         _t(train_samples, 2, torch.float32)),
        (_t(val_samples,   0, torch.long),
         _t(val_samples,   1, torch.long),
         _t(val_samples,   2, torch.float32)),
    )

# ── Devices ───────────────────────────────────────────────────────────────────
# DGL Windows CUDA build does not support comparison ops (LT, GT …) needed
# by GNNExplainer.  GCN + Explainer run on CPU; NCF can use CUDA.
cpu = torch.device('cpu')
gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"GNN / Explainer device : cpu  (DGL CUDA comparison ops unsupported on Windows)")
print(f"NCF device             : {gpu}")

# ── Fixed dimensions ──────────────────────────────────────────────────────────
num_users  = 5275
num_items  = 1513
num_groups = 995
embed_size = 64

# ── Hyperparameter grids ──────────────────────────────────────────────────────
NUM_NEGATIVES_LIST = [2, 4, 6, 8]
LAMBDA_REG_LIST    = [0.0, 1e-4, 1e-3, 1e-2]
dropout_rates      = [0.3, 0.4, 0.5]
learning_rates     = [0.04, 0.05, 0.06, 0.07, 0.08, 0.09]

# Each run uses a fixed, distinct seed for full reproducibility
RUN_SEEDS  = [40, 101, 163]
num_runs   = len(RUN_SEEDS)        # 3

batch_size     = 256
epochs         = 200
EXPLAINER_FREQ = 5
PATIENCE       = 20

# ── Load embeddings once (shared across all configs) ─────────────────────────
print("Loading Node2Vec embeddings...")
try:
    user_tensors = torch.load("./user_tensors_all_64.pt", map_location=cpu,
                              weights_only=False)
    item_tensors = torch.load("./item_tensors_all_64.pt", map_location=cpu,
                              weights_only=False)
    print("Loaded pre-trained Node2Vec embeddings")
    if user_tensors.shape != (num_users, embed_size):
        raise ValueError(f"user shape mismatch {user_tensors.shape}")
    if item_tensors.shape != (num_items, embed_size):
        raise ValueError(f"item shape mismatch {item_tensors.shape}")
except (FileNotFoundError, ValueError) as e:
    print(f"Warning: {e}  -> using random embeddings")
    user_tensors = torch.randn(num_users, embed_size)
    item_tensors = torch.randn(num_items, embed_size)

# master_features always on CPU (explainer runs on CPU)
master_features = {'user': user_tensors.to(cpu), 'item': item_tensors.to(cpu)}

# ── Evaluation helpers ────────────────────────────────────────────────────────
def hr(model, test_g, test_i, neg_i, n=5):
    hit = 0
    model.eval()
    dev = next(model.parameters()).device
    with torch.no_grad():
        for idx in range(len(test_g)):
            pos  = model(test_g[idx:idx+1].to(dev),
                         test_i[idx:idx+1].to(dev))
            negs = model(test_g[idx].expand(len(neg_i[idx])).to(dev),
                         neg_i[idx].to(dev))
            all_p = torch.cat([pos.view(-1), negs.view(-1)])
            rank  = (torch.sort(all_p, descending=True)[1] == 0
                     ).nonzero(as_tuple=True)[0]
            if len(rank) > 0 and rank[0] < n:
                hit += 1
    return hit / len(test_g)

def ndcg(model, test_g, test_i, neg_i, n=5):
    s = 0.0
    model.eval()
    dev = next(model.parameters()).device
    with torch.no_grad():
        for idx in range(len(test_g)):
            pos  = model(test_g[idx:idx+1].to(dev),
                         test_i[idx:idx+1].to(dev))
            negs = model(test_g[idx].expand(len(neg_i[idx])).to(dev),
                         neg_i[idx].to(dev))
            all_p = torch.cat([pos.view(-1), negs.view(-1)])
            rank  = (torch.sort(all_p, descending=True)[1][:n] == 0
                     ).nonzero(as_tuple=True)[0]
            if len(rank) > 0:
                s += 1.0 / np.log2(rank[0].item() + 2.0)
    return s / len(test_g)

def bpr_loss(pos_scores, neg_scores):
    pos_scores = pos_scores.view(-1)
    neg_scores = neg_scores.view(-1)
    n_pos, n_neg = pos_scores.size(0), neg_scores.size(0)
    if n_pos == 0 or n_neg == 0:
        return torch.tensor(0.0, requires_grad=True, device=pos_scores.device)
    repeats = n_neg // n_pos
    if repeats == 0:
        mn = min(n_pos, n_neg)
        return -torch.mean(torch.log(
            torch.sigmoid(pos_scores[:mn] - neg_scores[:mn]) + 1e-15))
    neg_scores = neg_scores[:repeats * n_pos]
    return -torch.mean(torch.log(
        torch.sigmoid(pos_scores.repeat_interleave(repeats) - neg_scores) + 1e-15))


# ── Grid search ───────────────────────────────────────────────────────────────
all_results  = []
results_file = './results_Final_grid_Hard_Random_negative_3_5_GRU3_Reg.txt'
open(results_file, 'a').close()

print(f"\nGrid search")
print(f"  num_negatives  : {NUM_NEGATIVES_LIST}")
print(f"  lambda_reg     : {LAMBDA_REG_LIST}")
print(f"  dropout rates  : {dropout_rates}")
print(f"  learning rates : {learning_rates}")
print(f"  run seeds      : {RUN_SEEDS}")
print(f"  epochs         : {epochs}   patience: {PATIENCE}")

for num_negatives in NUM_NEGATIVES_LIST:

    print(f"\n{'#'*80}")
    print(f"  num_negatives = {num_negatives}")
    print(f"{'#'*80}")

    # Re-load dataset with current num_negatives
    print(f"  Loading dataset (num_negatives={num_negatives})...")
    dataset = GDataset(
        user_path  ="./Data/Mafengwo/userRating",
        group_path ="./Data/Mafengwo/groupRating",
        num_negatives=num_negatives,
        random_seed=42
    )

    group_user_ids = dataset.load_group_membership(
        "./Data/Mafengwo/groupMember.csv")
    if group_user_ids is None:
        print("  Warning: group membership not found, using random groups")
        group_user_ids = torch.full((num_groups, 4), num_users, dtype=torch.long)
        for i in range(num_groups):
            sz      = np.random.randint(2, 5)
            members = np.random.choice(num_users, sz, replace=False)
            for j, m in enumerate(members):
                group_user_ids[i, j] = m

    # Build CPU graph
    users_list, items_list = zip(*dataset.user_trainMatrix.keys())
    edge_src = torch.tensor(users_list, dtype=torch.long)
    edge_dst = torch.tensor(items_list, dtype=torch.long)
    graph_data_cpu = {
        ('user', 'interacts',     'item'): (edge_src, edge_dst),
        ('item', 'rev_interacts', 'user'): (edge_dst, edge_src),
    }
    graph = dgl.heterograph(graph_data_cpu)   # stays on CPU

    interacted_set = set(zip(users_list, items_list))

    # Test data
    print("  Setting up test data...")
    dataset.generate_negative_samples_if_needed(num_negatives=100)
    try:
        test_group_indices = torch.tensor(
            [p[0] for p in dataset.group_testRatings], dtype=torch.long)
        test_item_indices  = torch.tensor(
            [p[1] for p in dataset.group_testRatings], dtype=torch.long)
        negative_item_indices = []
        for neg_list in dataset.group_testNegatives:
            if len(neg_list) >= 100:
                row = neg_list[:100]
            else:
                row = neg_list + np.random.choice(
                    num_items, 100 - len(neg_list), replace=True).tolist()
            negative_item_indices.append(torch.tensor(row, dtype=torch.long))
        negative_item_indices = torch.stack(negative_item_indices)
        print(f"  Test cases: {len(test_group_indices)}")
    except Exception as e:
        print(f"  Error loading test data: {e}  -> dummy data")
        test_group_indices    = torch.randint(0, num_groups, (100,))
        test_item_indices     = torch.randint(0, num_items,  (100,))
        negative_item_indices = torch.randint(0, num_items,  (100, 100))

    # FIX OOM: move group_user_ids to GPU once per num_negatives config
    group_user_ids_gpu = group_user_ids.to(gpu)

    for lambda_reg in LAMBDA_REG_LIST:
      for dropout_rate in dropout_rates:
        for learning_rate in learning_rates:

            print(f"\n{'='*80}")
            print(f"  neg={num_negatives}  λ={lambda_reg}  dropout={dropout_rate}  lr={learning_rate}")
            print(f"{'='*80}")
            test_metrics_runs = []

            # ── Run loop — one run per seed ───────────────────────────────────
            for run_idx, run_seed in enumerate(RUN_SEEDS):
                print(f"\n{'-'*40}  "
                      f"Run {run_idx+1}/{num_runs}  seed={run_seed}  "
                      f"{'-'*40}")

                # FIX OOM: delete previous run models before allocating new ones
                if run_idx > 0:
                    try:
                        del model1, model2
                    except NameError:
                        pass
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()

                # Each run gets its own fixed seed
                set_seed(run_seed)

                train_data, val_data = create_data_splits(
                    dataset, val_ratio=0.2, random_seed=run_seed)

                trn_g, trn_i, trn_r = train_data     # CPU
                val_g, val_i, val_r  = val_data       # CPU, moved to GPU per epoch

                train_loader = DataLoader(
                    CustomDataset(trn_g, trn_i, trn_r),
                    batch_size=batch_size, shuffle=True)

                print(f"  Train: {len(trn_g)}  Val: {len(val_g)}")

                # GCN on CPU
                model1 = HeteroGCN(
                    in_feats=embed_size, hidden_size=32, out_feats=embed_size,
                    etypes=['interacts', 'rev_interacts'],
                    num_users=num_users, num_items=num_items, device=cpu
                ).to(cpu)

                # NCF on GPU
                model2 = NeuralCollaborativeFiltering(
                    num_users, num_items, num_groups,
                    group_user_ids_gpu, embed_size, dropout_rate
                ).to(gpu)
                model2.update_embeddings(
                    master_features['user'].to(gpu),
                    master_features['item'].to(gpu))

                opt_gnn = torch.optim.Adam(model1.parameters(), lr=learning_rate)
                opt_ncf = torch.optim.Adam(model2.parameters(), lr=learning_rate)

                min_val_loss = float('inf')
                best_epoch   = None
                patience_ctr = 0

                # EXPLAINER created once per run — not inside epoch loop
                features_cpu = {k: v.clone() for k, v in master_features.items()}
                explainer    = EXPLAINER(features_cpu, cpu)

                # ── Epoch loop ────────────────────────────────────────────────
                for epoch in range(epochs):
                    model1.train()
                    model2.train()

                    # Stage 1: GCN BCE (CPU)
                    t0 = time.time()
                    opt_gnn.zero_grad()
                    feat_cpu = {k: v.clone() for k, v in master_features.items()}
                    gnn_pos  = model1(graph, feat_cpu, edge_src, edge_dst)

                    neg_items_list = []
                    for u in users_list:
                        neg = random.randint(0, num_items - 1)
                        while (u, neg) in interacted_set:
                            neg = random.randint(0, num_items - 1)
                        neg_items_list.append(neg)
                    neg_dst = torch.tensor(neg_items_list, dtype=torch.long)
                    gnn_neg = model1(graph, feat_cpu, edge_src, neg_dst)

                    all_scores = torch.cat([gnn_pos, gnn_neg])
                    all_labels = torch.cat([torch.ones_like(gnn_pos),
                                            torch.zeros_like(gnn_neg)])
                    gnn_loss = nn.functional.binary_cross_entropy_with_logits(
                        all_scores, all_labels)
                    gnn_loss.backward()
                    opt_gnn.step()
                    t_gnn = time.time() - t0

                    # Stage 2: GNNExplainer on CPU (throttled)
                    t0 = time.time()
                    if epoch % EXPLAINER_FREQ == 0:
                        explainer.explain(model1, graph, 1, graph_data_cpu)
                    t_exp = time.time() - t0

                    # Stage 3: fuse embeddings → NCF (CPU → GPU)
                    t0 = time.time()
                    enh_u = (master_features['user'] +
                             explainer.Z_INFLUENTIAL['user']).to(gpu)
                    enh_i = (master_features['item'] +
                             explainer.Z_INFLUENTIAL['item']).to(gpu)
                    model2.update_embeddings(enh_u, enh_i)
                    del enh_u, enh_i     # FIX OOM
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    t_emb = time.time() - t0

                    # Stage 4: NCF BPR (GPU)
                    t0 = time.time()
                    total_loss    = 0.0
                    valid_batches = 0
                    for bg, bi, br in train_loader:
                        bg, bi, br = bg.to(gpu), bi.to(gpu), br.to(gpu)
                        opt_ncf.zero_grad()
                        pos_mask = br == 1.0
                        neg_mask = br == 0.0
                        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                            continue
                        pos_scores = model2(bg[pos_mask], bi[pos_mask])
                        neg_scores = model2(bg[neg_mask], bi[neg_mask])
                        loss = bpr_loss(pos_scores, neg_scores)

                        # Batch-level L2 on embeddings touched in this batch only
                        # (user embeddings of group members + pos/neg item embeddings)
                        if lambda_reg > 0.0:
                            members_batch = model2.group_user_ids[bg[pos_mask]]
                            valid_members = members_batch[members_batch < num_users]
                            l2 = (model2.user_embedding(valid_members).norm(2).pow(2)
                                  + model2.item_embedding(bi[pos_mask]).norm(2).pow(2)
                                  + model2.item_embedding(bi[neg_mask]).norm(2).pow(2))
                            loss = loss + lambda_reg * l2

                        loss.backward()
                        opt_ncf.step()
                        total_loss    += loss.item()
                        valid_batches += 1
                    avg_loss = total_loss / max(valid_batches, 1)
                    t_ncf = time.time() - t0

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    print(f'  Epoch [{epoch+1:03d}/{epochs}]  '
                          f'GNN={gnn_loss.item():.6f}  NCF_BPR={avg_loss:.6f}  '
                          f'[gnn:{t_gnn:.2f}s exp:{t_exp:.2f}s '
                          f'emb:{t_emb:.2f}s ncf:{t_ncf:.2f}s]')

                    # Validation
                    model2.eval()
                    with torch.no_grad():
                        vg = val_g.to(gpu)
                        vi = val_i.to(gpu)
                        vr = val_r.to(gpu)
                        pm = vr == 1.0
                        nm = vr == 0.0
                        vl = bpr_loss(model2(vg[pm], vi[pm]),
                                      model2(vg[nm], vi[nm]))
                        val_loss_val = vl.item()
                        del vg, vi, vr
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    print(f'  val_loss={val_loss_val:.6f}  '
                          f'best={best_epoch+1 if best_epoch is not None else "-"}  '
                          f'patience={patience_ctr}/{PATIENCE}')

                    # Early stopping
                    if val_loss_val < min_val_loss:
                        best_epoch   = epoch
                        min_val_loss = val_loss_val
                        patience_ctr = 0
                        os.makedirs("./Models", exist_ok=True)
                        tag = (f"neg{num_negatives}_lreg{lambda_reg}_dr{dropout_rate}"
                               f"_lr{learning_rate}_seed{run_seed}")
                        sv_u = (master_features['user'] +
                                explainer.Z_INFLUENTIAL['user']).to(gpu)
                        sv_i = (master_features['item'] +
                                explainer.Z_INFLUENTIAL['item']).to(gpu)
                        for path, obj in [
                            (f"./Models/model1_{tag}.pth",     model1.state_dict()),
                            (f"./Models/model2_{tag}.pth",     model2.state_dict()),
                            (f"./Models/embeddings_{tag}.pth", {'user': sv_u.cpu(),
                                                                'item': sv_i.cpu()}),
                        ]:
                            tmp = path + ".tmp"
                            try:
                                torch.save(obj, tmp)
                                if os.path.exists(path):
                                    os.remove(path)
                                os.rename(tmp, path)
                            except Exception as se:
                                print(f"  [WARN] save failed {path}: {se}")
                                if os.path.exists(tmp):
                                    try: os.remove(tmp)
                                    except: pass
                        del sv_u, sv_i
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    else:
                        patience_ctr += 1

                    if patience_ctr >= PATIENCE:
                        print(f"  Early stopping at epoch {epoch+1} "
                              f"(no improvement for {PATIENCE} epochs).")
                        break
                # ── end epoch loop ────────────────────────────────────────────

                print(f"  Run seed={run_seed} done  "
                      f"best_val={min_val_loss:.6f}  "
                      f"best_epoch={best_epoch+1 if best_epoch is not None else '-'}")

                if best_epoch is None:
                    print(f"  seed={run_seed}: no checkpoint. Skipping test.")
                    test_metrics_runs.append(dict(
                        hr5=0.0, hr10=0.0, hr20=0.0,
                        ndcg5=0.0, ndcg10=0.0, ndcg20=0.0))
                    continue

                # ── Test ─────────────────────────────────────────────────────
                tag = (f"neg{num_negatives}_lreg{lambda_reg}_dr{dropout_rate}"
                       f"_lr{learning_rate}_seed{run_seed}")
                m2t = NeuralCollaborativeFiltering(
                    num_users, num_items, num_groups,
                    group_user_ids_gpu, embed_size, dropout_rate=0.0
                ).to(gpu)
                m2t.load_state_dict(
                    torch.load(f"./Models/model2_{tag}.pth",
                               map_location=gpu, weights_only=False))
                saved = torch.load(f"./Models/embeddings_{tag}.pth",
                                   map_location=gpu, weights_only=False)
                m2t.update_embeddings(saved['user'], saved['item'])
                del saved
                m2t.eval()

                tg = test_group_indices.to(gpu)
                ti = test_item_indices.to(gpu)
                ni = negative_item_indices.to(gpu)

                hr5  = hr(m2t, tg, ti, ni, 5)
                hr10 = hr(m2t, tg, ti, ni, 10)
                hr20 = hr(m2t, tg, ti, ni, 20)
                nd5  = ndcg(m2t, tg, ti, ni, 5)
                nd10 = ndcg(m2t, tg, ti, ni, 10)
                nd20 = ndcg(m2t, tg, ti, ni, 20)

                tm = dict(hr5=hr5, hr10=hr10, hr20=hr20,
                          ndcg5=nd5, ndcg10=nd10, ndcg20=nd20)
                test_metrics_runs.append(tm)
                print(f"  seed={run_seed} -> "
                      f"HR@5={hr5:.4f} HR@10={hr10:.4f} HR@20={hr20:.4f}")
                print(f"  seed={run_seed} -> "
                      f"NDCG@5={nd5:.4f} NDCG@10={nd10:.4f} NDCG@20={nd20:.4f}")

                # FIX OOM: free test allocations immediately
                del m2t, tg, ti, ni
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # ── end run loop ──────────────────────────────────────────────────

            # FIX OOM: free models after all runs for this combo
            try:
                del model1, model2
            except NameError:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            avg = {k: sum(r[k] for r in test_metrics_runs) / num_runs
                   for k in ('hr5','hr10','hr20','ndcg5','ndcg10','ndcg20')}
            all_results.append(dict(
                num_negatives=num_negatives,
                lambda_reg=lambda_reg,
                dropout_rate=dropout_rate,
                learning_rate=learning_rate,
                **avg,
                test_metrics_runs=test_metrics_runs))

            with open(results_file, 'a') as f:
                f.write(f"NumNeg={num_negatives}  LambdaReg={lambda_reg}  Dropout={dropout_rate}  "
                        f"LR={learning_rate}  Seeds={RUN_SEEDS}\n")
                for k, v in avg.items():
                    f.write(f"{k.upper()}: {v:.4f}\n")
                f.write("Individual Runs:\n")
                for i, (r, s) in enumerate(
                        zip(test_metrics_runs, RUN_SEEDS)):
                    f.write(f"  Run {i+1} (seed={s}): " +
                            ", ".join(f"{k}={r[k]:.4f}" for k in r) + "\n")
                f.write(f"\n{'-'*80}\n\n")

            print(f"\n  Completed neg={num_negatives} λ={lambda_reg} dr={dropout_rate} "
                  f"lr={learning_rate} -> "
                  f"HR@10={avg['hr10']:.4f}  NDCG@10={avg['ndcg10']:.4f}")

# ── Final summary ─────────────────────────────────────────────────────────────
print(f"\n{'='*80}\nGRID SEARCH COMPLETED!\n{'='*80}")
best = max(all_results, key=lambda x: x['hr10'])
print(f"\nBest config (HR@10):")
print(f"  num_negatives={best['num_negatives']}  lambda_reg={best['lambda_reg']}  "
      f"dropout={best['dropout_rate']}  lr={best['learning_rate']}")
for k in ('hr5','hr10','hr20','ndcg5','ndcg10','ndcg20'):
    print(f"  {k.upper()}: {best[k]:.4f}")

print(f"\nBest HR@10 per num_negatives:")
for nn in NUM_NEGATIVES_LIST:
    subset = [r for r in all_results if r['num_negatives'] == nn]
    if subset:
        b = max(subset, key=lambda x: x['hr10'])
        print(f"  neg={nn}: HR@10={b['hr10']:.4f}  "
              f"dropout={b['dropout_rate']}  lr={b['learning_rate']}")

print(f"\nAll results saved to '{results_file}'")
print("Mafengwo evaluation completed successfully!")