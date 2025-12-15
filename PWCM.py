#!/usr/bin/env python3


from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple, Optional

import numpy as np
import pandas as pd
import networkx as nx


EdgeProb = Dict[Tuple[int, int], float]


# ------------------------
# Paths / loading
# ------------------------

def resolve_paths(data_dir: Optional[str], edges_file: str, features_file: str):
    base_dir = Path(__file__).resolve().parent
    default_data_dir = (base_dir / "data").resolve()
    data_path = Path(data_dir).resolve() if data_dir else default_data_dir
    return base_dir, data_path, (data_path / edges_file), (data_path / features_file)


def load_graph_edges_csv(
    edges_path: Path,
    src_col: str = "numeric_id_1",
    dst_col: str = "numeric_id_2",
    undirected: bool = True,
) -> nx.Graph:
    if not edges_path.exists():
        raise FileNotFoundError(f"Edges file not found: {edges_path}")

    df = pd.read_csv(edges_path)

    # 1) If default columns aren't present, try common alternatives
    if src_col not in df.columns or dst_col not in df.columns:
        candidates = [
            ("id_1", "id_2"),
            ("source", "target"),
            ("src", "dst"),
            ("from", "to"),
            ("u", "v"),
        ]
        found = None
        for a, b in candidates:
            if a in df.columns and b in df.columns:
                found = (a, b)
                break

        # 2) If still not found, fall back to the first two columns if they look plausible
        if found is None and df.shape[1] >= 2:
            a, b = df.columns[0], df.columns[1]
            found = (a, b)

        if found is None:
            raise ValueError(
                f"Edges CSV must contain columns '{src_col}' and '{dst_col}', "
                f"or a known alternative. Found: {list(df.columns)}"
            )

        src_col, dst_col = found
        print(f"[info] Auto-detected edge columns: src_col='{src_col}', dst_col='{dst_col}'")

    edges = list(zip(df[src_col].astype(int).tolist(), df[dst_col].astype(int).tolist()))
    G = nx.Graph() if undirected else nx.DiGraph()
    G.add_edges_from(edges)
    return G



# ------------------------
# PWCM model (paper Eq. (1))
# ------------------------

def compute_pwcm_probs(G: nx.Graph, p_scale: float = 1.0) -> EdgeProb:
    """
    PWCM with scaling:
      p_ij = min(1, p_scale * (deg(j) / sum_{k in N(i)} deg(k)))
    """
    deg = dict(G.degree())
    pwcm_p: EdgeProb = {}

    for i in G.nodes():
        nbrs = list(G.neighbors(i))
        if not nbrs:
            continue

        denom = sum(deg[k] for k in nbrs)
        if denom <= 0:
            continue

        for j in nbrs:
            base_p = deg[j] / denom
            pwcm_p[(int(i), int(j))] = min(1.0, p_scale * base_p)

    return pwcm_p



def pwcm_cascade(G: nx.Graph, seeds: Iterable[int], pwcm_p: EdgeProb, rng: np.random.Generator) -> int:
    """
    One PWCM cascade simulation (IC-style).
    """
    active_queue = list(seeds)
    activated: Set[int] = set(seeds)

    while active_queue:
        i = active_queue.pop(0)
        for j in G.neighbors(i):
            if j in activated:
                continue
            pij = pwcm_p.get((int(i), int(j)), 0.0)
            if pij > 0.0 and rng.random() <= pij:
                activated.add(int(j))
                active_queue.append(int(j))

    return len(activated)


def pwcm_mc(G: nx.Graph, seeds: List[int], pwcm_p: EdgeProb, perms: int, rng_seed: int) -> float:
    """
    Average spread across `perms` permutations/simulations.
    """
    rng = np.random.default_rng(rng_seed)
    spreads = [pwcm_cascade(G, seeds, pwcm_p, rng) for _ in range(perms)]
    return float(np.mean(spreads))


# ------------------------
# Expected-benefit heuristic (paper Eq. (2) + algorithm)
# ------------------------

def expected_benefit(i: int, S_i: Set[int], NS_i: Set[int], pwcm_p: EdgeProb) -> float:
    """
    Paper Eq. (2):
      E(i) = Π_{j in S_i} (1 - p_{j,i}) * (1 + Σ_{k in NS_i} p_{i,k})
    """
    prod_term = 1.0
    for j in S_i:
        pji = pwcm_p.get((int(j), int(i)), 0.0)
        prod_term *= (1.0 - pji)

    sum_term = 0.0
    for k in NS_i:
        sum_term += pwcm_p.get((int(i), int(k)), 0.0)

    return prod_term * (1.0 + sum_term)


def expected_benefit_heuristic_order(G: nx.Graph, pwcm_p: EdgeProb, k_max: int) -> List[int]:
    """
    Returns an ORDERED seed list up to k_max (so we can reuse prefixes for curves).

    Implements the paper's seed selection algorithm based on expected benefit:
    - init E(i)=1+Σ_{k in N_i} p_{ik}, S_i empty, NS_i = N_i
    - pick max E(i), add to seeds
    - for neighbors i of the chosen seed j: move j from NS_i to S_i and recompute E(i) using Eq. (2)
    - repeat until k_max seeds
    """
    nodes = list(G.nodes())
    neighbors: Dict[int, Set[int]] = {int(i): set(map(int, G.neighbors(i))) for i in nodes}

    S_i: Dict[int, Set[int]] = {int(i): set() for i in nodes}
    NS_i: Dict[int, Set[int]] = {int(i): set(neighbors[int(i)]) for i in nodes}

    E: Dict[int, float] = {}
    for i in nodes:
        ii = int(i)
        E[ii] = 1.0 + sum(pwcm_p.get((ii, nb), 0.0) for nb in neighbors[ii])

    seeds_set: Set[int] = set()
    seeds_ordered: List[int] = []

    for _ in range(k_max):
        candidates = [int(v) for v in nodes if int(v) not in seeds_set]
        if not candidates:
            break

        j = max(candidates, key=lambda v: E.get(v, -1.0))
        seeds_set.add(j)
        seeds_ordered.append(j)

        for i in neighbors[j]:
            if i in seeds_set:
                continue
            if j in NS_i[i]:
                S_i[i].add(j)
                NS_i[i].remove(j)
                E[i] = expected_benefit(i, S_i[i], NS_i[i], pwcm_p)

        E[j] = -1.0

    return seeds_ordered


# ------------------------
# Heuristics: degree, pagerank, random
# ------------------------

def degree_heuristic_order(G: nx.Graph, k_max: int) -> List[int]:
    deg = dict(G.degree())
    return [int(n) for n, _ in sorted(deg.items(), key=lambda x: x[1], reverse=True)[:k_max]]


def pagerank_heuristic_order(G: nx.Graph, k_max: int) -> List[int]:
    # NetworkX PageRank
    pr = nx.pagerank(G)
    return [int(n) for n, _ in sorted(pr.items(), key=lambda x: x[1], reverse=True)[:k_max]]


def random_heuristic_order(G: nx.Graph, k_max: int, rng: np.random.Generator) -> List[int]:
    nodes = np.array(list(G.nodes()), dtype=int)
    if k_max >= len(nodes):
        return nodes.tolist()
    return rng.choice(nodes, size=k_max, replace=False).tolist()


# ------------------------
# Plotting outputs
# ------------------------

def plot_spread_curves(xs: List[int], series: Dict[str, List[float]], out_path: Path, title: str):
    import matplotlib.pyplot as plt

    plt.figure()
    for name, ys in series.items():
        plt.plot(xs, ys, label=name)

    plt.title(title)
    plt.xlabel("Seed set size")
    plt.ylabel("Average spread value")
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def save_graph_image(
    G: nx.Graph,
    out_path: Path,
    mode: str = "top_degree",
    sample_n: int = 300,
    seed: int = 42,
):
    """
    Save a DRAWABLE graph image by sampling a subgraph.
    Rendering the full Twitch graph is not practical.

    mode:
      - top_degree: take top-N nodes by degree and induced subgraph
      - random: random N nodes induced subgraph
    """
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)

    if sample_n <= 0:
        raise ValueError("sample_n must be > 0")

    if sample_n >= G.number_of_nodes():
        H = G
    else:
        if mode == "top_degree":
            deg = dict(G.degree())
            nodes = [n for n, _ in sorted(deg.items(), key=lambda x: x[1], reverse=True)[:sample_n]]
        elif mode == "random":
            nodes = rng.choice(np.array(list(G.nodes()), dtype=int), size=sample_n, replace=False).tolist()
        else:
            raise ValueError(f"Unknown graph image mode: {mode}")

        H = G.subgraph(nodes).copy()

    # Layout can be slow; spring_layout is ok for ~300 nodes
    pos = nx.spring_layout(H, seed=seed)

    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(H, pos, node_size=20)
    nx.draw_networkx_edges(H, pos, width=0.5, alpha=0.6)
    plt.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


# ------------------------
# Experiment runner
# ------------------------

def run_experiment(
    G: nx.Graph,
    pwcm_p: EdgeProb,
    seeds: int,
    perms: int,
    rng_seed: int,
    compare_methods: List[str],
) -> Dict[str, List[float]]:
    """
    Compute spread curves from k=1..seeds for the requested heuristics.
    Uses ordered seed lists so we can reuse prefixes efficiently.
    """
    xs = list(range(1, seeds + 1))
    series: Dict[str, List[float]] = {m: [] for m in compare_methods}

    # Precompute ordered seed lists up to seeds
    rng = np.random.default_rng(rng_seed)

    seed_orders: Dict[str, List[int]] = {}
    for m in compare_methods:
        if m == "degree":
            seed_orders[m] = degree_heuristic_order(G, seeds)
        elif m == "pagerank":
            seed_orders[m] = pagerank_heuristic_order(G, seeds)
        elif m == "random":
            seed_orders[m] = random_heuristic_order(G, seeds, rng)
        elif m == "expected_benefit":
            seed_orders[m] = expected_benefit_heuristic_order(G, pwcm_p, seeds)
        else:
            raise ValueError(f"Unknown method: {m}")

    # Evaluate curves
    for k in xs:
        for m in compare_methods:
            S = seed_orders[m][:k]
            spread = pwcm_mc(G, S, pwcm_p, perms=perms, rng_seed=rng_seed)
            series[m].append(spread)

        if k == 1 or k % 10 == 0 or k == seeds:
            print(f"[info] curve progress: k={k}/{seeds}")

    return series


# ------------------------
# CLI
# ------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PWCM experiments on Twitch (Degree vs PageRank vs Random vs Expected-Benefit).")

    p.add_argument("--data-dir", default=None, help="Folder containing the CSVs (default: <script_dir>/data)")
    p.add_argument("--edges-file", default="large_twitch_edges.csv", help="Edges filename inside data-dir")
    p.add_argument("--features-file", default="large_twitch_features.csv", help="Features filename inside data-dir")
    p.add_argument("--load-features", action="store_true", help="Optionally load features CSV (not required).")

    # Your requested command-line inputs
    p.add_argument("--seeds", type=int, required=True, help="Seed set size (k). Also used as curve max if --curve.")
    p.add_argument("--perms", type=int, required=True, help="Number of permutations/simulations (MC runs).")
    p.add_argument("--rng-seed", type=int, default=42, help="RNG seed for reproducibility.")
    p.add_argument(
    "--p-scale",
    type=float,
    default=1.0,
    help="Scaling factor for PWCM probabilities (p_ij = min(1, p_scale * base_p_ij)).")


    # Compare heuristics (paper compares degree, pagerank, random, expected benefit) :contentReference[oaicite:2]{index=2}
    p.add_argument(
        "--methods",
        default="degree,pagerank,random,expected_benefit",
        help="Comma-separated methods to compare: degree,pagerank,random,expected_benefit",
    )

    # Outputs
    p.add_argument("--curve", action="store_true", help="Compute spread curve from 1..--seeds and plot it.")
    p.add_argument("--plot-out", default=None, help="Where to save spread comparison PNG.")
    p.add_argument("--graph-out", default=None, help="Where to save a graph image PNG (sampled subgraph).")

    p.add_argument("--graph-sample-n", type=int, default=300, help="Nodes to include in graph image sample.")
    p.add_argument("--graph-sample-mode", choices=["top_degree", "random"], default="top_degree",
                   help="How to sample the subgraph for drawing.")

    return p.parse_args()


def main() -> int:
    args = parse_args()

    base_dir, data_dir, edges_path, features_path = resolve_paths(args.data_dir, args.edges_file, args.features_file)

    print(f"[info] base_dir: {base_dir}")
    print(f"[info] data_dir: {data_dir}")
    print(f"[info] edges:    {edges_path}")

    # Load graph
    G = load_graph_edges_csv(edges_path=edges_path, undirected=True)
    print(f"[info] Loaded graph: |V|={G.number_of_nodes():,} |E|={G.number_of_edges():,}")

    if args.load_features:
        if not features_path.exists():
            raise FileNotFoundError(f"Features file not found: {features_path}")
        fdf = pd.read_csv(features_path)
        print(f"[info] Loaded features: rows={fdf.shape[0]:,} cols={fdf.shape[1]:,}")

    # PWCM
    print(f"[info] PWCM probability scale p_scale = {args.p_scale}")
    pwcm_p = compute_pwcm_probs(G, p_scale=args.p_scale)

    # Graph image output
    if args.graph_out:
        graph_out = Path(args.graph_out)
    else:
        graph_out = base_dir / "results" / f"graph_sample_{args.graph_sample_mode}_{args.graph_sample_n}.png"

    print(f"[info] Saving graph image (sampled) -> {graph_out}")
    save_graph_image(
        G,
        out_path=graph_out,
        mode=args.graph_sample_mode,
        sample_n=args.graph_sample_n,
        seed=args.rng_seed,
    )

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    print(f"[info] Comparing methods: {methods}")

    # If curve mode: compute curve and plot like the paper's Fig. 3 :contentReference[oaicite:3]{index=3}
    if args.curve:
        series = run_experiment(
            G=G,
            pwcm_p=pwcm_p,
            seeds=args.seeds,
            perms=args.perms,
            rng_seed=args.rng_seed,
            compare_methods=methods,
        )

        xs = list(range(1, args.seeds + 1))
        if args.plot_out:
            plot_out = Path(args.plot_out)
        else:
            plot_out = base_dir / "results" / f"spread_pwcm_seeds{args.seeds}_perms{args.perms}.png"

        print(f"[info] Saving spread comparison plot -> {plot_out}")
        plot_spread_curves(xs, series, plot_out, title="Spread Values By Seed Size (PWCM)")

        # print final values
        for m in methods:
            print(f"[result] {m}: spread@k={args.seeds} = {series[m][-1]:.3f}")

        return 0

    # Otherwise: single-k evaluation only (no curve)
    print("[info] Single-k run (no curve). Computing spreads at k = --seeds")
    rng = np.random.default_rng(args.rng_seed)

    # Seed orders, take prefix k
    seed_orders: Dict[str, List[int]] = {}
    for m in methods:
        if m == "degree":
            seed_orders[m] = degree_heuristic_order(G, args.seeds)
        elif m == "pagerank":
            seed_orders[m] = pagerank_heuristic_order(G, args.seeds)
        elif m == "random":
            seed_orders[m] = random_heuristic_order(G, args.seeds, rng)
        elif m == "expected_benefit":
            seed_orders[m] = expected_benefit_heuristic_order(G, pwcm_p, args.seeds)
        else:
            raise ValueError(f"Unknown method: {m}")

    for m in methods:
        S = seed_orders[m][:args.seeds]
        spread = pwcm_mc(G, S, pwcm_p, perms=args.perms, rng_seed=args.rng_seed)
        print(f"[result] method={m} k={args.seeds} perms={args.perms} expected_spread={spread:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
