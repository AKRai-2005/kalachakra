"""Graph-state invariants.

The graph is the model's entire view of the network, so anything wrong here is
invisible downstream - it just shows up as a model that does not work, with no
indication why. These tests pin the node set, the feature ranges, the novelty
flag that carries the lateral-movement signal, and the absence of label leakage.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kalachakra.data.episodes import (  # noqa: E402
    Flow, EpisodeConfig, EpisodeGenerator, WindowSnapshot,
)
from kalachakra.state.graph import (  # noqa: E402
    GLOBAL_FEATURES, N_GLOBAL_F, N_NODE_F, NODE_FEATURES, GraphBuilder,
    flat_window_features, hosts_from_config, stack_episode,
)

CFG = EpisodeConfig(n_windows=20, n_internal=8)
HOSTS = hosts_from_config(CFG.internal_prefix, CFG.n_internal)


def builder():
    return GraphBuilder(HOSTS)


def win(flows, stage="BENIGN", index=0):
    return WindowSnapshot(index=index, flows=flows, stage=stage)


def flow(src, dst, dport=443, packets=10, bytes_=1000, syn=1, rst=0, novel=False):
    return Flow(src, dst, dport, "tcp", packets, bytes_, syn, rst, novel)


# ------------------------------------------------------------------ node set

def test_node_set_is_fixed_and_internal_only():
    """External addresses are unbounded, mostly singletons, and not ours to
    defend. They enter through per-node external-facing features instead."""
    b = builder()
    assert b.n == CFG.n_internal
    assert set(b.index) == set(HOSTS)
    assert all(b._is_internal(h) for h in HOSTS)
    assert not b._is_internal("8.8.8.8")


def test_external_traffic_lands_on_the_internal_endpoint_not_a_new_node():
    b = builder()
    g = b.build_window(win([
        flow(HOSTS[0], "203.0.113.7", packets=5, bytes_=500),
        flow("203.0.113.9", HOSTS[1], packets=7, bytes_=700),
    ]))
    assert g.x.shape == (b.n, N_NODE_F)
    col = {n: i for i, n in enumerate(NODE_FEATURES)}
    assert g.x[0, col["out_pkts_ext"]] > 0
    assert g.x[1, col["in_pkts_ext"]] > 0
    # nothing internal-to-internal happened, so the adjacency stays empty
    assert g.adj.sum() == 0


def test_internal_traffic_populates_the_adjacency():
    b = builder()
    g = b.build_window(win([flow(HOSTS[2], HOSTS[5], bytes_=4000)]))
    i, j = b.index[HOSTS[2]], b.index[HOSTS[5]]
    assert g.adj[i, j] > 0
    assert g.adj[j, i] == 0, "adjacency is directed"


# ------------------------------------------------------------------ features

def test_shapes_and_finiteness():
    b = builder()
    g = b.build_window(win([flow(HOSTS[0], HOSTS[1]), flow(HOSTS[1], "1.2.3.4")]))
    assert g.x.shape == (b.n, N_NODE_F)
    assert g.adj.shape == (b.n, b.n)
    assert g.novel.shape == (b.n, b.n)
    assert g.g.shape == (N_GLOBAL_F,)
    for arr, name in ((g.x, "x"), (g.adj, "adj"), (g.novel, "novel"), (g.g, "g")):
        assert np.isfinite(arr).all(), f"{name} contains non-finite values"
        assert (arr >= 0).all(), f"{name} should be non-negative"


def test_feature_name_lists_match_the_tensors():
    assert len(NODE_FEATURES) == N_NODE_F
    assert len(GLOBAL_FEATURES) == N_GLOBAL_F
    assert len(set(NODE_FEATURES)) == N_NODE_F, "duplicate node feature name"
    assert len(set(GLOBAL_FEATURES)) == N_GLOBAL_F


def test_heavy_tailed_columns_are_log_scaled():
    """Byte counts span six orders of magnitude. Without log scaling the model
    spends its capacity on scale rather than structure."""
    b = builder()
    small = b.build_window(win([flow(HOSTS[0], HOSTS[1], bytes_=1_000)]))
    large = b.build_window(win([flow(HOSTS[0], HOSTS[1], bytes_=1_000_000)]))
    col = NODE_FEATURES.index("out_bytes_int")
    ratio = large.x[b.index[HOSTS[0]], col] / max(1e-9, small.x[b.index[HOSTS[0]], col])
    assert ratio < 3.0, f"bytes look unscaled (ratio {ratio:.1f} for a 1000x increase)"


def test_novelty_flag_marks_first_seen_internal_edges():
    """A first-ever edge between two internal hosts is the single most
    informative bit for lateral movement."""
    b = builder()
    g = b.build_window(win([
        flow(HOSTS[0], HOSTS[3], novel=True),
        flow(HOSTS[1], HOSTS[4], novel=False),
    ]))
    assert g.novel[b.index[HOSTS[0]], b.index[HOSTS[3]]] == 1.0
    assert g.novel[b.index[HOSTS[1]], b.index[HOSTS[4]]] == 0.0
    col = {n: i for i, n in enumerate(NODE_FEATURES)}
    assert g.x[b.index[HOSTS[0]], col["novel_out"]] == 1
    assert g.x[b.index[HOSTS[3]], col["novel_in"]] == 1


def test_peer_and_port_counts_are_distinct_counts():
    b = builder()
    g = b.build_window(win([
        flow(HOSTS[0], HOSTS[1], dport=80),
        flow(HOSTS[0], HOSTS[1], dport=443),   # same peer, second port
        flow(HOSTS[0], HOSTS[2], dport=80),    # second peer
    ]))
    col = {n: i for i, n in enumerate(NODE_FEATURES)}
    i = b.index[HOSTS[0]]
    assert g.x[i, col["n_int_peers_out"]] == 2
    assert g.x[i, col["n_dports_out"]] == 2


def test_empty_window_is_handled():
    g = builder().build_window(win([]))
    assert np.isfinite(g.x).all() and np.isfinite(g.g).all()
    assert g.adj.sum() == 0


# ------------------------------------------------------------------ episodes

def test_stack_episode_shapes():
    gen = EpisodeGenerator(CFG, seed=5)
    ep = gen.generate(n_attack=1, n_benign=0)[0]
    b = builder()
    x, adj, novel, g, stage = stack_episode(b.build_episode(ep))
    T = CFG.n_windows
    assert x.shape == (T, b.n, N_NODE_F)
    assert adj.shape == (T, b.n, b.n)
    assert novel.shape == (T, b.n, b.n)
    assert g.shape == (T, N_GLOBAL_F)
    assert stage.shape == (T,)
    assert stage.dtype == np.int64


def test_flat_features_are_a_strong_baseline_not_a_strawman():
    """The non-graph baseline gets summary statistics over all nodes plus the
    global vector. If a flat representation were enough, the graph model would
    have to justify itself against this."""
    b = builder()
    g = b.build_window(win([flow(HOSTS[0], HOSTS[1])]))
    f = flat_window_features(g)
    assert f.ndim == 1
    assert f.shape[0] == 3 * N_NODE_F + N_GLOBAL_F + 3
    assert np.isfinite(f).all()


def test_graph_carries_no_label_leakage():
    """Two windows with identical traffic must produce identical tensors even
    when their stage labels differ. If the stage leaked into the features the
    model would be reading the answer."""
    b = builder()
    flows = [flow(HOSTS[0], HOSTS[1]), flow(HOSTS[2], "9.9.9.9")]
    benign = b.build_window(win(list(flows), stage="BENIGN"))
    lateral = b.build_window(win(list(flows), stage="LATERAL_MOVEMENT"))
    assert np.array_equal(benign.x, lateral.x)
    assert np.array_equal(benign.adj, lateral.adj)
    assert np.array_equal(benign.g, lateral.g)
    assert benign.stage != lateral.stage, "the label itself should still differ"


def test_builder_is_deterministic():
    gen = EpisodeGenerator(CFG, seed=8)
    ep = gen.generate(n_attack=1, n_benign=0)[0]
    a = stack_episode(builder().build_episode(ep))
    c = stack_episode(builder().build_episode(ep))
    for m, n in zip(a, c):
        assert np.array_equal(m, n)
