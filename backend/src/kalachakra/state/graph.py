"""Network state as a host-interaction graph.

Why a graph and not a flat vector
---------------------------------
The phenomena we forecast are relational. Lateral movement *is* an edge
appearing between two internal hosts that have never communicated before.
Flattening the window into a vector of aggregate counters destroys exactly the
structure that distinguishes "busy Tuesday" from "attacker pivoting".

Node set
--------
Fixed: the internal hosts we defend. External addresses are not nodes — they
are unbounded, mostly singletons, and we do not defend them. Their influence
enters through per-node external-facing features and the global vector. This
mirrors how an operator actually reasons, and it gives a fixed-size graph,
which makes batching straightforward and the model far cheaper to train.

Edges
-----
Internal host -> internal host, one per observed direction in the window,
carrying volume and a novelty flag. Novelty matters: a first-ever edge between
two internal hosts is the single most informative bit for lateral movement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..data.episodes import STAGE_IDX, Episode, WindowSnapshot

NODE_FEATURES = [
    "out_pkts_ext", "out_bytes_ext", "in_pkts_ext", "in_bytes_ext",
    "out_pkts_int", "out_bytes_int", "in_pkts_int", "in_bytes_int",
    "n_int_peers_out", "n_int_peers_in", "n_dports_out",
    "syn", "rst", "novel_out", "novel_in",
]
GLOBAL_FEATURES = [
    "n_flows", "n_ext_peers", "ext_bytes_out", "ext_bytes_in",
    "int_flow_frac", "novel_edge_frac", "mean_fanout", "max_fanout",
]

N_NODE_F = len(NODE_FEATURES)
N_GLOBAL_F = len(GLOBAL_FEATURES)


def _log1p(x: float) -> float:
    return math.log1p(max(0.0, x))


@dataclass
class GraphSnapshot:
    """One window, as tensors.

    `x`        (N, N_NODE_F)  node features
    `adj`      (N, N)         weighted internal adjacency, log-scaled bytes
    `novel`    (N, N)         1.0 where the edge is first-seen this episode
    `g`        (N_GLOBAL_F,)  window-level features
    `stage`    int            ATT&CK tactic index (label)
    """

    x: np.ndarray
    adj: np.ndarray
    novel: np.ndarray
    g: np.ndarray
    stage: int
    intervened: bool = False


class GraphBuilder:
    """Turns episode windows into graph snapshots with a fixed node set."""

    def __init__(self, hosts: Sequence[str]) -> None:
        self.hosts = list(hosts)
        self.index: Dict[str, int] = {h: i for i, h in enumerate(self.hosts)}
        self.n = len(self.hosts)

    def _is_internal(self, ip: str) -> bool:
        return ip in self.index

    def build_window(self, w: WindowSnapshot) -> GraphSnapshot:
        n = self.n
        x = np.zeros((n, N_NODE_F), dtype=np.float32)
        adj = np.zeros((n, n), dtype=np.float32)
        novel = np.zeros((n, n), dtype=np.float32)

        peers_out: List[set] = [set() for _ in range(n)]
        peers_in: List[set] = [set() for _ in range(n)]
        dports_out: List[set] = [set() for _ in range(n)]
        ext_peers: set = set()
        n_int_flows = 0
        n_novel = 0

        col = {name: i for i, name in enumerate(NODE_FEATURES)}

        for f in w.flows:
            s_int, d_int = self._is_internal(f.src), self._is_internal(f.dst)
            if f.novel:
                n_novel += 1

            if s_int and d_int:
                si, di = self.index[f.src], self.index[f.dst]
                n_int_flows += 1
                x[si, col["out_pkts_int"]] += f.packets
                x[si, col["out_bytes_int"]] += f.bytes
                x[di, col["in_pkts_int"]] += f.packets
                x[di, col["in_bytes_int"]] += f.bytes
                x[si, col["syn"]] += f.syn
                x[si, col["rst"]] += f.rst
                peers_out[si].add(di)
                peers_in[di].add(si)
                dports_out[si].add(f.dport)
                adj[si, di] += f.bytes
                if f.novel:
                    novel[si, di] = 1.0
                    x[si, col["novel_out"]] += 1
                    x[di, col["novel_in"]] += 1
            elif s_int:
                si = self.index[f.src]
                ext_peers.add(f.dst)
                x[si, col["out_pkts_ext"]] += f.packets
                x[si, col["out_bytes_ext"]] += f.bytes
                x[si, col["syn"]] += f.syn
                x[si, col["rst"]] += f.rst
                dports_out[si].add(f.dport)
            elif d_int:
                di = self.index[f.dst]
                ext_peers.add(f.src)
                x[di, col["in_pkts_ext"]] += f.packets
                x[di, col["in_bytes_ext"]] += f.bytes
                x[di, col["rst"]] += f.rst

        for i in range(n):
            x[i, col["n_int_peers_out"]] = len(peers_out[i])
            x[i, col["n_int_peers_in"]] = len(peers_in[i])
            x[i, col["n_dports_out"]] = len(dports_out[i])

        # Log-scale the heavy-tailed columns. Byte counts span six orders of
        # magnitude; without this the model spends its capacity on scale.
        for name in ("out_bytes_ext", "in_bytes_ext", "out_bytes_int", "in_bytes_int",
                     "out_pkts_ext", "in_pkts_ext", "out_pkts_int", "in_pkts_int"):
            j = col[name]
            x[:, j] = np.log1p(x[:, j])
        adj = np.log1p(adj)

        fanouts = np.array([len(p) for p in peers_out], dtype=np.float32)
        g = np.array([
            _log1p(len(w.flows)),
            _log1p(len(ext_peers)),
            _log1p(float(x[:, col["out_bytes_ext"]].sum())),
            _log1p(float(x[:, col["in_bytes_ext"]].sum())),
            n_int_flows / max(1, len(w.flows)),
            n_novel / max(1, len(w.flows)),
            float(fanouts.mean()),
            float(fanouts.max()),
        ], dtype=np.float32)

        return GraphSnapshot(x=x, adj=adj, novel=novel, g=g,
                             stage=STAGE_IDX[w.stage], intervened=w.intervened)

    def build_episode(self, ep: Episode) -> List[GraphSnapshot]:
        return [self.build_window(w) for w in ep.windows]


def hosts_from_config(prefix: str, n: int) -> List[str]:
    return [f"{prefix}.{i+10}" for i in range(n)]


def stack_episode(snaps: Sequence[GraphSnapshot]) -> Tuple[np.ndarray, ...]:
    """(T,N,F), (T,N,N), (T,N,N), (T,G), (T,)"""
    return (
        np.stack([s.x for s in snaps]),
        np.stack([s.adj for s in snaps]),
        np.stack([s.novel for s in snaps]),
        np.stack([s.g for s in snaps]),
        np.array([s.stage for s in snaps], dtype=np.int64),
    )


def flat_window_features(snap: GraphSnapshot) -> np.ndarray:
    """Flattened per-window vector for the non-graph baselines.

    Deliberately a *strong* baseline: node-feature summary statistics plus the
    global vector. If a flat representation were enough, the graph model would
    have to justify itself against this, not against a strawman.
    """
    x = snap.x
    return np.concatenate([
        x.mean(0), x.max(0), x.std(0),
        snap.g,
        [snap.adj.sum(), (snap.adj > 0).sum(), snap.novel.sum()],
    ]).astype(np.float32)
