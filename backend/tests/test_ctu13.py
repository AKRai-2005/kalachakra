"""The CTU-13 adapter, and the three construction bugs it already shipped.

Every test here that names a bug is pinning one that actually happened during
development. The adapter decides what "an attack episode" and "a benign episode"
mean for the only real-data experiment in this project, so a silent mistake in it
would not surface as a crash - it would surface as a believable number.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kalachakra.data.ctu13 import (  # noqa: E402
    COMPROMISED_STAGE_NAME, ScenarioConfig, _flags, choose_host_sets,
    compromise_window, load_scenario, survey,
)

HEADER = ("StartTime,Dur,Proto,SrcAddr,Sport,Dir,DstAddr,Dport,State,sTos,dTos,"
          "TotPkts,TotBytes,SrcBytes,Label")


def row(t, src, dst, label, dport=80, state="CON", pkts=4, byts=400):
    return (f"2011/08/17 {t},0.5,tcp,{src},1234,   ->,{dst},{dport},{state},"
            f"0,0,{pkts},{byts},200,{label}")


def write_capture(path: Path, rows) -> Path:
    path.write_text(HEADER + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def synth(tmp_path, *, onset_min=30, n_infected=3, minutes=90):
    """A capture with background traffic, some busy hosts, and a late onset."""
    rows = []
    for m in range(minutes):
        t = f"{11 + m // 60:02d}:{m % 60:02d}:00.000000"
        # a few very busy hosts, so the "busiest fill" path is exercised
        for b in range(6):
            for _ in range(8):
                rows.append(row(t, f"147.32.80.{b}", "8.8.8.8",
                                "flow=Background-TCP-Established"))
        # a long tail of quiet hosts
        for q in range(40):
            rows.append(row(t, f"147.32.90.{q}", "8.8.4.4", "flow=Background"))
        # infected hosts: normal before onset, botnet after
        for i in range(n_infected):
            lab = ("flow=From-Botnet-V50-TCP-CC" if m >= onset_min + i
                   else "flow=From-Normal-V50-Host")
            rows.append(row(t, f"147.32.84.{100 + i}", "1.2.3.4", lab))
    return write_capture(tmp_path / "9-Test-20110817.binetflow.csv", rows)


# ------------------------------------------------------------------- flags

@pytest.mark.parametrize("state,syn,rst", [
    ("S_", 1, 0), ("SRPA_FSPA", 1, 1), ("FSPA_FSPA", 1, 0),
    ("CON", 0, 0), ("R_", 0, 1), ("", 0, 0),
])
def test_argus_flags_read_only_the_source_half(state, syn, rst):
    """`FSPA_FSPA` is source flags then destination flags. On a one-way tap only
    the source half is observable, so reading the whole string would credit us
    with evidence the sensor cannot have."""
    assert _flags(state) == (syn, rst)


def test_flags_survive_a_missing_state():
    assert _flags(None) == (0, 0)


# ----------------------------------------------------------- host selection

def test_the_two_host_sets_have_comparable_activity(tmp_path):
    """BUG THIS PINS: the first version matched the clean set to the *infected*
    hosts' median. The monitored set is quiet infected hosts plus the busiest
    machines on the network, so that produced attack episodes of 167,000 flows
    against benign episodes of 300 - a 500x gap any model could exploit by
    learning "busy means attack", with no behaviour involved."""
    cfg = ScenarioConfig(n_hosts=12)
    p = synth(tmp_path)
    info = survey(p, cfg)
    mon, clean = choose_host_sets(info, cfg)
    act = info["activity"]
    m_tot = sum(act.get(h, 0) for h in mon)
    c_tot = sum(act.get(h, 0) for h in clean)
    assert c_tot > 0
    ratio = max(m_tot, c_tot) / min(m_tot, c_tot)
    assert ratio < 2.0, f"host sets differ {ratio:.1f}x in activity"


def test_infected_hosts_are_always_monitored(tmp_path):
    """An episode set that sometimes omits the host that gets compromised would
    measure the sampler rather than the model."""
    cfg = ScenarioConfig(n_hosts=12)
    p = synth(tmp_path)
    info = survey(p, cfg)
    mon, clean = choose_host_sets(info, cfg)
    for h in info["botnet_hosts"]:
        assert h in mon
        assert h not in clean


def test_the_clean_set_contains_no_infected_host(tmp_path):
    cfg = ScenarioConfig(n_hosts=12)
    p = synth(tmp_path)
    info = survey(p, cfg)
    _, clean = choose_host_sets(info, cfg)
    assert not (set(clean) & set(info["botnet_hosts"]))


# --------------------------------------------------------------- episodes

def test_one_attack_episode_per_scenario(tmp_path):
    """BUG THIS PINS: a scenario's infected hosts wake within minutes of each
    other, so one episode per infected host would share almost all of its
    pre-compromise context. That is pseudo-replication - it multiplies n
    without adding evidence. CTU-13 is thirteen independent infection events."""
    cfg = ScenarioConfig(n_hosts=12, episode_windows=20, pre_windows=15)
    eps, _, meta = load_scenario(synth(tmp_path, n_infected=5), cfg)
    assert meta["n_attack"] == 1, f"{meta['n_attack']} attack episodes from one onset"


def test_the_attack_episode_has_pre_compromise_context(tmp_path):
    """Lead time is compromise_window minus alarm_window. With no benign prefix
    there is no room to earn any, and the experiment would measure nothing."""
    cfg = ScenarioConfig(n_hosts=12, episode_windows=20, pre_windows=15)
    eps, _, _ = load_scenario(synth(tmp_path, onset_min=30), cfg)
    atk = [e for e in eps if e.is_attack]
    assert atk
    c = compromise_window(atk[0])
    assert c is not None and c > 0, "compromise at window 0 leaves no lead room"
    assert atk[0].windows[c].stage == COMPROMISED_STAGE_NAME
    assert atk[0].windows[c - 1].stage == "BENIGN"


def test_stages_never_revert_after_compromise(tmp_path):
    cfg = ScenarioConfig(n_hosts=12, episode_windows=20, pre_windows=15)
    eps, _, _ = load_scenario(synth(tmp_path), cfg)
    for e in eps:
        seen = False
        for w in e.windows:
            if w.stage == COMPROMISED_STAGE_NAME:
                seen = True
            elif seen:
                pytest.fail(f"{e.id} reverts to {w.stage} after compromise")


def test_benign_episodes_exist_and_are_labelled_benign(tmp_path):
    """BUG THIS PINS: taking benign episodes only from the pre-infection period
    yielded four across all thirteen scenarios. A false-alarm rate over four
    episodes quantises to {0, .25, .5, .75, 1}, and lead time is measured *at a
    matched false-alarm rate* - four is unusable, not merely small."""
    cfg = ScenarioConfig(n_hosts=12, episode_windows=20, pre_windows=15,
                         min_flows=1)
    eps, _, meta = load_scenario(synth(tmp_path, onset_min=30), cfg)
    benign = [e for e in eps if not e.is_attack]
    assert len(benign) >= 2, f"only {len(benign)} benign episodes"
    for e in benign:
        assert compromise_window(e) is None
        assert all(w.stage == "BENIGN" for w in e.windows)


def test_episodes_are_uniform_length(tmp_path):
    """`stack_episode` requires it, and a ragged episode would fail far from
    here with a shape error rather than a diagnosis."""
    cfg = ScenarioConfig(n_hosts=12, episode_windows=20, pre_windows=15,
                         min_flows=1)
    eps, _, _ = load_scenario(synth(tmp_path), cfg)
    assert eps
    assert {len(e.windows) for e in eps} == {20}


def test_a_capture_with_no_botnet_yields_only_benign(tmp_path):
    rows = []
    for m in range(60):
        t = f"11:{m:02d}:00.000000"
        for q in range(12):
            rows.append(row(t, f"147.32.90.{q}", "8.8.4.4", "flow=Background"))
    p = write_capture(tmp_path / "1-Clean-20110810.binetflow.csv", rows)
    cfg = ScenarioConfig(n_hosts=8, episode_windows=20, pre_windows=15,
                         min_flows=1)
    eps, _, meta = load_scenario(p, cfg)
    assert meta["n_attack"] == 0
    assert all(not e.is_attack for e in eps)


def test_unparseable_timestamps_are_refused_not_guessed(tmp_path):
    p = write_capture(tmp_path / "2-Bad-20110811.binetflow.csv",
                      ["not-a-time,0.5,tcp,147.32.84.1,1,   ->,8.8.8.8,80,CON,"
                       "0,0,1,100,50,flow=Background"])
    with pytest.raises(ValueError, match="timestamps"):
        load_scenario(p, ScenarioConfig())
