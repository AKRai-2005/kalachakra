"""One command to open the intervention console.

    python demo.py

Why this one needs no server
---------------------------
EKAGRA's console reads a live evidence log through a read-only API, because
there is a running sensor to read from. This one is a replay of a finished
model run: the trajectories are computed once by
`experiments/exp13_demo_trajectories.py` and written as a plain script that
assigns a global. So the console is a single file that opens straight from the
filesystem, with no server, no port and nothing to fail.

That is not laziness - it is the honest shape of the thing. Nothing here is
live, and a fake server would imply it was.

If the trajectories are missing this offers to compute them, which takes about
five minutes and is why it is not done on every launch.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONSOLE = ROOT.parent / "frontend" / "index.html"
DATA = ROOT.parent / "frontend" / "data.js"
PRECOMPUTE = ROOT / "experiments" / "exp13_demo_trajectories.py"

BANNER = r"""
   _  __   _   _      _   ___ _  _   _   _  _____   _
  | |/ /  /_\ | |    /_\ / __| || | /_\ | |/ / _ \ /_\    intervention console
  | ' <  / _ \| |__ / _ \ (__| __ |/ _ \|   <   // _ \    SIH26153 - NTRO
  |_|\_\/_/ \_\____/_/ \_\___|_||_/_/ \_\_|\_\_|\_\_/ \_\  world model, offline
"""

TALK_TRACK = """
  What to show, in order
  ----------------------
  1. The header       four numbers, and the third is the one that matters. We
                      say "isolating helps" on 94% of pairs; an action-aware
                      LSTM given the same action says so on 48%, a coin flip.
                      A trivial "always isolate" predictor scores 100%, and
                      that baseline is on screen because it beats us. The
                      number is evidence the LSTM never learned the
                      intervention - not that we can rank hosts.
  2. Pick a case      each is a matched pair: the same episode played twice
                      from a bit-identical prefix, differing only in whether
                      the host was isolated.
  3. The two curves   this is the point. A detector scores the trajectory it
                      observed. Only a model with an action input can be asked
                      what the OTHER branch would have looked like.
  4. Ground truth     what actually happened in both branches - which exists
                      only because we generated the data, and we say so.
  5. What it cannot do
                      the panel is always on screen. It predicts whether
                      isolating helps, not which host to isolate first, and it
                      loses to an LSTM on lead time. Both are stated before a
                      judge has to ask.

  The question this answers is not "is this host compromised" - it is
  "what happens if we act, and what happens if we do not".
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="Open the KALACHAKRA console.")
    ap.add_argument("--regenerate", action="store_true",
                    help="recompute the trajectories even if they exist (~5 min)")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:      # pragma: no cover
        pass

    print(BANNER)

    if not DATA.exists() or args.regenerate:
        why = "regenerating on request" if args.regenerate else "no trajectories yet"
        print(f"  [..]   {why} - training and rolling out (about 5 minutes)")
        proc = subprocess.run([sys.executable, "-u", str(PRECOMPUTE)],
                              cwd=str(ROOT), capture_output=True, text=True)
        if proc.returncode != 0:
            print("  [!!]   precompute failed. Last lines:\n")
            print("\n".join(proc.stdout.strip().splitlines()[-12:]))
            print(proc.stderr.strip()[-600:])
            return 1
        for line in proc.stdout.splitlines():
            if "isolating helps" in line or "cases written" in line or "trivial always-isolate" in line:
                print(f"  [ok]   {line.strip()}")

    if not DATA.exists():
        print("  [!!]   frontend/data.js still missing. Nothing to show.")
        return 1

    size = DATA.stat().st_size / 1024
    print(f"  [ok]   trajectories present ({size:.0f} KB)")
    print(f"  [ok]   console at {CONSOLE}")
    print(TALK_TRACK)

    if not args.no_browser:
        try:
            webbrowser.open(CONSOLE.as_uri())
        except Exception:
            print("  [--]   could not open a browser; open the path above")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
