"""Fetch the CTU-13 botnet dataset and report what actually arrived.

Why this script exists rather than a wget line
-----------------------------------------------
The canonical host, `mcfp.felk.cvut.cz`, is unreachable from some networks -
including the one this project was built on. It is the Malware Capture Facility
Project and serves live malware samples, so ISPs block it, and the block is at
the IP level rather than DNS: connecting to 147.32.82.194 directly with a Host
header fails too. Its parent `felk.cvut.cz` answers fine.

So this pulls the Kaggle mirror, which is the same Stratosphere data converted
to parquet with dtypes fixed and duplicates removed, kept **per scenario** - the
separation matters, because a lead-time measurement needs each capture's own
timeline, and a merged file destroys exactly that.

Licence note worth carrying into the references slide: the original CTU-13 is
CC-BY, but this mirror is published under **CC BY-NC-SA 4.0**. Non-commercial
is fine for an academic submission; it is not fine for a product, and the
difference should not be discovered later.

Setup, once
-----------
1. Sign in at kaggle.com, open Settings, and under API choose
   "Create New Token". A `kaggle.json` downloads.
2. Move it to `C:\\Users\\<you>\\.kaggle\\kaggle.json`.
3. Run this script.

    python fetch_ctu13.py
    python fetch_ctu13.py --check      # report on an existing download only
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "data" / "ctu13"
SLUG = "dhoogla/ctu13"

# The thirteen scenarios, by the botnet each ran. Carried here so the script can
# say which are missing rather than just counting files.
BOTNETS = ("Neris", "Neris", "Rbot", "Rbot", "Virut", "Menti", "Sogou",
           "Murlo", "Neris", "Rbot", "Rbot", "NSIS", "Virut")


def have_credentials() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return (Path.home() / ".kaggle" / "kaggle.json").exists()


def download(version: int | None = None) -> int:
    if not have_credentials():
        print("No Kaggle credentials found.\n")
        print("  1. Sign in at https://www.kaggle.com, open Settings")
        print("  2. Under 'API', choose 'Create New Token' - kaggle.json downloads")
        print(f"  3. Move it to {Path.home() / '.kaggle' / 'kaggle.json'}")
        print("  4. Re-run this script")
        return 2

    DEST.mkdir(parents=True, exist_ok=True)
    try:
        import kaggle
    except (ImportError, OSError) as e:
        print(f"kaggle client unavailable: {e}")
        print("  pip install kaggle")
        return 2

    if version is None:
        print(f"Downloading {SLUG} (latest) into {DEST} ...")
        kaggle.api.dataset_download_files(SLUG, path=str(DEST), unzip=True, quiet=False)
        return 0

    # The client has no version parameter, but the REST API does - and we need
    # it: the latest version of this mirror was "cleaned" in a way that dropped
    # StartTime, SrcAddr and DstAddr. Those are precisely the columns a windowed
    # host graph is built from, so the tidiest version of the data is the one
    # version that cannot be used here.
    import base64
    import json as _json
    import urllib.request
    import zipfile

    cfg = _json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())
    auth = base64.b64encode(f"{cfg['username']}:{cfg['key']}".encode()).decode()
    url = (f"https://www.kaggle.com/api/v1/datasets/download/{SLUG}"
           f"?datasetVersionNumber={version}")
    req = urllib.request.Request(url, headers={"Authorization": f"Basic {auth}"})
    zpath = DEST / f"ctu13-v{version}.zip"
    print(f"Downloading {SLUG} v{version} into {DEST} ...")
    with urllib.request.urlopen(req, timeout=120) as r, zpath.open("wb") as fh:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            fh.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done/1e6:6.0f} / {total/1e6:.0f} MB", end="", flush=True)
    print()
    with zipfile.ZipFile(zpath) as z:
        z.extractall(DEST)
    zpath.unlink()
    return 0


def report() -> int:
    """Say what arrived, in the terms the experiment needs."""
    if not DEST.exists():
        print(f"{DEST} does not exist - run without --check first.")
        return 1

    files = sorted(p for p in DEST.rglob("*")
                   if p.suffix in (".parquet", ".csv") and p.is_file())
    if not files:
        print(f"No parquet or csv files under {DEST}.")
        return 1

    total = sum(p.stat().st_size for p in files)
    print(f"\n{len(files)} data file(s), {total/1e6:.0f} MB total\n")

    try:
        import pandas as pd
    except ImportError:
        for p in files:
            print(f"  {p.name:52s} {p.stat().st_size/1e6:7.1f} MB")
        return 0

    print(f"  {'file':46s} {'rows':>10s} {'botnet':>8s} {'span (h)':>9s}")
    ok = 0
    for p in files:
        try:
            df = pd.read_parquet(p) if p.suffix == ".parquet" else \
                pd.read_csv(p, low_memory=False)
        except Exception as e:
            print(f"  {p.name:46s}  unreadable: {e}")
            continue

        lab = next((c for c in df.columns if c.lower() == "label"), None)
        st = next((c for c in df.columns if c.lower() in ("starttime", "stime")), None)
        bot = 0
        if lab is not None:
            bot = int(df[lab].astype(str).str.contains("otnet", case=False).sum())
        hours = float("nan")
        if st is not None:
            t = pd.to_datetime(df[st], errors="coerce", format="mixed")
            hours = (t.max() - t.min()).total_seconds() / 3600.0
        print(f"  {p.name:46s} {len(df):10,} {bot:8,} {hours:9.1f}")
        ok += 1

    if ok:
        df = pd.read_parquet(files[0]) if files[0].suffix == ".parquet" else \
            pd.read_csv(files[0], nrows=5, low_memory=False)
        print("\ncolumns in the first file:")
        print("  " + ", ".join(map(str, df.columns)))
        missing = [c for c in ("StartTime", "SrcAddr", "DstAddr", "Label")
                   if c not in df.columns]
        if missing:
            print(f"\n  WARNING: missing {missing} - the adapter needs these to "
                  f"build host graphs over time.")
        else:
            print("\n  StartTime / SrcAddr / DstAddr / Label all present - "
                  "the adapter can build windowed host graphs from this.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report on an existing download without fetching")
    ap.add_argument("--version", type=int, default=None,
                    help="dataset version. v1 is the base CSV; the latest was "
                         "cleaned in a way that drops StartTime and the IPs")
    args = ap.parse_args()
    if not args.check:
        rc = download(args.version)
        if rc:
            return rc
    return report()


if __name__ == "__main__":
    raise SystemExit(main())
