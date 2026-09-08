#!/usr/bin/env python3
"""Print the decode ms/token from the newest bench_<name>_*.json in a log dir."""
import glob, json, os, sys
f = sorted(glob.glob(os.path.join(sys.argv[1], f"bench_{sys.argv[2]}_*.json")))
# .get chain, not indexing: a bench that refused to run writes no summary, and this
# helper feeding "" to --wall-ms is a missing host split, not a crash.
print((json.load(open(f[-1])).get("summary") or {}).get("decode_ms_per_token", "") if f else "")
