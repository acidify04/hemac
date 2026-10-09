#!/usr/bin/env bash
set -euo pipefail

# Evaluate decentralized BC with the exact population/seed protocol used for
# the HiSSD zero-shot population-generalization evaluation.
#
# Usage:
#   bash src/skill_discovery/run_decentralized_bc_hissd_matrix.sh [difficulty] [episodes]
#
# Examples:
#   bash src/skill_discovery/run_decentralized_bc_hissd_matrix.sh 1 50
#   bash src/skill_discovery/run_decentralized_bc_hissd_matrix.sh 2 50

DIFFICULTY="${1:-1}"
EPISODES="${2:-50}"

CKPT="${CKPT:-src/skill_discovery/checkpoints/decentralized-bc-source-combined/decentralized_bc_best.pt}"
EVAL="${EVAL:-src/skill_discovery/evaluate_decentralized_bc_hissd_protocol.py}"
ENV_ROOT="${ENV_ROOT:-src/skill_discovery/offline_data-4-2}"
SEED_BASE="${SEED_BASE:-100000000}"
DEVICE="${DEVICE:-cuda}"
OUT="${OUT:-src/skill_discovery/checkpoints/decentralized-bc-source-combined/eval-d${DIFFICULTY}}"

mkdir -p "$OUT"

echo "============================================================"
echo "Decentralized BC / HiSSD-matched evaluation"
echo "checkpoint : $CKPT"
echo "difficulty : $DIFFICULTY"
echo "episodes   : $EPISODES"
echo "seed_base  : $SEED_BASE"
echo "output     : $OUT"
echo "============================================================"

for SPEC in \
  "2 1" \
  "2 2" \
  "3 1" \
  "3 2" \
  "4 1" \
  "4 2" \
  "5 1" \
  "5 2"
do
  read -r D O <<< "$SPEC"

  echo
  echo "------------------------------------------------------------"
  echo "D${DIFFICULTY} / D${D}O${O}"
  echo "------------------------------------------------------------"

  python -u "$EVAL" \
    --checkpoint "$CKPT" \
    --env-template-data-root "$ENV_ROOT" \
    --n-drones "$D" \
    --n-observers "$O" \
    --difficulty "$DIFFICULTY" \
    --episodes "$EPISODES" \
    --seed-base "$SEED_BASE" \
    --device "$DEVICE" \
    --output-json "$OUT/D${D}O${O}.json"
done

python - "$OUT" "$DIFFICULTY" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
difficulty = int(sys.argv[2])

populations = [
    ("D2O1", 2, 1),
    ("D2O2", 2, 2),
    ("D3O1", 3, 1),
    ("D3O2", 3, 2),
    ("D4O1", 4, 1),
    ("D4O2", 4, 2),
    ("D5O1", 5, 1),
    ("D5O2", 5, 2),
]

rows = []
for name, d, o in populations:
    path = out / f"{name}.json"
    payload = json.loads(path.read_text())
    s = payload["summary"]
    rows.append({
        "population": name,
        "n_drones": d,
        "n_observers": o,
        "source_population": name in {"D3O1", "D4O2"},
        "success": s["success"],
        "goal_found": s["goal_found"],
        "fatal_crash": s["fatal_crash"],
        "drone_crash": s["drone_crash"],
        "observer_crash": s["observer_crash"],
        "coverage": s["coverage"],
        "cycles": s["cycles"],
    })

def mean(key, selected):
    vals = [r[key] for r in rows if selected(r)]
    return sum(vals) / len(vals)

aggregate = {
    "difficulty": difficulty,
    "overall_success": mean("success", lambda r: True),
    "source_success": mean("success", lambda r: r["source_population"]),
    "unseen_success": mean("success", lambda r: not r["source_population"]),
    "o1_success": mean("success", lambda r: r["n_observers"] == 1),
    "o2_success": mean("success", lambda r: r["n_observers"] == 2),
}

summary = {
    "evaluation": "decentralized_bc_hissd_protocol_matrix",
    "difficulty": difficulty,
    "rows": rows,
    "aggregate": aggregate,
}
(out / "summary.json").write_text(json.dumps(summary, indent=2))

print()
print("============================================================")
print("FINAL MATRIX")
print("============================================================")
print(
    f"{'Population':10s} {'Success':>8s} {'Goal':>8s} "
    f"{'Fatal':>8s} {'Coverage':>10s}"
)
for r in rows:
    print(
        f"{r['population']:10s} "
        f"{r['success']*100:7.1f}% "
        f"{r['goal_found']*100:7.1f}% "
        f"{r['fatal_crash']*100:7.1f}% "
        f"{r['coverage']:10.4f}"
    )

print()
print(f"Overall success = {aggregate['overall_success']*100:.2f}%")
print(f"Source success  = {aggregate['source_success']*100:.2f}%")
print(f"Unseen success  = {aggregate['unseen_success']*100:.2f}%")
print(f"O1 success      = {aggregate['o1_success']*100:.2f}%")
print(f"O2 success      = {aggregate['o2_success']*100:.2f}%")
print(f"summary         = {out / 'summary.json'}")
PY
