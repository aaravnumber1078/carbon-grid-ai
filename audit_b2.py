import json
import numpy as np

with open("data/processed/b2_dataset.json", "r") as f:
    data = json.load(f)

complete = [
    r for r in data
    if r.get("measurement_status") == "complete"
]

diffs = np.array([
    r["int4_energy_wh"] - r["fp16_energy_wh"]
    for r in complete
])

print("=" * 60)
print("B2 ENERGY DIFFERENCE ANALYSIS")
print("=" * 60)

print(f"Complete samples: {len(complete)}")

print("\nSIGN COUNTS")
print(f"INT4 < FP16 : {np.sum(diffs < 0)}")
print(f"INT4 > FP16 : {np.sum(diffs > 0)}")
print(f"INT4 = FP16 : {np.sum(diffs == 0)}")

print("\nENERGY DIFFERENCE")
print(f"Mean   : {np.mean(diffs):.6f} Wh")
print(f"Median : {np.median(diffs):.6f} Wh")
print(f"Std    : {np.std(diffs):.6f} Wh")
print(f"Min    : {np.min(diffs):.6f} Wh")
print(f"Max    : {np.max(diffs):.6f} Wh")

print("\nSTATIC PHASE 0.5 BASELINE")

static_fp16 = 0.0488
static_int4 = 0.0493
static_diff = static_int4 - static_fp16

baseline_mae = np.mean(np.abs(diffs - static_diff))

print(f"Static FP16 : {static_fp16:.4f} Wh")
print(f"Static INT4 : {static_int4:.4f} Wh")
print(f"Static diff : {static_diff:.4f} Wh")
print(f"Baseline MAE: {baseline_mae:.6f} Wh")

print("\nDIRECTION")
print(
    f"Static baseline directional accuracy: "
    f"{np.mean(np.sign(diffs) == np.sign(static_diff))*100:.1f}%"
)

print("=" * 60)
