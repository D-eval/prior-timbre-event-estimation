import json
import numpy as np
import matplotlib.pyplot as plt

# ---- Load JSON ----
with open("./fig/save_dir/eval_results.json", "r") as f:
    data = json.load(f)

# data[k] = [lead, lag, mean_error, surplus, dE_avg]

mean_errors = []
energy_drop = []

for k, v in data.items():
    mean_errors.append(float(v[2]))   # 第3个指标
    energy_drop.append(float(v[4]))   # 第5个指标

mean_errors = np.array(mean_errors)
energy_drop = np.array(energy_drop)

print("Loaded", len(mean_errors), "samples.")


def report_stats(name, arr):
    print(f"\n=== {name} Statistics ===")
    print("Mean :", np.mean(arr))
    print("Std  :", np.std(arr))
    print("Median:", np.median(arr))
    print("Min :", np.min(arr))
    print("Max :", np.max(arr))

report_stats("Mean Prediction Error", mean_errors)
report_stats("Normalized Energy Drop", energy_drop)

plt.figure(figsize=(6,4))
plt.hist(mean_errors, bins=15, color='black', alpha=0.7)
plt.xlabel("Mean Prediction Error (frames)")
plt.ylabel("Count")
plt.title("Distribution of Mean Prediction Error")
plt.tight_layout()
plt.savefig("mean_error_hist.pdf", dpi=300)
plt.close()


plt.figure(figsize=(6,4))
plt.hist(energy_drop, bins=15, color='black', alpha=0.7)
plt.xlabel("Normalized Energy Drop")
plt.ylabel("Count")
plt.title("Distribution of Average Energy Reduction")
plt.tight_layout()
plt.savefig("energy_drop_hist.pdf", dpi=300)
plt.close()


plt.figure(figsize=(5,4))
plt.boxplot(mean_errors, vert=True)
plt.ylabel("Mean Prediction Error (frames)")
plt.title("Boxplot of Mean Prediction Error")
plt.tight_layout()
plt.savefig("mean_error_box.pdf", dpi=300)
plt.close()

plt.figure(figsize=(5,4))
plt.boxplot(energy_drop, vert=True)
plt.ylabel("Normalized Energy Drop")
plt.title("Boxplot of Energy Drop")
plt.tight_layout()
plt.savefig("energy_drop_box.pdf", dpi=300)
plt.close()



plt.figure(figsize=(6,5))
plt.scatter(mean_errors, energy_drop, s=40, c='black')
plt.xlabel("Mean Prediction Error (frames)")
plt.ylabel("Normalized Energy Drop")
plt.title("Correlation Between Temporal Error and Energy Drop")
plt.tight_layout()
plt.savefig("error_vs_energy.pdf", dpi=300)
plt.close()


print("\n=== Summary Table (for paper) ===")
print(f"Mean Error: {np.mean(mean_errors):.2f} ± {np.std(mean_errors):.2f}")
print(f"Energy Drop: {np.mean(energy_drop):.3f} ± {np.std(energy_drop):.3f}")
