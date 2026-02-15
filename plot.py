import re
import numpy as np
import matplotlib.pyplot as plt

with open("training.log") as f:
    log_text = f.read()

pattern = r"Step:\s*(\d+)\s*Training Loss:\s*([\d.]+)\s*Validation Loss:\s*([\d.]+)"

steps, train, val = [], [], []

for s, t, v in re.findall(pattern, log_text):
    steps.append(int(s))
    train.append(float(t))
    val.append(float(v))

steps = np.array(steps)
train = np.array(train)
val = np.array(val)

# -----------------------------
# Smooth curve (moving average)
# -----------------------------
def smooth(y, window=3):
    return np.convolve(y, np.ones(window)/window, mode="same")

train_s = smooth(train)
val_s = smooth(val)

# -----------------------------
# Plot
# -----------------------------
# plt.rcParams['font.family'] = 'DejaVu Sans'
# plt.rcParams["font.family"] = "Comic Neue"
plt.xkcd()
plt.rcParams["font.family"] = "Comic Neue"
plt.figure(figsize=(12,7))
plt.plot(steps, train, "--", alpha=0.3)
plt.plot(steps, val, "--", alpha=0.3)
plt.plot(steps, train_s, linewidth=3, label="Training Loss")
plt.plot(steps, val_s, linewidth=3, label="Validation Loss")
plt.xlabel("Training Steps", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.title("Training Loss Curve", fontsize=16)
plt.legend()
plt.grid(True)
# -----------------------------
# AXIS SCALE SETTINGS (your request)
# -----------------------------
# Y axis ticks every 1
ymin = int(min(train.min(), val.min())) - 1
ymax = int(max(train.max(), val.max())) + 1
plt.yticks(np.arange(ymin, ymax, 1))
# X axis ticks every 1000
xmax = steps.max()
plt.xticks(np.arange(0, xmax + 1000, 1000))
# -----------------------------
# Save image
# -----------------------------
plt.tight_layout()
plt.savefig("lcurve.png", dpi=300)