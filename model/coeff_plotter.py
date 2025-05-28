import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import csv
import pickle
import os

# 全局统一字体大小，避免局部更新导致图内空白不一致
matplotlib.rcParams.update({"font.size": 12})

# ------------------------------------------------------------------
# Common metadata (cycle headers + feature names)
# ------------------------------------------------------------------
filename = "./results/norm_coeffs.csv"

# --- cycles
with open(filename, "rt") as f:
    reader = csv.reader(f)
    Cycles = next(reader)           # header row
    raw_rows = list(reader)         # rest of file (to extract feature names)

# remove the first blank left‑top cell
Cycles = Cycles[1:]

# --- feature names
Features = [row[0] for row in raw_rows]

nF = len(Features)
nC = len(Cycles)

# ------------------------------------------------------------------
# Helper to plot & save a single heat‑map
# ------------------------------------------------------------------
def _plot_heatmap(FMatrix, model_tag: str):
    """
    Plot and save heat‑map of feature importance / coefficients.
    `FMatrix` shape: n_features × n_cycles
    """
    FMatrix = np.abs(FMatrix * 100)   # convert to percentage
    FMatrix[FMatrix == 0] = np.nan    # hide zeros

    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    im = ax.imshow(FMatrix)

    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Feature importance (%)",
                       rotation=-90, va="bottom", size=18)

    # Ticks
    ax.set_xticks(np.arange(len(Cycles)))
    ax.set_xticklabels(Cycles)
    ax.set_yticks(np.arange(len(Features)))
    ax.set_yticklabels(Features)

    # Annotate cells
    for i in range(nF):
        for j in range(nC):
            if not np.isnan(FMatrix[i, j]):
                text = ax.text(
                    j, i,
                    int(np.round(FMatrix[i, j], 0)),
                    ha="center", va="center", color="w"
                )

    ax.set_title(f"Feature importance for {model_tag}", weight="bold", size=20)
    fig.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)

    # Ensure output folder
    os.makedirs("./plots", exist_ok=True)
    outfile = f"./plots/{model_tag}_features.png"
    plt.savefig(outfile)
    plt.close(fig)
    print(f"[saved] {outfile}")


# ------------------------------------------------------------------
# Iterate over all three models
# ------------------------------------------------------------------
MODELS = ["AB", "RF", "enet"]

for MODEL in MODELS:
    if MODEL == "enet":
        # ElasticNet coefficients are stored directly in CSV
        FMatrix = np.genfromtxt(filename, delimiter=",")[1:, 1:]
    else:
        coeff_path = f"./results/{MODEL}_features_coeffs.pkl"
        if not os.path.isfile(coeff_path):
            raise FileNotFoundError(f"{coeff_path} not found, "
                                    "please run `model.py` to generate it.")
        FMatrix = pickle.load(open(coeff_path, "rb"))

    _plot_heatmap(FMatrix, MODEL)

print("All feature‑importance plots generated in ./plots/")