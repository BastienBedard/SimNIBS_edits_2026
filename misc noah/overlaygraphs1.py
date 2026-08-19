#overlaygraphs1
import csv
from pathlib import Path

import matplotlib.pyplot as plt

#This script is used to overlay multiple graphs into one graph.

# ------------------------------------------------------------
# Folder containing saved graph data
# ------------------------------------------------------------

graph_data_folder = Path.home() / "Desktop" / "meshcomparegraphdata"

if not graph_data_folder.exists():
    raise FileNotFoundError(f"Folder does not exist: {graph_data_folder}")


# ------------------------------------------------------------
# Find all CSV graph-data files
# ------------------------------------------------------------

csv_files = sorted(graph_data_folder.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(f"No .csv files found in: {graph_data_folder}")


# ------------------------------------------------------------
# Read and plot each graph
# ------------------------------------------------------------

plt.figure()

for csv_file in csv_files:
    tetra_counts = []
    relative_errors = []

    with open(csv_file, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            tetra_counts.append(float(row["tetrahedra_kept"]))
            relative_errors.append(float(row["relative_error_percent"]))

    # Use filename without extension as the legend label
    label = csv_file.stem

    plt.plot(
        tetra_counts,
        relative_errors,
        marker="o",
        label=label
    )


# ------------------------------------------------------------
# Format overlay plot
# ------------------------------------------------------------

plt.xlabel("Number of tetrahedra kept")
plt.ylabel("Relative vector-field error compared to reference (%)")
plt.title("Overlay of meshcompare convergence graphs")

plt.grid(True)
plt.ylim(bottom=0)

plt.legend(fontsize=8)
plt.tight_layout()

plt.show()