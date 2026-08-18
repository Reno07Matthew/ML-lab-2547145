import json

nb_path = "2547145_Lab3.ipynb"
with open(nb_path, "r") as f:
    nb = json.load(f)

# 1. Replace random_state=42 with random_state=71 in all code cells
for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        new_source = []
        for line in cell.get("source", []):
            new_line = line.replace("random_state=42", "random_state=71")
            new_source.append(new_line)
        cell["source"] = new_source

# 2. Add a new code cell at the end to compute R2
r2_cell = {
    "cell_type": "code",
    "metadata": {},
    "source": [
        "from sklearn.metrics import r2_score\n",
        "\n",
        "r2_1 = r2_score(Y1_test, Y1_pred_sklearn)\n",
        "r2_2 = r2_score(Y2_test, Y2_pred_sklearn)\n",
        "\n",
        "print(\"=\" * 60)\n",
        "print(\"R-SQUARED (R²) EVALUATION\")\n",
        "print(\"=\" * 60)\n",
        "print(f\"Experiment 1 (CIA % -> GPA) R² Score: {r2_1:.4f}\")\n",
        "print(f\"Experiment 2 (Attendance % -> GPA) R² Score: {r2_2:.4f}\")\n",
        "print(\"=\" * 60)\n"
    ],
    "execution_count": None,
    "outputs": []
}

nb["cells"].append(r2_cell)

with open(nb_path, "w") as f:
    json.dump(nb, f, indent=1)

print("Notebook modified successfully.")
