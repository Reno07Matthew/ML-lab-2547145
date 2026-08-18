import json

with open("2547145_Lab3.ipynb") as f:
    nb = json.load(f)

# Insert the LR class definition before cell 35
lr_class_cell = {
    "cell_type": "code",
    "metadata": {},
    "source": [
        "### Create Custom Linear Regression Class\n",
        "class LR:\n",
        "    def __init__(self):\n",
        "        self.m = None\n",
        "        self.b = None\n",
        "\n",
        "    def fit(self, X_train, y_train):\n",
        "        # Manual OLS Computation using the deviation-from-mean OLS formula\n",
        "        num = 0\n",
        "        den = 0\n",
        "        x_mean = X_train.mean()\n",
        "        y_mean = y_train.mean()\n",
        "        \n",
        "        for i in range(X_train.shape[0]):\n",
        "            num = num + ((X_train[i] - x_mean) * (y_train[i] - y_mean))\n",
        "            den = den + ((X_train[i] - x_mean) * (X_train[i] - x_mean))\n",
        "            \n",
        "        self.m = num / den\n",
        "        self.b = y_mean - (self.m * x_mean)\n",
        "        \n",
        "        print(\"Slope (m) =\", self.m)\n",
        "        print(\"Intercept (b) =\", self.b)\n",
        "\n",
        "    def predict(self, X_test):\n",
        "        return self.m * X_test + self.b\n"
    ],
    "execution_count": None,
    "outputs": []
}

# Find the cell index for Cell 35
index_35 = 35
for i, cell in enumerate(nb["cells"]):
    source_str = "".join(cell.get("source", []))
    if "Manual OLS Computation for Experiment 1" in source_str:
        index_35 = i
        break

# We can insert lr_class_cell at index_35
nb["cells"].insert(index_35, lr_class_cell)

# Now find the indices again, since everything shifted by 1
index_35 += 1

# Update Cell 35 (now index_35)
nb["cells"][index_35]["source"] = [
    "# Manual OLS Computation for Experiment 1 using Custom LR Class\n",
    "lr1 = LR()\n",
    "x_train = X1_train.flatten()\n",
    "y_train = Y1_train\n",
    "\n",
    "print(\"Manual OLS Results (CIA % → GPA):\\n\")\n",
    "lr1.fit(x_train, y_train)\n",
    "\n",
    "slope1_manual = lr1.m\n",
    "intercept1_manual = lr1.b\n",
    "print(f\"\\nRegression Equation: GPA = {slope1_manual:.6f} * CIA_Percentage + {intercept1_manual:.6f}\")\n"
]

# Update Cell 36 (now index_35 + 1)
nb["cells"][index_35 + 1]["source"] = [
    "# Predict using custom LR class\n",
    "Y1_pred_manual = lr1.predict(X1_test.flatten())\n",
    "\n",
    "print(\"Predictions from Custom LR (CIA % → GPA):\\n\")\n",
    "results1_manual = pd.DataFrame({\n",
    "    \"CIA_Percentage\": X1_test.flatten(),\n",
    "    \"Actual_GPA\": Y1_test,\n",
    "    \"Predicted_GPA (Custom LR)\": Y1_pred_manual.round(4)\n",
    "})\n",
    "print(results1_manual.to_string(index=False))\n"
]

# Update Cell 51 (which has now shifted down, we should search for it)
index_51 = 0
for i, cell in enumerate(nb["cells"]):
    source_str = "".join(cell.get("source", []))
    if "Manual OLS Computation for Experiment 2" in source_str:
        index_51 = i
        break

nb["cells"][index_51]["source"] = [
    "# Manual OLS Computation for Experiment 2 using Custom LR Class\n",
    "lr2 = LR()\n",
    "x_train2 = X2_train.flatten()\n",
    "y_train2 = Y2_train\n",
    "\n",
    "print(\"Manual OLS Results (Attendance % → GPA):\\n\")\n",
    "lr2.fit(x_train2, y_train2)\n",
    "\n",
    "slope2_manual = lr2.m\n",
    "intercept2_manual = lr2.b\n",
    "print(f\"\\nRegression Equation: GPA = {slope2_manual:.6f} * Attendance_Percentage + {intercept2_manual:.6f}\")\n"
]

# Update Cell 52 (which is index_51 + 1)
nb["cells"][index_51 + 1]["source"] = [
    "# Predict using custom LR class\n",
    "Y2_pred_manual = lr2.predict(X2_test.flatten())\n",
    "\n",
    "print(\"Predictions from Custom LR (Attendance % → GPA):\\n\")\n",
    "results2_manual = pd.DataFrame({\n",
    "    \"Attendance_Percentage\": X2_test.flatten(),\n",
    "    \"Actual_GPA\": Y2_test,\n",
    "    \"Predicted_GPA (Custom LR)\": Y2_pred_manual.round(4)\n",
    "})\n",
    "print(results2_manual.to_string(index=False))\n"
]

with open("2547145_Lab3.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print("Notebook cells updated.")
