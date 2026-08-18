import json

nb_path = "2547145_Lab3.ipynb"
with open(nb_path, "r") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") != "code":
        continue
    
    source_str = "".join(cell.get("source", []))
    
    # 1. Update the outlier handling cell
    if "Handle outliers:" in source_str:
        cell["source"] = [
            "# Handle outliers using IQR Method\n",
            "for col in ['CIA_Percentage', 'GPA', 'Attendance_Percentage']:\n",
            "    Q1 = df_reg[col].quantile(0.25)\n",
            "    Q3 = df_reg[col].quantile(0.75)\n",
            "    IQR = Q3 - Q1\n",
            "    lower_bound = Q1 - 1.5 * IQR\n",
            "    upper_bound = Q3 + 1.5 * IQR\n",
            "    # We will replace outliers with NaN first, then impute\n",
            "    df_reg.loc[(df_reg[col] < lower_bound) | (df_reg[col] > upper_bound), col] = np.nan\n",
            "\n",
            "print(\"After conversion and IQR outlier removal:\")\n",
            "print(df_reg.dtypes)\n",
            "print(f\"\\nNull values after conversion:\")\n",
            "print(df_reg.isnull().sum())\n"
        ]
        
    # 2. Update the NaN handling cell (drop -> mean imputation)
    if "Drop rows with NaN values" in source_str:
        cell["source"] = [
            "# Updated EDA: Fill missing values with mean (Mean Imputation)\n",
            "df_reg['CIA_Percentage'] = df_reg['CIA_Percentage'].fillna(df_reg['CIA_Percentage'].mean())\n",
            "df_reg['GPA'] = df_reg['GPA'].fillna(df_reg['GPA'].mean())\n",
            "df_reg['Attendance_Percentage'] = df_reg['Attendance_Percentage'].fillna(df_reg['Attendance_Percentage'].mean())\n",
            "\n",
            "print(f\"After Mean Imputation:\")\n",
            "print(f\"Shape: {df_reg.shape}\")\n",
            "print(f\"Null values: {df_reg.isnull().sum().sum()}\")\n",
            "print(f\"Duplicate rows before removal: {df_reg.duplicated().sum()}\")\n"
        ]

    # 3. Revert random_state=71 back to random_state=42
    if "random_state=" in source_str:
        new_source = []
        for line in cell["source"]:
            new_source.append(line.replace("random_state=71", "random_state=42"))
        cell["source"] = new_source

with open(nb_path, "w") as f:
    json.dump(nb, f, indent=1)

print("Notebook EDA updated successfully.")
