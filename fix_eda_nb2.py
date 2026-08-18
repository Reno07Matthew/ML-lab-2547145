import json

nb_path = "2547145_Lab3.ipynb"
with open(nb_path, "r") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") != "code":
        continue
    
    source_str = "".join(cell.get("source", []))
    
    if "# Handle outliers using IQR Method\n" in source_str:
        if "to_numeric" not in source_str:
            new_source = [
                "# Convert to numeric, coercing errors to NaN\n",
                "df_reg['CIA_Percentage'] = pd.to_numeric(df_reg['CIA_Percentage'], errors='coerce')\n",
                "df_reg['GPA'] = pd.to_numeric(df_reg['GPA'], errors='coerce')\n",
                "df_reg['Attendance_Percentage'] = pd.to_numeric(df_reg['Attendance_Percentage'], errors='coerce')\n",
                "\n",
                "# Handle values entered as decimals\n",
                "df_reg.loc[df_reg['CIA_Percentage'] < 1, 'CIA_Percentage'] = df_reg.loc[df_reg['CIA_Percentage'] < 1, 'CIA_Percentage'] * 100\n",
                "df_reg.loc[df_reg['Attendance_Percentage'] < 1, 'Attendance_Percentage'] = df_reg.loc[df_reg['Attendance_Percentage'] < 1, 'Attendance_Percentage'] * 100\n",
                "\n"
            ] + cell["source"]
            
            cell["source"] = new_source

with open(nb_path, "w") as f:
    json.dump(nb, f, indent=1)

print("Notebook EDA fixed successfully.")
