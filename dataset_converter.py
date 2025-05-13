import pandas as pd

# Load the Excel file
file_path = "C:/Users/Legion 5 Pro/Downloads/water_quality_dataset.xlsx"
output_file = "CSV//merged_stations.xlsx"

# Read all sheets
xls = pd.ExcelFile(file_path)
merged_data = []

for sheet_name in xls.sheet_names:
    df = pd.read_excel(xls, sheet_name=sheet_name)

    # Replace "NaN" or "Nan" strings with actual NaN (empty) values
    df.replace(["NaN", "Nan"], pd.NA, inplace=True)

    df["Station"] = sheet_name  # Add station column
    merged_data.append(df)

# Concatenate all data into a single DataFrame
final_df = pd.concat(merged_data, ignore_index=True)

# Save to a new Excel file
final_df.to_excel(output_file, index=False)

print(f"Data merged successfully into {output_file}")
