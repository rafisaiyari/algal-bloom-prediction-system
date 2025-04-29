import pandas as pd
import os
from datetime import datetime


def validate_stations(filepath):
    # Read the CSV file
    df = pd.read_excel("merged_stations.xlsx")

    # Expected station names
    expected_stations = [
        'Station_1_CWB',
        'Station_2_EastB',
        'Station_4_CentralB',
        'Station_5_NorthernWestBay',
        'Station_8_SouthB',
        'Station_15_SanPedro',
        'Station_16_Sta. Rosa',
        'Station_17_Sanctuary',
        'Station_18_Pagsanjan'
    ]

    # Check if all expected stations exist in the dataset
    actual_stations = df['Station'].unique()
    missing_stations = set(expected_stations) - set(actual_stations)
    extra_stations = set(actual_stations) - set(expected_stations)

    if missing_stations:
        print(f"Warning: Missing stations in dataset: {missing_stations}")

    if extra_stations:
        print(f"Warning: Extra stations in dataset: {extra_stations}")

    # Convert date column to datetime for easier month extraction
    df['Date'] = pd.to_datetime(df['Date'])

    # Extract month and year for grouping
    df['Month_Year'] = df['Date'].dt.strftime('%Y-%m')

    # Group by month and get station counts
    monthly_counts = df.groupby(['Month_Year', 'Station']).size().reset_index(name='count')

    # Create a pivot table to see which stations have data for each month
    station_matrix = monthly_counts.pivot_table(
        index='Month_Year',
        columns='Station',
        values='count',
        fill_value=0
    )

    # Find months with incomplete station data
    incomplete_months = []
    for month, row in station_matrix.iterrows():
        missing = [station for station in expected_stations if
                   station not in station_matrix.columns or row[station] == 0]
        if missing:
            incomplete_months.append((month, missing))

    # Print validation results
    print(f"\nTotal timeframes (months) in dataset: {len(station_matrix)}")

    if not incomplete_months:
        print("All timeframes have complete data for all 9 stations!")
    else:
        print(f"\n{len(incomplete_months)} timeframes have incomplete station data:")
        for month, missing in incomplete_months:
            print(f"  - {month}: Missing {len(missing)} stations: {', '.join(missing)}")

    # Print the station availability matrix for visualization
    print("\nStation data availability by month:")
    print(station_matrix)

    return station_matrix, incomplete_months


if __name__ == "__main__":
    filepath = "train/merged_stations.xlsx"  # Update here too
    if os.path.exists(filepath):
        validate_stations(filepath)
    else:
        print(f"Error: File {filepath} not found.")