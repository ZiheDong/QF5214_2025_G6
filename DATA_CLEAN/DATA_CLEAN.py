# Name: Dong Zihe, Rong Yining

import os
import pandas as pd
import numpy as np
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

# === Configuration ===
DataDir = './Data'
OutputDir = './Data_Clean'
MaxIterations = 10   # Fixed Number Of Space Interpolation Iterations
NeighborSize = 1     # Neighborhood = 3x3 Window

os.makedirs(OutputDir, exist_ok=True)

# === Build Full Coordinate-Time Grid ===
print("Building Full Coordinate-Time Grid...")

Latitudes = np.arange(24, 50)
Longitudes = np.arange(-125, -65)
Dates = pd.date_range('2018-01-01', '2024-12-31')

FullGrid = pd.MultiIndex.from_product(
    [Longitudes, Latitudes, Dates],
    names=['Longitude', 'Latitude', 'Timestamp']
).to_frame(index=False)

print(f"✔ Full Grid Generated With {FullGrid.shape[0]} Rows.")

# === CSV Processing ===
CSVFiles = [f for f in os.listdir(DataDir) if f.endswith('.csv') and f.startswith('US_')]
RepairInfo = {}

print("\nStarting Time + Local Space Interpolation...\n")

for File in tqdm(CSVFiles, desc="Processing CSV Files"):
    FilePath = os.path.join(DataDir, File)
    DF = pd.read_csv(FilePath)
    VarName = File.split('_')[1]

    if DF.shape[1] < 4:
        print(f"⚠ Warning: {File} Skipped Due To Insufficient Columns.")
        continue

    # === Standardize CSV Column Order ===
    DF.columns = ['Latitude', 'Longitude', 'Timestamp', VarName]
    DF = DF[['Longitude', 'Latitude', 'Timestamp', VarName]]

    DF['Timestamp'] = pd.to_datetime(DF['Timestamp'])

    # === Merge With Full Grid ===
    Merged = pd.merge(FullGrid, DF, on=['Longitude', 'Latitude', 'Timestamp'], how='left')
    Merged[VarName] = Merged[VarName].replace(-999, np.nan)
    MissingBefore = Merged[VarName].isna().sum()

    # === Time Interpolation ===
    Merged.sort_values(by=['Latitude', 'Longitude', 'Timestamp'], inplace=True)
    Merged[VarName] = Merged.groupby(['Latitude', 'Longitude'])[VarName].transform(
        lambda Group: Group.interpolate(method='linear', limit_direction='both')
    )

    # === Reshape To Cube ===
    DataCube = Merged[VarName].values.reshape(len(Dates), len(Latitudes), len(Longitudes)).astype(np.float32)

    # === Local Space Interpolation ===
    print(f"\nInterpolating {VarName}: Fixed {MaxIterations} Iterations")

    for Iter in trange(MaxIterations, desc=f"Space Interpolation For {VarName}"):
        MissingMask = np.isnan(DataCube)
        MissingIndices = np.argwhere(MissingMask)

        for idx in tqdm(range(len(MissingIndices)), leave=False, desc=f"Iter {Iter+1} Fill Progress"):
            t, i, j = MissingIndices[idx]
            Window = DataCube[t,
                              max(i - NeighborSize, 0): i + NeighborSize + 1,
                              max(j - NeighborSize, 0): j + NeighborSize + 1]
            Value = np.nanmean(Window)
            if not np.isnan(Value):
                DataCube[t, i, j] = Value

    # === Flatten Back ===
    FlatValues = DataCube.reshape(-1)
    Merged[VarName] = FlatValues

    # === Repair Statistics ===
    MissingAfter = np.isnan(Merged[VarName]).sum()
    Repaired = MissingBefore - MissingAfter
    RepairRatio = Repaired / Merged.shape[0]
    RepairInfo[VarName] = RepairRatio

    # === Save Cleaned CSV ===
    OutputPath = os.path.join(OutputDir, File)
    Merged.to_csv(OutputPath, index=False)

# === Repair Ratio Visualization ===
plt.figure(figsize=(max(10, len(RepairInfo) * 0.4), 6))
plt.bar(RepairInfo.keys(), [v * 100 for v in RepairInfo.values()])
plt.ylabel("Repair Ratio (%)")
plt.xlabel("Variable")
plt.title("Time + Local Space Interpolation Repair Ratio")
plt.xticks(rotation=45, ha='right')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("Repair_Ratio_Per_Variable.png")
plt.show()

print("✔ All Cleaned CSV Files Saved To ./Data_Clean")
print("✔ Repair Ratio Plot Saved As Repair_Ratio_Per_Variable.png")
