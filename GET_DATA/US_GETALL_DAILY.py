# Name:Dong Zihe, Rong Yining, Gao Xinyu

import requests
import pandas as pd
from datetime import datetime
import time

# The latitude and longitude range of the continental United States (excluding Hawaii and Alaska)
latitude_min, latitude_max = 24, 49   # 24°N - 49°N
longitude_min, longitude_max = -125, -66  # -125°W - -66°W
grid_step = 1  # Grid spacing 1°

# Time Range
start_date = "20180101"
end_date = "20241231"

# NASA POWER API
api_url = "https://power.larc.nasa.gov/api/temporal/daily/point"

# Variable List
variables = [
    "T2M", "RH2M", "WS2M", "T2MDEW", "T2M_MAX", "T2M_MIN", "Surface Pressure", "T2MWET", "WS50M",
    "WD50M", "WD2M", "PRECTOTCORR", "WS10M", "GWETPROF", "TS", "WD10M", "QV2M", "T2M_RANGE",
    "GWETROOT", "GWETTOP", "WS10M_MAX", "WS10M_MIN", "WS10M_RANGE", "WS50M_MAX", "WS50M_MIN",
    "WS50M_RANGE", "FROST_DAYS", "HDD0", "HDD18_3", "SNODP", "TS_MAX", "TS_MIN", "WS2M_MAX",
    "WS2M_MIN", "WS2M_RANGE", "PW", "U10M", "V10M", "EVLAND", "EVPTRNS", "FRSEAICE",
    "FRSNO", "IMERG_PRECLIQUID_PROB", "IMERG_PRECTOT", "IMERG_PRECTOT_COUNT", "PBLTOP", "PRECSNO",
    "PRECSNOLAND", "PSH", "QV10M", "RHOA", "SLP", "T10M", "T10M_MAX", "T10M_MIN", "T10M_RANGE",
    "TO3", "TQV", "TROPPB", "TROPQ", "TROPT", "TSOIL1", "TSOIL2", "TSOIL3", "TSOIL4", "TSOIL5",
    "TSOIL6", "TSURF", "TS_ADJ", "TS_RANGE", "U2M", "U50M", "V2M", "V50M", "Z0M", "ALLSKY_SFC_SW_DWN",
    "ALLSKY_SFC_LW_DWN", "ALLSKY_SFC_SW_DIFF", "ALLSKY_SFC_SW_DNI", "ALLSKY_SRF_ALB", "CLOUD_AMT",
    "TOA_SW_DWN", "ALLSKY_SFC_PAR_TOT", "ALLSKY_KT", "CLRSKY_SFC_SW_DWN", "ALLSKY_SFC_UVA",
    "ALLSKY_SFC_UVB", "ALLSKY_SFC_UV_INDEX", "CDD0", "CDD18_3", "CLRSKY_KT", "CLRSKY_SFC_PAR_TOT",
    "MIDDAY_INSOL", "SZA", "TOA_SW_DNI", "AOD_55", "AIRMASS", "ALLSKY_NKT", "ALLSKY_SFC_LW_UP",
    "ALLSKY_SFC_PAR_DIFF", "ALLSKY_SFC_PAR_DIRH", "ALLSKY_SFC_SW_DIRH", "ALLSKY_SFC_SW_UP",
    "AOD_55_ADJ", "AOD_84", "CLOUD_AMT_DAY", "CLOUD_AMT_NIGHT", "CLOUD_OD", "CLRSKY_DAYS", 
    "CLRSKY_NKT", "CLRSKY_SFC_LW_DWN", "CLRSKY_SFC_LW_UP", "CLRSKY_SFC_PAR_DIFF", 
    "CLRSKY_SFC_PAR_DIRH", "CLRSKY_SFC_SW_DIFF", "CLRSKY_SFC_SW_DIRH", "CLRSKY_SFC_SW_DNI", 
    "CLRSKY_SFC_SW_UP", "CLRSKY_SRF_ALB", "DISPH", "ORIGINAL_ALLSKY_SFC_LW_DWN", 
    "ORIGINAL_ALLSKY_SFC_SW_DIFF", "ORIGINAL_ALLSKY_SFC_SW_DIRH", "ORIGINAL_ALLSKY_SFC_SW_DWN", 
    "ORIGINAL_CLRSKY_SFC_LW_DWN", "ORIGINAL_CLRSKY_SFC_SW_DWN", "SRF_ALB_ADJ"
]

for variable in variables:
    print(f"📍 Getting variable {variable} ...")

    # Data Records List
    data_records = []

    # Calculate all latitude and longitude points
    latitudes = range(latitude_min, latitude_max + 1, grid_step)
    longitudes = range(longitude_min, longitude_max + 1, grid_step)

    print(f"📍 Expected request for {len(latitudes) * len(longitudes)} sample points")

    # Traverse the entire area (divided into grids)
    for lat in latitudes:
        for lon in longitudes:
            print(f"📍 Getting {variable} data for {lat}N, {lon}W ...")

            # API params
            params = {
                "parameters": variable,
                "community": "RE",
                "longitude": lon,
                "latitude": lat,
                "start": start_date,
                "end": end_date,
                "format": "JSON"
            }

            # Send Request
            response = requests.get(api_url, params=params)

            if response.status_code == 200:
                data = response.json()

                if "properties" in data and "parameter" in data["properties"]:
                    # Get Variable Data
                    var_data = data["properties"]["parameter"].get(variable, {})

                    # Iterate through Timestamps
                    for timestamp, value in var_data.items():
                        data_records.append([lat, lon, timestamp, value])

                    print(f"✅  {variable} data of {lat}N, {lon}W was successfully obtained")
                else:
                    print(f"⚠️Data for {variable} is missing for {lat}N, {lon}W")

            else:
                print(f"❌ {lat}N, {lon}W Request failed, status code: {response.status_code}, error message: {response.text}")

            # Rest for 0.5 seconds to prevent API throttling
            time.sleep(0.5)

    # Convert Data to DataFrame
    df = pd.DataFrame(data_records, columns=["Latitude", "Longitude", "Timestamp", variable])

    # Formatting the Time Column
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y%m%d")

    output_filename = f"US_{variable}_20180101_20241231_Daily.csv"
    df.to_csv(output_filename, index=False)

    print(f"✅ {variable} data saved as {output_filename}")

print("✅ Data acquisition of all variables completed!")
