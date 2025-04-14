
#name:Huang RuiQi, Yang Tan

import os
import pandas as pd
import polars as pl
from torch.utils.data import Dataset, DataLoader
import pymysql
import time
from numpy.lib.stride_tricks import sliding_window_view
import pickle
import torch
import json
from sqlalchemy import create_engine


user = 'g6_qf5214_3'
password = 'qf5214_3'
host = 'rm-zf8608n9yz8lywnj1go.mysql.kualalumpur.rds.aliyuncs.com'
port = 3306
db = 'nasa_power'


class Power_Dataset(Dataset):
    def __init__(self, samples, seq_len=30):
        self.samples = samples
        self.seq_len = seq_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return {
            'input_seq': torch.tensor(item['input_seq'], dtype=torch.float32).unsqueeze(-1),
            'target': torch.tensor(item['target'], dtype=torch.float32),
            'region_id': torch.tensor(item['region_id'], dtype=torch.long),
            'indicator_id': torch.tensor(item['indicator_id'], dtype=torch.long)
        }


def indicator_mapping_dict():
    mapping_dict = {'GWETTOP': 0, 'T10M_MIN': 1, 'V10M': 2, 'TS_MAX': 3, 'V50M': 4, 'V2M': 5,
                    'PBLTOP': 6, 'WS50M': 7, 'WS50M_RANGE': 8, 'T10M': 9, 'U2M': 10,
                    'WS2M_MIN': 11, 'TO3': 12, 'TROPT': 13, 'TS': 14, 'T2M': 15,
                    'T2M_RANGE': 16, 'QV2M': 17, 'WD2M': 18,'QV10M': 19, 'T2MDEW': 20,
                    'WS10M': 21, 'RHOA': 22, 'TS_RANGE': 23, 'U10M': 24, 'WD50M': 25,
                    'TROPPB': 26, 'WS2M_RANGE': 27, 'RH2M': 28, 'T2MWET': 29, 'U50M': 30,
                    'WS10M_MAX': 31, 'T10M_RANGE': 32, 'TS_MIN': 33, 'WS2M_MAX': 34, 'PW': 35,
                    'TQV': 36, 'PSH': 37, 'T10M_MAX': 38, 'T2M_MAX': 39, 'TROPQ': 40,
                    'WS2M': 41, 'WS50M_MIN': 42, 'T2M_MIN': 43, 'WD10M': 44, 'SLP': 45,
                    'TS_ADJ': 46, 'ALLSKY_SFC_PAR_DIFF': 47, 'CLRSKY_KT': 48, 'ORIGINAL_ALLSKY_SFC_SW_DIFF': 49, 'MIDDAY_INSOL': 50,
                    'AOD_55': 51, 'ALLSKY_SFC_SW_DWN': 52, 'CLRSKY_SFC_PAR_DIRH': 53, 'CLOUD_AMT_NIGHT': 54, 'CLRSKY_SFC_SW_UP': 55,
                    'CLRSKY_NKT': 56, 'CLRSKY_SFC_LW_UP': 57, 'ALLSKY_SFC_SW_UP': 58, 'CLRSKY_SFC_SW_DIFF': 59, 'ALLSKY_SFC_SW_DIFF': 60,
                    'TOA_SW_DWN': 61, 'ALLSKY_SFC_SW_DIRH': 62, 'CLRSKY_SRF_ALB': 63, 'ALLSKY_SRF_ALB': 64, 'CLOUD_AMT': 65,
                    'AOD_84': 66, 'SRF_ALB_ADJ': 67, 'ALLSKY_SFC_PAR_DIRH': 68, 'ALLSKY_SFC_SW_DNI': 69, 'ALLSKY_SFC_PAR_TOT': 70,
                    'ALLSKY_SFC_UVA': 71, 'ALLSKY_SFC_LW_DWN': 72, 'ALLSKY_KT': 73, 'ORIGINAL_ALLSKY_SFC_SW_DIRH': 74, 'ALLSKY_SFC_LW_UP': 75,
                    'CLRSKY_SFC_PAR_TOT': 76, 'ALLSKY_SFC_UV_INDEX': 77, 'AIRMASS': 78, 'CLRSKY_SFC_SW_DIRH': 79, 'CLRSKY_SFC_SW_DNI': 80,
                    'TOA_SW_DNI': 81, 'CLOUD_OD': 82, 'CLOUD_AMT_DAY': 83, 'CLRSKY_SFC_LW_DWN': 84, 'ALLSKY_NKT': 85,
                    'CDD0': 86, 'CLRSKY_SFC_PAR_DIFF': 87, 'CLRSKY_SFC_SW_DWN': 88, 'AOD_55_ADJ': 89}
    return mapping_dict


def preprocess_polars(df):
    exprs = []

    schema = df.collect_schema() if isinstance(df, pl.LazyFrame) else df.schema


    if schema.get("Timestamp") == pl.Utf8:
        exprs.append(
            pl.col("Timestamp").str.strptime(pl.Date, format="%Y-%m-%d")
        )

    exprs += [
        pl.col("Latitude").cast(pl.Int32),
        pl.col("Longitude").cast(pl.Int32),
    ]


    df = df.with_columns(exprs).sort("Timestamp")

    if isinstance(df, pl.LazyFrame):
        col_names = df.collect_schema().names()
    else:
        col_names = df.columns

    #on = [col for col in col_names if col not in ['Timestamp', 'Latitude', 'Longitude']]

    df = df.unpivot(
        index=['Timestamp', 'Latitude', 'Longitude'],
        on=[col for col in col_names if col not in ['Timestamp', 'Latitude', 'Longitude']],
        variable_name='indicator_id',
        value_name='value'
    )

    # mapping indicator
    mapping_dict = indicator_mapping_dict()
    df = df.with_columns([
        pl.col('indicator_id').replace(mapping_dict).cast(pl.Int32)
    ])

    # mapping region id
    min_latitude = 24
    min_longitude = -125
    max_longitude = -66

    longitude_range = max_longitude - min_longitude + 1

    df = (df.with_columns([
        ((pl.col("Latitude") - min_latitude)*longitude_range + (pl.col("Longitude") - min_longitude))
        .cast(pl.Int32)
        .alias("region_id")
    ]).drop(["Latitude", "Longitude"]).sort("region_id", "indicator_id", "Timestamp"))

    # ---------- 0-1 Normalization value ----------
    with open("region_minmax_dict.json") as f:
        raw_dict = json.load(f)

    rows = []
    for region_id, indicators in raw_dict.items():
        for indicator, stats in indicators.items():
            if indicator in mapping_dict:
                rows.append({
                    "region_id": int(region_id),
                    "indicator_id": mapping_dict[indicator],  # 💡 映射为 Int
                    "min": stats["min_value"],
                    "max": stats["max_value"]
                })

    #minmax_df = pl.DataFrame(rows)
    minmax_df = pl.DataFrame(rows).lazy()

    # 2. join + 标准化
    df = df.join(minmax_df, on=["region_id", "indicator_id"], how="left")

    df = df.with_columns([
        ((pl.col("value") - pl.col("min")) /
         (pl.col("max") - pl.col("min")).replace(0, 1e-6)).alias("value")
    ]).drop(["min", "max"])

    return df





def generate_sliding_window_samples(df: pl.DataFrame, seq_len: int = 30):
    input_seqs = []
    targets = []
    region_ids = []
    indicator_ids = []

    group_keys = df.select(["region_id", "indicator_id"]).unique().to_numpy()

    for region_id, indicator_id in group_keys:
        group_df = df.filter(
            (pl.col("region_id") == region_id) &
            (pl.col("indicator_id") == indicator_id)
        )

        values = group_df["value"].to_numpy()
        if len(values) <= seq_len:
            continue

        x = sliding_window_view(values, window_shape=seq_len)
        y = values[seq_len:]

        input_seqs.extend(x.tolist())
        targets.extend(y.tolist())
        region_ids.extend([int(region_id)] * len(y))
        indicator_ids.extend([int(indicator_id)] * len(y))

    samples = [
        {
            "input_seq": input_seqs[i],
            "target": targets[i],
            "region_id": region_ids[i],
            "indicator_id": indicator_ids[i]
        }
        for i in range(len(targets))
    ]
    #print(f"[Check] sample 0: {samples[0]}")
    #print(type(samples[0]))

    return samples

def indicator_table_dict():
    result = {'GWETTOP': 'meteorology', 'T10M_MIN': 'meteorology', 'V10M': 'meteorology',
              'TS_MAX': 'meteorology', 'V50M': 'meteorology', 'V2M': 'meteorology', 'PBLTOP': 'meteorology',
              'WS50M': 'meteorology', 'WS50M_RANGE': 'meteorology', 'T10M': 'meteorology', 'U2M': 'meteorology',
              'WS2M_MIN': 'meteorology', 'TO3': 'meteorology', 'TROPT': 'meteorology', 'TS': 'meteorology',
              'T2M': 'meteorology', 'T2M_RANGE': 'meteorology', 'QV2M': 'meteorology', 'WD2M': 'meteorology',
              'QV10M': 'meteorology', 'T2MDEW': 'meteorology', 'WS10M': 'meteorology', 'RHOA': 'meteorology',
              'TS_RANGE': 'meteorology', 'U10M': 'meteorology', 'WD50M': 'meteorology', 'TROPPB': 'meteorology',
              'WS2M_RANGE': 'meteorology', 'RH2M': 'meteorology', 'T2MWET': 'meteorology', 'U50M': 'meteorology',
              'WS10M_MAX': 'meteorology', 'T10M_RANGE': 'meteorology', 'TS_MIN': 'meteorology', 'WS2M_MAX': 'meteorology',
              'PW': 'meteorology', 'TQV': 'meteorology', 'PSH': 'meteorology', 'T10M_MAX': 'meteorology',
              'T2M_MAX': 'meteorology', 'TROPQ': 'meteorology', 'WS2M': 'meteorology', 'WS50M_MIN': 'meteorology',
              'T2M_MIN': 'meteorology', 'WD10M': 'meteorology', 'SLP': 'meteorology', 'TS_ADJ': 'meteorology',
              'ALLSKY_SFC_PAR_DIFF': 'radiation', 'CLRSKY_KT': 'radiation', 'ORIGINAL_ALLSKY_SFC_SW_DIFF': 'radiation',
              'MIDDAY_INSOL': 'radiation', 'AOD_55': 'radiation', 'ALLSKY_SFC_SW_DWN': 'radiation', 'CLRSKY_SFC_PAR_DIRH': 'radiation',
              'CLOUD_AMT_NIGHT': 'radiation', 'CLRSKY_SFC_SW_UP': 'radiation', 'CLRSKY_NKT': 'radiation',
              'CLRSKY_SFC_LW_UP': 'radiation', 'ALLSKY_SFC_SW_UP': 'radiation', 'CLRSKY_SFC_SW_DIFF': 'radiation',
              'ALLSKY_SFC_SW_DIFF': 'radiation', 'TOA_SW_DWN': 'radiation', 'ALLSKY_SFC_SW_DIRH': 'radiation',
              'CLRSKY_SRF_ALB': 'radiation', 'ALLSKY_SRF_ALB': 'radiation', 'CLOUD_AMT': 'radiation', 'AOD_84': 'radiation',
              'SRF_ALB_ADJ': 'radiation', 'ALLSKY_SFC_PAR_DIRH': 'radiation', 'ALLSKY_SFC_SW_DNI': 'radiation',
              'ALLSKY_SFC_PAR_TOT': 'radiation', 'ALLSKY_SFC_UVA': 'radiation', 'ALLSKY_SFC_LW_DWN': 'radiation',
              'ALLSKY_KT': 'radiation', 'ORIGINAL_ALLSKY_SFC_SW_DIRH': 'radiation', 'ALLSKY_SFC_LW_UP': 'radiation',
              'CLRSKY_SFC_PAR_TOT': 'radiation', 'ALLSKY_SFC_UV_INDEX': 'radiation', 'AIRMASS': 'radiation',
              'CLRSKY_SFC_SW_DIRH': 'radiation', 'CLRSKY_SFC_SW_DNI': 'radiation', 'TOA_SW_DNI': 'radiation',
              'CLOUD_OD': 'radiation', 'CLOUD_AMT_DAY': 'radiation', 'CLRSKY_SFC_LW_DWN': 'radiation',
              'ALLSKY_NKT': 'radiation', 'CDD0': 'radiation', 'CLRSKY_SFC_PAR_DIFF': 'radiation',
              'CLRSKY_SFC_SW_DWN': 'radiation', 'AOD_55_ADJ': 'radiation'}
    return result

class Mysql_Data_Source:
    def __init__(self, chunk_size=500_000):
        self.chunk_size = chunk_size
        self.offset_dict = {}

    def __iter__(self):
        engine = create_engine(f'mysql+pymysql://{user}:{password}@{host}:{port}/{db}?charset=utf8mb4')
        sampling_latitude = [24, 29, 34, 39, 44, 49]
        #sampling_latitude = [24,]
        sampling_longitude = [-125, -119, -113, -107, -101, -95, -89, -83, -77, -71, -66]
        #sampling_longitude = [-125]
        lat_values = ", ".join(str(lat) for lat in sampling_latitude)
        lon_values = ", ".join(str(lon) for lon in sampling_longitude)
        indicator_table_mapping_dict = indicator_table_dict()

        # for test
        # indicator_table_mapping_dict = {'GWETTOP': 'meteorology',}
        while True:
            all_empty = True

            for col, table in indicator_table_mapping_dict.items():
                offset = self.offset_dict.get((col, table), 0)

                query = f"""
                            SELECT Timestamp, Latitude, Longitude, {col} FROM {table}
                            WHERE Timestamp >= '2024-11-01'
                            AND Latitude IN ({lat_values})
                            AND Longitude IN ({lon_values})
                            ORDER BY Timestamp
                            LIMIT {self.chunk_size} OFFSET {offset}
                        """

                '''
                
                query = f"""
                                            SELECT Timestamp, Latitude, Longitude, {col} FROM {table}
                                            WHERE Timestamp >= '2024-11-01'
                                            ORDER BY Timestamp
                                            LIMIT {self.chunk_size} OFFSET {offset}
                                        """
                '''


                df = pd.read_sql(query, con=engine)
                #print(df)

                if not df.empty:
                    all_empty = False
                    df = df.rename(columns={col: str(col)})
                    self.offset_dict[(col, table)] = offset + self.chunk_size
                    yield df

            if all_empty:
                break

class CSV_Data_Source:
    def __init__(self, data_dir="./Data", file_ext='.csv', shuffle=True):
        self.file_list = sorted([
            os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(file_ext)
        ])
        if shuffle:
            import random
            random.shuffle(self.file_list)

    def __iter__(self):
        for file_path in self.file_list:
            print(f"Loading CSV file: {os.path.basename(file_path)}")
            df = pl.read_csv(file_path)
            yield df

class Power_Dataset_Pipeline:
    def __init__(self, data_source, batch_size=64, seq_len=30):
        self.data_source = data_source
        self.batch_size = batch_size
        self.seq_len = seq_len

    def __iter__(self):
        for df_chunk in self.data_source:
            df_chunk = pl.from_pandas(df_chunk).lazy()
            preprocessed_df_chunk = preprocess_polars(df_chunk).collect()
            samples = generate_sliding_window_samples(preprocessed_df_chunk, seq_len=self.seq_len)
            if len(samples) == 0:
                continue
            dataset = Power_Dataset(samples, seq_len=self.seq_len)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            yield loader

class Test:
    @staticmethod # done
    def test_csv_data_source():
        data_dir = "./Data"
        data_source = CSV_Data_Source(data_dir=data_dir)
        for i in data_source:
            print(i)
            break

    @staticmethod # done
    def test_data_preprocess():
        data_dir = "./Data"
        data_source = CSV_Data_Source(data_dir=data_dir)
        for df in data_source:
            #print(df)
            preprocessed_df = preprocess_polars(df)
            print(preprocessed_df)
            break

    @staticmethod  # done
    def test_sample_generate():
        data_dir = "./Data"
        data_source = CSV_Data_Source(data_dir=data_dir)
        for df in data_source:
            # print(df)
            preprocessed_df = preprocess_polars(df)

            start_time = time.time()
            samples_serial = generate_sliding_window_samples(preprocessed_df, seq_len=30)
            end_time = time.time()
            print(f"[Serial] Time taken: {end_time - start_time:.2f} seconds | Samples: {len(samples_serial)}")

            with open("./Data/sample_data_demo.pkl", "wb") as f:
                pickle.dump(samples_serial, f)

            print(f"Saved {len(samples_serial)} samples to './Data/samples_serial.pt'")

            break


    @staticmethod
    def test_load_demo_sample():
        start_time = time.time()
        with open("./Data/sample_data_demo.pkl", "rb") as f:
            samples = pickle.load(f)

        end_time = time.time()

        #print(f"✅ Loaded {len(samples)} samples from 'sample_data_demo.pkl'")
        print(f"[Serial] Time taken: {end_time - start_time:.2f} seconds | Samples: {len(samples)}")

        return samples

    @staticmethod
    def test_load_dataset():
        with open("./Data/sample_data_demo.pkl", "rb") as f:
            samples = pickle.load(f)

        dataset = Power_Dataset(samples=samples)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        print(dataset)
        return loader

    @staticmethod
    def test_data_pipeline():
        data_source = CSV_Data_Source()
        data_pipeline = Power_Dataset_Pipeline(data_source=data_source)
        for i in data_pipeline:
            print(i)



def load_test_pipeline():
    with open("./Data/sample_data_demo.pkl", "rb") as f:
        samples = pickle.load(f)

    samples = samples[:int(0.01 * len(samples))]
    split = int(0.8 * len(samples))
    train_samples = samples[:split]
    val_samples = samples[split:]

    train_dataset = Power_Dataset(samples=train_samples,seq_len=30)
    val_dataset = Power_Dataset(samples=val_samples,seq_len=30)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    return train_loader, val_loader


def load_pipeline():
    print("Loading Data...")
    data_source = Mysql_Data_Source()
    pipeline = Power_Dataset_Pipeline(data_source=data_source, batch_size=128, seq_len=30)
    pipeline_iter = iter(pipeline)


if __name__ == '__main__':
    load_pipeline()


