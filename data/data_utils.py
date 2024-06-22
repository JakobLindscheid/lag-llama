# Copyright 2024 Arjun Ashok
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import random
import requests
import warnings
import gzip, json
from pathlib import Path
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

import numpy as np
import pandas as pd
from tqdm import tqdm
from gluonts.dataset.common import ListDataset, TrainDatasets, MetaData
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.transform import InstanceSampler
from pandas.tseries.frequencies import to_offset

from gluonts.time_feature import TimeFeature
import os
from scipy.io import arff
from aeon.datasets import load_forecasting

from data.read_new_dataset import get_ett_dataset, create_train_dataset_without_last_k_timesteps, TrainDatasets, MetaData

class CombinedDatasetIterator:
    def __init__(self, datasets, seed, weights):
        self._datasets = [iter(el) for el in datasets]
        self._weights = weights
        self._rng = random.Random(seed)
        
    def __next__(self):
        (dataset,) = self._rng.choices(self._datasets, weights=self._weights, k=1)
        return next(dataset)


class CombinedDataset:
    def __init__(self, datasets, seed=None, weights=None):
        self._seed = seed
        self._datasets = datasets
        self._weights = weights
        n_datasets = len(datasets)
        if weights is None:
            self._weights = [1 / n_datasets] * n_datasets

    def __iter__(self):
        return CombinedDatasetIterator(self._datasets, self._seed, self._weights)

    def __len__(self):
        return sum([len(ds) for ds in self._datasets])


class SingleInstanceSampler(InstanceSampler):
    """
    Randomly pick a single valid window in the given time series.
    This fix the bias in ExpectedNumInstanceSampler which leads to varying sampling frequency
    of time series of unequal length, not only based on their length, but when they were sampled.
    """

    """End index of the history"""

    def __call__(self, ts: np.ndarray) -> np.ndarray:
        a, b = self._get_bounds(ts)
        window_size = b - a + 1
        if window_size <= 0:
            return np.array([], dtype=int)
        indices = np.random.randint(window_size, size=1)
        return indices + a


def _count_timesteps(
    left: pd.Timestamp, right: pd.Timestamp, delta: pd.DateOffset
) -> int:
    """
    Count how many timesteps there are between left and right, according to the given timesteps delta.
    If the number if not integer, round down.
    """
    # This is due to GluonTS replacing Timestamp by Period for version 0.10.0.
    # Original code was tested on version 0.9.4
    if type(left) == pd.Period:
        left = left.to_timestamp()
    if type(right) == pd.Period:
        right = right.to_timestamp()
    assert (
        right >= left
    ), f"Case where left ({left}) is after right ({right}) is not implemented in _count_timesteps()."
    try:
        return (right - left) // delta
    except TypeError:
        # For MonthEnd offsets, the division does not work, so we count months one by one.
        for i in range(10000):
            if left + (i + 1) * delta > right:
                return i
        else:
            raise RuntimeError(
                f"Too large difference between both timestamps ({left} and {right}) for _count_timesteps()."
            )

from pathlib import Path
from gluonts.dataset.common import ListDataset
from gluonts.dataset.repository.datasets import get_dataset

def create_train_dataset_last_k_percentage(
    raw_train_dataset,
    freq,
    k=100
):
    # Get training data
    train_data = []
    for i, series in enumerate(raw_train_dataset):
        s_train = series.copy()
        number_of_values = int(len(s_train["target"]) * k / 100)
        train_start_index = len(s_train["target"]) - number_of_values
        s_train["target"] = s_train["target"][train_start_index:]
        train_data.append(s_train)

    train_data = ListDataset(train_data, freq=freq)

    return train_data
    
def transform_data(data, metadata, dataset_name):
    # Transforms the data DataFrame and metadata into JSON format.

    output = {"data": [], "metadata": {}}

    # Define the mapping of frequency to its corresponding prediction length
    prediction_length_mapping = {
        "300Hz": 30, # 300 Hz -> prediction length =30
        "4S": 30,   # 4 seconds -> prediction length = 30
        "30T": 24,  # Half-hourly (30 minutes) -> prediction length = 24
        "1H": 24,   # Hourly -> prediction length = 24
        "1D": 30,   # Daily -> prediction length = 30
        "1W": 26,   # Weekly -> prediction length = 26
        "1M": 12,   # Monthly -> prediction length = 12
        "3M": 12,   # Quarterly -> prediction length = 12
        "1Y": 5     # Yearly -> prediction length = 5
    }

    frequency_mapping = {
        "4_seconds" : "4S",
        "half_hourly": "30T",
        "hourly": "1H",
        "daily": "1D",
        "weekly": "1W",
        "monthly": "1M",
        "quarterly": "3M",
        "yearly": "1Y"
    }
    freq = frequency_mapping.get(metadata.get("frequency", "hourly"))
    prediction_length = prediction_length_mapping.get(freq, 24)  # Default to 24 if not found

    output["metadata"] = {
        "freq": freq,
        "prediction_length": prediction_length
    }

    # Define the desired datetime format and default initial timestamps
    datetime_format = "%Y-%m-%d %H:%M:%S"
    initial_timestamp = "2000-01-01 00:00:00"  # Use a reasonable default value

    if dataset_name in ("cif_2016_dataset"):
      initial_timestamp = "2015-01-01 00:00:00"
    elif dataset_name in ("dominick_dataset"):
      initial_timestamp = "1989-09-01 00:00:00"

    if "start_timestamp" in data.columns:
      for index, row in data.iterrows():
        start_timestamp = row["start_timestamp"].strftime(datetime_format)
        series_value = list(row["series_value"])
        series_name = row["series_name"]

        data_item = {
            "start": start_timestamp,
            "target": series_value,
            "item_id": series_name
        }
        output["data"].append(data_item)
    else:
      # Use the fixed initial timestamp
      start_timestamp = initial_timestamp

      for index, row in data.iterrows():
        series_value = list(row["series_value"])
        series_name = row["series_name"]

        data_item = {
            "start": start_timestamp,
            "target": series_value,
            "item_id": series_name
        }
        output["data"].append(data_item)

    output_json = json.dumps(output, indent=4)
    return output_json
    
def download_file(url, local_path):
    response = requests.get(url)
    response.raise_for_status()  # Ensure we notice bad responses
    with open(local_path, 'wb') as file:
        file.write(response.content)

def load_abnormal_heartbeat():
    url_train = "https://drive.google.com/uc?export=download&id=1gYd80B0P1OzAebqXQYOOuLViBaM6gYBG" 
    url_test = "https://drive.google.com/uc?export=download&id=1U7Z_XGptYaFzkc4BoyLm8NGRXT3psA1W"
 
    local_path_train = "AbnormalHeartbeat_TRAIN.arff"
    local_path_test = "AbnormalHeartbeat_TEST.arff"
    
    download_file(url_train, local_path_train)
    download_file(url_test, local_path_test)
    
    train_data, train_meta = arff.loadarff(local_path_train)
    test_data, test_meta = arff.loadarff(local_path_test)


    df_train = pd.DataFrame(train_data)
    df_test = pd.DataFrame(test_data)

    start_date = "2016-01-09 00:00:00"  # Modify as needed

    train_series_list = []
    test_series_list = []

    for column in df_train.columns[:-1]:
        target_data = df_train[column].values

        train_series_entry = {
            "start": start_date,
            "target": target_data,
            "item_id": column
        }
        train_series_list.append(train_series_entry)
    train_ds = ListDataset(train_series_list, freq="1H")

    for column in df_test.columns[:-1]:
        target_data = df_test[column].values
        test_series_entry = {
            "start": start_date,
            "target": target_data,
            "item_id": column
        }
        test_series_list.append(test_series_entry)
    test_ds = ListDataset(test_series_list, freq="0.003S")
    metadata = MetaData(freq="300Hz", prediction_length=30)  # Adjust as needed

    ds = TrainDatasets(metadata=metadata, train=train_ds, test=test_ds)
    return ds
    
def create_train_and_val_datasets_with_dates(
    name,
    dataset_path,
    data_id,
    history_length,
    prediction_length=None,
    num_val_windows=None,
    val_start_date=None,
    train_start_date=None,
    freq=None,
    last_k_percentage=None
):
    """
    Train Start date is assumed to be the start of the series if not provided
    Freq is not given is inferred from the data
    We can use ListDataset to just group multiple time series - https://github.com/awslabs/gluonts/issues/695
    """

    if name in ("ett_h1", "ett_h2", "ett_m1", "ett_m2"):
        path = "datasets/ett_datasets"
        raw_dataset = get_ett_dataset(name, path)
    elif name in ("cpu_limit_minute", "cpu_usage_minute", \
                        "function_delay_minute", "instances_minute", \
                        "memory_limit_minute", "memory_usage_minute", \
                        "platform_delay_minute", "requests_minute"):
        path = "datasets/huawei/" + name + ".json"
        with open(path, "r") as f: data = json.load(f)
        metadata = MetaData(**data["metadata"])
        train_data = [x for x in data["train"] if type(x["target"][0]) != str]
        test_data = [x for x in data["test"] if type(x["target"][0]) != str]
        train_ds = ListDataset(train_data, freq=metadata.freq)
        test_ds = ListDataset(test_data, freq=metadata.freq)
        raw_dataset = TrainDatasets(metadata=metadata, train=train_ds, test=test_ds)
    elif name in ("beijing_pm25", "AirQualityUCI", "beijing_multisite"):
        path = "datasets/air_quality/" + name + ".json"
        with open(path, "r") as f:
            data = json.load(f)
        metadata = MetaData(**data["metadata"])
        train_test_data = [x for x in data["data"] if type(x["target"][0]) != str]
        full_dataset = ListDataset(train_test_data, freq=metadata.freq)
        train_ds = create_train_dataset_without_last_k_timesteps(full_dataset, freq=metadata.freq, k=24)
        raw_dataset = TrainDatasets(metadata=metadata, train=train_ds, test=full_dataset)
    elif name in ('car_parts_dataset_without_missing_values', 'cif_2016_dataset',
                  'covid_deaths_dataset', 'covid_mobility_dataset_without_missing_values', 
                  'dominick_dataset', 'fred_md_dataset', 'hospital_dataset',
                  'kaggle_web_traffic_dataset_without_missing_values',
                  'kaggle_web_traffic_weekly_dataset', 'm1_monthly_dataset',
                  'm1_quarterly_dataset', 'm1_yearly_dataset', 'm3_monthly_dataset',
                  'm3_quarterly_dataset', 'm3_yearly_dataset', 'm4_daily_dataset',
                  'm4_hourly_dataset',  'm4_monthly_dataset', 'm4_quarterly_dataset',
                  'm4_weekly_dataset', 'm4_yearly_dataset',
                  'nn5_daily_dataset_without_missing_values', 'nn5_weekly_dataset',
                  'solar_4_seconds_dataset', 'solar_weekly_dataset', 'tourism_monthly_dataset',
                  'tourism_quarterly_dataset', 'tourism_yearly_dataset', 'traffic_weekly_dataset',
                  'us_births_dataset', 'wind_4_seconds_dataset'):
        data, metadata = load_forecasting(name, return_metadata=True)
        output_json = transform_data(data, metadata, name)
        output_dict = json.loads(output_json)
        data = output_dict.get("data", [])
        metadata = MetaData(**output_dict.get("metadata", {}))
        train_test_data = [x for x in data if type(x["target"][0]) != str]
        full_dataset = ListDataset(train_test_data, freq=metadata.freq)
        train_ds = create_train_dataset_without_last_k_timesteps(full_dataset, freq=metadata.freq, k=metadata.prediction_length)
        raw_dataset = TrainDatasets(metadata=metadata, train=train_ds, test=full_dataset)
    elif name in ('AbnormalHeartbeat'):
        raw_dataset = load_abnormal_heartbeat()
    else:
        raw_dataset = get_dataset(name, path=dataset_path)

    if prediction_length is None:
        prediction_length = raw_dataset.metadata.prediction_length
    if freq is None:
        freq = raw_dataset.metadata.freq
    timestep_delta = pd.tseries.frequencies.to_offset(freq)
    raw_train_dataset = raw_dataset.train

    if not num_val_windows and not val_start_date:
        raise Exception("Either num_val_windows or val_start_date must be provided")
    if num_val_windows and val_start_date:
        raise Exception("Either num_val_windows or val_start_date must be provided")

    max_train_end_date = None

    # Get training data
    total_train_points = 0
    train_data = []
    for i, series in enumerate(raw_train_dataset):
        s_train = series.copy()
        if val_start_date is not None:
            train_end_index = _count_timesteps(
                series["start"] if not train_start_date else train_start_date,
                val_start_date,
                timestep_delta,
            )
        else:
            train_end_index = len(series["target"]) - num_val_windows
        # Compute train_start_index based on last_k_percentage
        if last_k_percentage:
            number_of_values = int(len(s_train["target"]) * last_k_percentage / 100)
            train_start_index = max(train_end_index - number_of_values, 0) # NOTE: hotfix for 100% fine tuning to not change few-shot results
        else:
            train_start_index = 0
        s_train["target"] = series["target"][train_start_index:train_end_index]
        s_train["item_id"] = i
        s_train["data_id"] = data_id
        train_data.append(s_train)
        total_train_points += len(s_train["target"])

        # Calculate the end date
        end_date = s_train["start"] + to_offset(freq) * (len(s_train["target"]) - 1)
        if max_train_end_date is None or end_date > max_train_end_date:
            max_train_end_date = end_date

    train_data = ListDataset(train_data, freq=freq)

    # Get validation data
    total_val_points = 0
    total_val_windows = 0
    val_data = []
    for i, series in enumerate(raw_train_dataset):
        s_val = series.copy()
        if val_start_date is not None:
            train_end_index = _count_timesteps(
                series["start"], val_start_date, timestep_delta
            )
        else:
            train_end_index = len(series["target"]) - num_val_windows
        val_start_index = train_end_index - prediction_length - history_length
        s_val["start"] = series["start"] + val_start_index * timestep_delta
        s_val["target"] = series["target"][val_start_index:]
        s_val["item_id"] = i
        s_val["data_id"] = data_id
        val_data.append(s_val)
        total_val_points += len(s_val["target"])
        total_val_windows += len(s_val["target"]) - prediction_length - history_length
    val_data = ListDataset(val_data, freq=freq)

    total_points = (
        total_train_points
        + total_val_points
        - (len(raw_train_dataset) * (prediction_length + history_length))
    )

    return (
        train_data,
        val_data,
        total_train_points,
        total_val_points,
        total_val_windows,
        max_train_end_date,
        total_points,
    )


def create_test_dataset(
    name, dataset_path, history_length, freq=None, data_id=None
):
    """
    For now, only window per series is used.
    make_evaluation_predictions automatically only predicts for the last "prediction_length" timesteps
    NOTE / TODO: For datasets where the test set has more series (possibly due to more timestamps), \
    we should check if we only use the last N series where N = series per single timestamp, or if we should do something else.
    """

    if name in ("ett_h1", "ett_h2", "ett_m1", "ett_m2"):
        path = "datasets/ett_datasets"
        dataset = get_ett_dataset(name, path)
    elif name in ("cpu_limit_minute", "cpu_usage_minute", \
                        "function_delay_minute", "instances_minute", \
                        "memory_limit_minute", "memory_usage_minute", \
                        "platform_delay_minute", "requests_minute"):
        path = "datasets/huawei/" + name + ".json"
        with open(path, "r") as f: data = json.load(f)
        metadata = MetaData(**data["metadata"])
        train_data = [x for x in data["train"] if type(x["target"][0]) != str]
        test_data = [x for x in data["test"] if type(x["target"][0]) != str]
        train_ds = ListDataset(train_data, freq=metadata.freq)
        test_ds = ListDataset(test_data, freq=metadata.freq)
        dataset = TrainDatasets(metadata=metadata, train=train_ds, test=test_ds)
    elif name in ("beijing_pm25", "AirQualityUCI", "beijing_multisite"):
        path = "datasets/air_quality/" + name + ".json"
        with open(path, "r") as f:
            data = json.load(f)
        metadata = MetaData(**data["metadata"])
        train_test_data = [x for x in data["data"] if type(x["target"][0]) != str]
        full_dataset = ListDataset(train_test_data, freq=metadata.freq)
        train_ds = create_train_dataset_without_last_k_timesteps(full_dataset, freq=metadata.freq, k=24)
        dataset = TrainDatasets(metadata=metadata, train=train_ds, test=full_dataset)
    elif name in ('car_parts_dataset_without_missing_values', 'cif_2016_dataset',
                  'covid_deaths_dataset', 'covid_mobility_dataset_without_missing_values', 
                  'dominick_dataset', 'fred_md_dataset', 'hospital_dataset',
                  'kaggle_web_traffic_dataset_without_missing_values',
                  'kaggle_web_traffic_weekly_dataset', 'm1_monthly_dataset',
                  'm1_quarterly_dataset', 'm1_yearly_dataset', 'm3_monthly_dataset',
                  'm3_quarterly_dataset', 'm3_yearly_dataset', 'm4_daily_dataset',
                  'm4_hourly_dataset',  'm4_monthly_dataset', 'm4_quarterly_dataset',
                  'm4_weekly_dataset', 'm4_yearly_dataset',
                  'nn5_daily_dataset_without_missing_values', 'nn5_weekly_dataset',
                  'solar_4_seconds_dataset', 'solar_weekly_dataset', 'tourism_monthly_dataset',
                  'tourism_quarterly_dataset', 'tourism_yearly_dataset', 'traffic_weekly_dataset',
                  'us_births_dataset', 'wind_4_seconds_dataset'):
        data, metadata = load_forecasting(name, return_metadata=True)
        output_json = transform_data(data, metadata, name)
        output_dict = json.loads(output_json)
        data = output_dict.get("data", [])
        metadata = MetaData(**output_dict.get("metadata", {}))
        train_test_data = [x for x in data if type(x["target"][0]) != str]
        full_dataset = ListDataset(train_test_data, freq=metadata.freq)
        train_ds = create_train_dataset_without_last_k_timesteps(full_dataset, freq=metadata.freq, k=metadata.prediction_length)
        dataset = TrainDatasets(metadata=metadata, train=train_ds, test=full_dataset)
    elif name in ('AbnormalHeartbeat'):
        dataset = load_abnormal_heartbeat()
    else:
        dataset = get_dataset(name, path=dataset_path)

    if freq is None:
        freq = dataset.metadata.freq
    prediction_length = dataset.metadata.prediction_length
    data = []
    total_points = 0
    for i, series in enumerate(dataset.test):
        offset = len(series["target"]) - (history_length + prediction_length)
        if offset > 0:
            target = series["target"][-(history_length + prediction_length) :]
            data.append(
                {
                    "target": target,
                    "start": series["start"] + offset,
                    "item_id": i,
                    "data_id": data_id,
                }
            )
        else:
            series_copy = copy.deepcopy(series)
            series_copy["item_id"] = i
            series_copy["data_id"] = data_id
            data.append(series_copy)
        total_points += len(data[-1]["target"])
    return ListDataset(data, freq=freq), prediction_length, total_points
