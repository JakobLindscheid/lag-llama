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

ALL_DATASETS = [ #27
    "australian_electricity_demand", 
    "electricity_hourly", 
    "london_smart_meters_without_missing", 
    "solar_10_minutes", 
    "wind_farms_without_missing", 
    "pedestrian_counts", 
    "uber_tlc_hourly", 
    "traffic", 
    "kdd_cup_2018_without_missing", 
    "saugeenday", 
    "sunspot_without_missing", 
    "exchange_rate", 
    "cpu_limit_minute", 
    "cpu_usage_minute", 
    "function_delay_minute", 
    "instances_minute", 
    "memory_limit_minute", 
    "memory_usage_minute", 
    "platform_delay_minute", 
    "requests_minute", 
    "ett_h1", 
    "ett_h2", 
    "ett_m1", 
    "ett_m2", 
    "beijing_pm25", 
    "AirQualityUCI", 
    "beijing_multisite"
]

ALL_DATASETS_NEW_REPOSITORY = [ #34 (Tot: 61)
    'AbnormalHeartbeat',                                     # 300 Hz       # Health 
    'car_parts_dataset_without_missing_values',              # Monthly,     # Sales      equal length
    'cif_2016_dataset',                                      # Monthly,     # Banking      
    'covid_deaths_dataset',                                  # Daily,       # Nature     equal length
    'covid_mobility_dataset_without_missing_values',         # Daily,       # Nature
    'dominick_dataset',                                      # Weekly,      # Sales 
    'elecdemand_dataset',                                    # half_hourly, # Energy     equal length
    'fred_md_dataset',                                       # Monthly,     # Economic   equal length
    'hospital_dataset',                                      # Monthly,     # Health     equal length
    'kaggle_web_traffic_dataset_without_missing_values',     # Daily,       # Web        equal length
    'kaggle_web_traffic_weekly_dataset',                     # Weekly,      # Web        equal length
    'm1_monthly_dataset',                                    # Monthly,     # Multiple
    'm1_quarterly_dataset',                                  # Quarterly,   # Multiple
    'm1_yearly_dataset',                                     # Yearly,      # Multiple
    'm3_monthly_dataset',                                    # Monthly,     # Multiple
    'm3_quarterly_dataset',                                  # Quarterly,   # Multiple
    'm3_yearly_dataset',                                     # Yearly,      # Multiple
    'm4_daily_dataset',                                      # Daily,       # Multiple
    'm4_hourly_dataset',                                     # Hourly,      # Multiple
    'm4_monthly_dataset',                                    # Monthly,     # Multiple
    'm4_quarterly_dataset',                                  # Quarterly,   # Multiple
    'm4_weekly_dataset',                                     # Weekly,      # Multiple
    'm4_yearly_dataset',                                     # Yearly,      # Multiple
    'nn5_daily_dataset_without_missing_values',              # Daily,       # Banking    equal length
    'nn5_weekly_dataset',                                    # Weekly,      # Banking    equal length
    'solar_4_seconds_dataset',                               # 4_seconds,   # Energy     equal length
    'solar_weekly_dataset',                                  # Weekly,      # Energy     equal length
    'tourism_monthly_dataset',                               # Monthly,     # Tourism
    'tourism_quarterly_dataset',                             # Quarterly,   # Tourism
    'tourism_yearly_dataset',                                # Yearly,      # Tourism
    'traffic_weekly_dataset',                                # Weekly,      # Transport  equal length
    'us_births_dataset',                                     # Daily,       # Nature     equal length
    'weather_dataset',                                       # Daily,       # Nature
    'wind_4_seconds_dataset',                                # 4_seconds,   # Energy     equal length
]
