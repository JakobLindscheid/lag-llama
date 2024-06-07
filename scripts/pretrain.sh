# PLEASE FOLLOW THE BELOW INSTRUCTIONS FIRST

# 1. Install the requirements. 
# It is recommend to use a new Anaconda environment with Python 3.10.8. Execute the below command (remove the #)
# Mamba integration has been tested with Python 3.11.
# !pip install -r requirements.txt

# 2. Please download https://drive.google.com/file/d/1JrDWMZyoPsc6d1wAAjgm3PosbGus-jCE/view?usp=sharing and use the below command to download the non-monash datasets (remove the #)
# tar -xvzf nonmonash_datasets.tar.gz -C datasets

mkdir -p experiments
mkdir -p experiments/seeds
mkdir -p experiments/results

EXP_NAME="new_datasets_m"
FILENAME="experiments/seeds/${EXP_NAME}"
CONFIGPATH="configs/lag_llama.json"

echo $EXP_NAME

# NUM_SEEDS used only if it is a new experiment
NUM_SEEDS=1

# Create seeds
if [ -f $FILENAME ]; then
    echo "${FILENAME} already exists."

    SEEDS=()
    while read -r LINE; do
        SEEDS+=("$LINE")
    done < $FILENAME
    echo "Found ${#SEEDS[@]} seeds for training."
else
    # Write seeds
    echo "${FILENAME} created. Writing seeds."
    touch $FILENAME
    for (( i = 0; i < $NUM_SEEDS; i++ )) 
    do 
        SEED=$((RANDOM + 1))
        echo $SEED >> $FILENAME
    done

    # Read them
    SEEDS=()
    while read -r LINE; do
        SEEDS+=("$LINE")
    done < $FILENAME
fi

# Train
for SEED in "${SEEDS[@]}"
do
    EXPERIMENT_NAME="${EXP_NAME}_seed_${SEED}"

    python run.py \
    -e $EXP_NAME -d "datasets" --seed $SEED \
    -r "experiments/results" \
    --batch_size 512 -m 1000 -n 128 \
    --wandb_entity "maranc" --wandb_project "lag-llama" \
    --all_datasets "australian_electricity_demand" "electricity_hourly" "london_smart_meters_without_missing" "solar_10_minutes" "wind_farms_without_missing" "pedestrian_counts" "uber_tlc_hourly" "traffic" "kdd_cup_2018_without_missing" "saugeenday" "sunspot_without_missing" "exchange_rate" "cpu_limit_minute" "cpu_usage_minute" "function_delay_minute" "instances_minute" "memory_limit_minute" "memory_usage_minute" "platform_delay_minute" "requests_minute" "ett_h1" "ett_h2" "ett_m1" "ett_m2" "beijing_pm25" "AirQualityUCI" "beijing_multisite" "weather" "tourism_yearly_dataset" "us_births_dataset" "m4_yearly_dataset" "dominick_dataset" "hospital_dataset" "kaggle_web_traffic_dataset_without_missing_values" "kaggle_web_traffic_weekly_dataset" "traffic_weekly_dataset" "tourism_monthly_dataset" "tourism_quarterly_dataset" "elecdemand_dataset" "wind_4_seconds_dataset" "solar_4_seconds_dataset" "solar_weekly_dataset" "covid_deaths_dataset" "covid_mobility_dataset_without_missing_values" "cif_2016_dataset" "fred_md_dataset" "nn5_daily_dataset_without_missing_values" "nn5_weekly_dataset" "m1_monthly_dataset" "m1_quarterly_dataset" "m1_yearly_dataset" "m3_monthly_dataset" "m3_quarterly_dataset" "m3_yearly_dataset" "m4_hourly_dataset" "m4_daily_dataset" "m4_weekly_dataset" "m4_monthly_dataset" "m4_quarterly_dataset" "car_parts_dataset_without_missing_values"\
    --test_datasets "weather" "pedestrian_counts" "exchange_rate" "ett_m2" "platform_delay_minute" "requests_minute" "beijing_pm25" "tourism_yearly_dataset" "us_births_dataset" "m4_yearly_dataset" "dominick_dataset" "hospital_dataset" "kaggle_web_traffic_dataset_without_missing_values" "kaggle_web_traffic_weekly_dataset"\
    --num_workers 2 --args_from_dict_path $CONFIGPATH --search_batch_size \
    --lr 0.0001
done

#    --wandb_entity "jakobl" --wandb_project "lag-llama" \
#    --all_datasets "australian_electricity_demand" "electricity_hourly" "london_smart_meters_without_missing" "solar_10_minutes" "wind_farms_without_missing" "pedestrian_counts" "uber_tlc_hourly" "traffic" "kdd_cup_2018_without_missing" "saugeenday" "sunspot_without_missing" "exchange_rate" "cpu_limit_minute" "cpu_usage_minute" "function_delay_minute" "instances_minute" "memory_limit_minute" "memory_usage_minute" "platform_delay_minute" "requests_minute" "ett_h1" "ett_h2" "ett_m1" "ett_m2" "beijing_pm25" "AirQualityUCI" "beijing_multisite" "weather" \
#    --test_datasets "weather" "pedestrian_counts" "exchange_rate" "ett_m2" "platform_delay_minute" "requests_minute" "beijing_pm25" \

#"traffic_weekly_dataset", "tourism_monthly_dataset", "tourism_quarterly_dataset", "elecdemand_dataset", "wind_4_seconds_dataset", "solar_4_seconds_dataset", "solar_weekly_dataset", "covid_deaths_dataset", "covid_mobility_dataset_without_missing_values "cif_2016_dataset", "fred_md_dataset", "nn5_daily_dataset_without_missing_values", "nn5_weekly_dataset", "m1_monthly_dataset", "m1_quarterly_dataset", "m1_yearly_dataset", "m3_monthly_dataset", "m3_quarterly_dataset", "m3_yearly_dataset", "m4_hourly_dataset", "m4_daily_dataset", "m4_weekly_dataset", "m4_monthly_dataset", "m4_quarterly_dataset", "car_parts_dataset_without_missing_values"       

# New training datasets
#"traffic_weekly_dataset" "tourism_monthly_dataset" "tourism_quarterly_dataset" "elecdemand_dataset" "wind_4_seconds_dataset" "solar_4_seconds_dataset" "solar_weekly_dataset" "covid_deaths_dataset" "covid_mobility_dataset_without_missing_values" "cif_2016_dataset" "fred_md_dataset" "nn5_daily_dataset_without_missing_values" "nn5_weekly_dataset" "m1_monthly_dataset" "m1_quarterly_dataset" "m1_yearly_dataset" "m3_monthly_dataset" "m3_quarterly_dataset" "m3_yearly_dataset" "m4_hourly_dataset" "m4_daily_dataset" "m4_weekly_dataset" "m4_monthly_dataset" "m4_quarterly_dataset" "car_parts_dataset_without_missing_values"

# New testing datasets
#"tourism_yearly_dataset" "us_births_dataset" "m4_yearly_dataset" "dominick_dataset" "hospital_dataset" "kaggle_web_traffic_dataset_without_missing_values" "kaggle_web_traffic_weekly_dataset" 

# All new datasets 
#"tourism_yearly_dataset" "us_births_dataset" "m4_yearly_dataset" "dominick_dataset" "hospital_dataset" "kaggle_web_traffic_dataset_without_missing_values" "kaggle_web_traffic_weekly_dataset" "traffic_weekly_dataset" "tourism_monthly_dataset" "tourism_quarterly_dataset" "elecdemand_dataset" "wind_4_seconds_dataset" "solar_4_seconds_dataset" "solar_weekly_dataset" "covid_deaths_dataset" "covid_mobility_dataset_without_missing_values" "cif_2016_dataset" "fred_md_dataset" "nn5_daily_dataset_without_missing_values" "nn5_weekly_dataset" "m1_monthly_dataset" "m1_quarterly_dataset" "m1_yearly_dataset" "m3_monthly_dataset" "m3_quarterly_dataset" "m3_yearly_dataset" "m4_hourly_dataset" "m4_daily_dataset" "m4_weekly_dataset" "m4_monthly_dataset" "m4_quarterly_dataset" "car_parts_dataset_without_missing_values"