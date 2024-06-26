# PLEASE FOLLOW THE BELOW INSTRUCTIONS FIRST

# 1. Install the requirements. It is recommend to use a new Anaconda environment with Python 3.10.8. Execute the below command (remove the #)
# !pip install -r requirements.txt

# 2. Please download https://drive.google.com/file/d/1JrDWMZyoPsc6d1wAAjgm3PosbGus-jCE/view?usp=sharing and use the below command to download the non-monash datasets (remove the #)
# tar -xvzf nonmonash_datasets.tar.gz -C datasets

New-Item -ItemType Directory -Path experiments -Force
New-Item -ItemType Directory -Path experiments/seeds -Force
New-Item -ItemType Directory -Path experiments/results -Force

$EXP_NAME = "pretraining_lag_llama"
$FILENAME = "experiments/seeds/${EXP_NAME}"
$CONFIGPATH = "configs/lag_llama.json"

Write-Output $EXP_NAME

# NUM_SEEDS used only if it is a new experiment
$NUM_SEEDS = 1

# Create seeds
if (Test-Path $FILENAME) {
    Write-Output "${FILENAME} already exists."

    $SEEDS = @()
    Get-Content $FILENAME | ForEach-Object {
        $SEEDS += $_
    }
    Write-Output "Found $($SEEDS.Count) seeds for training."
} else {
    # Write seeds
    Write-Output "${FILENAME} created. Writing seeds."
    New-Item -Path $FILENAME -ItemType File -Force
    for ($i = 0; $i -lt $NUM_SEEDS; $i++) {
        $SEED = (Get-Random) + 1
        Add-Content -Path $FILENAME -Value $SEED
    }

    # Read them
    $SEEDS = @()
    Get-Content $FILENAME | ForEach-Object {
        $SEEDS += $_
    }
}

# Train
foreach ($SEED in $SEEDS) {
    # $EXPERIMENT_NAME = "${EXP_NAME}_seed_${SEED}" # never used

    python run.py `
    -e $EXP_NAME -d "datasets" --seed $SEED `
    -r "experiments/results" `
    --batch_size 512 -m 1000 -n 128 `
    --wandb_entity "jakobl" --wandb_project "lag-llama" `
    --all_datasets "australian_electricity_demand" "electricity_hourly" "london_smart_meters_without_missing" "solar_10_minutes" "wind_farms_without_missing" "pedestrian_counts" "uber_tlc_hourly" "traffic" "kdd_cup_2018_without_missing" "saugeenday" "sunspot_without_missing" "exchange_rate" "cpu_limit_minute" "cpu_usage_minute" "function_delay_minute" "instances_minute" "memory_limit_minute" "memory_usage_minute" "platform_delay_minute" "requests_minute" "ett_h1" "ett_h2" "ett_m1" "ett_m2" "beijing_pm25" "AirQualityUCI" "beijing_multisite" "weather" `
    --test_datasets "weather" "pedestrian_counts" "exchange_rate" "ett_m2" "platform_delay_minute" "requests_minute" "beijing_pm25" `
    --num_workers 2 --args_from_dict_path $CONFIGPATH --search_batch_size `
    --lr 0.0001
}
