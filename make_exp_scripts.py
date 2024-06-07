import json
from itertools import product
import os 
import random

def get_config(use_mamba, use_jamba, use_moe):
    config = {
        "use_single_instance_sampler": True,
        "stratified_sampling": "series",
        "data_normalization": "robust",
        "n_layer": d,
        "n_head": 9,
        "n_embd_per_head": 16,
        "time_feat": True,
        "context_length": ctx,
        "aug_prob": 0.5,
        "freq_mask_rate": 0.5,
        "freq_mixing_rate": 0.25,
        "weight_decay": 0.0,
        "dropout": 0.0,
        "use_mamba": use_mamba,
        "use_jamba": use_jamba,
        "use_moe": use_moe
    }

def get_slurm_script(exp_name, seed, config_filepath, gpu_node, gpu_days, use_expanded_datasets):
    if use_expanded_datasets:
        all_datasets = """"australian_electricity_demand" "electricity_hourly" "london_smart_meters_without_missing" "solar_10_minutes" "wind_farms_without_missing" "pedestrian_counts" "uber_tlc_hourly" "traffic" "kdd_cup_2018_without_missing" "saugeenday" "sunspot_without_missing" "exchange_rate" "cpu_limit_minute" "cpu_usage_minute" "function_delay_minute" "instances_minute" "memory_limit_minute" "memory_usage_minute" "platform_delay_minute" "requests_minute" "ett_h1" "ett_h2" "ett_m1" "ett_m2" "beijing_pm25" "AirQualityUCI" "beijing_multisite" "weather""""
        test_datasets = """"weather" "pedestrian_counts" "exchange_rate" "ett_m2" "platform_delay_minute" "requests_minute" "beijing_pm25""""
    else:
        all_datasets = """"australian_electricity_demand" "electricity_hourly" "london_smart_meters_without_missing" "solar_10_minutes" "wind_farms_without_missing" "pedestrian_counts" "uber_tlc_hourly" "traffic" "kdd_cup_2018_without_missing" "saugeenday" "sunspot_without_missing" "exchange_rate" "cpu_limit_minute" "cpu_usage_minute" "function_delay_minute" "instances_minute" "memory_limit_minute" "memory_usage_minute" "platform_delay_minute" "requests_minute" "ett_h1" "ett_h2" "ett_m1" "ett_m2" "beijing_pm25" "AirQualityUCI" "beijing_multisite" "weather""""
        test_datasets = """"weather" "pedestrian_counts" "exchange_rate" "ett_m2" "platform_delay_minute" "requests_minute" "beijing_pm25""""
    
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name=ll
#SBATCH --output=output/%x_%j.out
#SBATCH --error=output/%x_%j.err
#SBATCH --mail-user="s3442209@vuw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --partition="gpu-{gpu_node}"
#SBATCH --time=0{gpu_days}-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --gpus=1

# Experiment variables setup
""" + f"""EXP_NAME="{exp_name}"
CONFIGPATH="{config_filepath}"
SEED="{seed}"
""" + """# ---------------------------------------------------


echo "#### Starting experiment"
echo "User: $SLURM_JOB_USER"
echo "Job ID: $SLURM_JOB_ID"

CWD=$(pwd)
DATE=$(date)
echo "This job was submitted from $SLURM_SUBMIT_DIR and I am currently in $CWD"
echo "It is now $DATE"

ml purge
ml load shared DefaultModules ALICE/default gcc/11.2.0 slurm/alice/23.02.7 CUDA/11.8

conda init bash
source ~/.bashrc
conda activate /home/s3442209/data1/SADL/envs/SADL310cu118t21

cd ..
CWD=$(pwd)
echo "Reached working directory $CWD"

echo "[$SHELL] Using GPU: "$CUDA_VISIBLE_DEVICES

### Actual experiment script

mkdir -p experiments
mkdir -p experiments/seeds
mkdir -p experiments/results

FILENAME="experiments/seeds/${EXP_NAME}"
echo $EXP_NAME


# Train
#EXPERIMENT_NAME="${EXP_NAME}_seed_${SEED}"

python run.py \
-e $EXP_NAME -d "datasets" --seed $SEED \
-r "experiments/results" \
--batch_size 512 -m 1000 -n 128 \
--wandb_entity "maranc" --wandb_project "lag-llama" \ """ + f"""
--all_datasets {all_datasets} \ 
--test_datasets {test_datasets} \ """ + """
--all_datasets 
--test_datasets 
--num_workers 2 --args_from_dict_path $CONFIGPATH --search_batch_size \
--lr 0.0001

echo "#### Finished experiment :)"
"""
    return slurm_script

for exp in ['exp_lmj', 'exp_moe_d', 'exp_data']:
    os.makedirs(f"config/{exp}", exist_ok=True)
    os.makedirs(f"scripts/{exp}", exist_ok=True)

# Experiment 1 Impact of SSM vs attention
exp_group = 'exp_lmj'
models = ["llama", "mamba", "jamba"]
context_length = [32, 64, 128, 256, 512, 1024]
moe = [False]
depth = [8]
n_seeds = 5

n_seeds = list(range(n_seeds))
for model, ctx, use_moe, d, seed_n in product(models, context_length, moe, depth, n_seeds):
    print(f"Writing files for model, ctx, use_moe, d, seed_n: {[model, ctx, use_moe, d, seed_n]}")
    
    # Computing experiment variables
    use_mamba = model in ['mamba', 'jamba'] 
    use_jamba = model in ['jamba']
    seed = random.randint(0, 50000)

    # Names for files and experiments
    usemoe_str = "_stmoe" if use_moe else ""
    exp_str = f"{model}{usemoe_str}_c{ctx}_d{d}"

    config_filename = exp_str
    script_filename = exp_str + f"_rep{seed_n}"
    wandb_exp_name = exp_group_name + "_" + exp_str
    
    # Writing config file
    config = get_config(use_mamba, use_jamba, use_moe)
    config_filepath = f"configs/{exp_group_name}/{config_filename}.json"
    with open(config_filepath, "w") as fp:
        json.dump(config , fp, indent=4) 

    # Writing slurm script
    is_long_training = (use_moe and ctx>=512) or (model=="llama" and ctx>=512) or (ctx>=1024)
    gpu_node = "long" if is_long_training else "medium"
    gpu_days = 3 if is_long_training else 1
    use_expanded_datasets = False

    slurm_script = get_slurm_script(exp_name, seed, config_filepath, gpu_node, gpu_days, use_expanded_datasets)
    slurm_filepath = f"scripts/{exp_group_name}/{script_filename}.slurm"
    with open(slurm_filepath, "w") as f:
        f.write(slurm_script)
