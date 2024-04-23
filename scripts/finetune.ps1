$CONFIGPATH = "configs/lag_llama.json"
$PRETRAINING_EXP_NAME = "pretraining_lag_llama"
$PERCENTAGE = 100 # Change to lesser value to limit the history. Use 20, 40, 60, 80 to reproduce experiments in the paper.

$FINETUNE_DATASETS = @("weather", "pedestrian_counts", "exchange_rate", "ett_m2", "platform_delay_minute", "requests_minute", "beijing_pm25")

foreach ($FINETUNE_DATASET in $FINETUNE_DATASETS) {
    $EXP_NAME = "{0}_finetune_on_{1}" -f $PRETRAINING_EXP_NAME, $FINETUNE_DATASET

    # We reuse the same seeds as used for pretraining
    $FILENAME = "experiments/seeds/$PRETRAINING_EXP_NAME"
    Write-Output $PRETRAINING_EXP_NAME

    # Get the seeds
    if (Test-Path -Path $FILENAME) {
        Write-Output "$FILENAME found. Reading seeds."
        $SEEDS = Get-Content -Path $FILENAME
        Write-Output "Found $($SEEDS.Count) seeds for finetuning."
    }
    else {
        Write-Output "$FILENAME does not exist. Cannot perform finetuning."
        exit 0
    }

    # Iterate through all training dataset
    foreach ($SEED in $SEEDS) {
        $EXPERIMENT_NAME = "{0}_seed_{1}" -f $EXP_NAME, $SEED

        python run.py `
        -e $EXPERIMENT_NAME -d "datasets" --seed $SEED `
        -r "experiments/results" `
        --batch_size 512 -m 1000 -n 128 `
        --wandb_entity "jakobl" --wandb_project "lag-llama" `
        --num_workers 2 --args_from_dict_path $CONFIGPATH --search_batch_size `
        --single_dataset $FINETUNE_DATASET `
        --get_ckpt_path_from_experiment_name $PRETRAINING_EXP_NAME --lr 0.00001 --use_dataset_prediction_length --num_validation_windows 1 `
        --single_dataset_last_k_percentage $PERCENTAGE
    }
}
