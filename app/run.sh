#!/bin/bash

INPUT_DATA_PATH="/saisdata/LIP_data_test_A.pkl"
OUTPUT_CSV_PATH="/saisresult/submit.csv"
CONFIG_FILE="/app/config.yaml"
MODEL_DIR="/app/outputs"


echo "=========================================================="
echo "           Starting Prediction Process"
echo "=========================================================="
echo "Using configuration file: ${CONFIG_FILE}"
echo "Loading models from: ${MODEL_DIR}"
echo "Reading input data from: ${INPUT_DATA_PATH}"
echo "Will write output to: ${OUTPUT_CSV_PATH}"
echo "----------------------------------------------------------"

python /app/predict.py \
    --config_path ${CONFIG_FILE} \
    --model_dir ${MODEL_DIR} \
    --input_path ${INPUT_DATA_PATH} \
    --output_path ${OUTPUT_CSV_PATH}

if [ $? -eq 0 ]; then
    echo "----------------------------------------------------------"
    echo "Prediction script finished successfully."
    echo "Submission file should be at ${OUTPUT_CSV_PATH}"
    echo "=========================================================="
else
    echo "----------------------------------------------------------"
    echo "!!! Prediction script failed with an error. !!!"
    echo "=========================================================="
    exit 1
fi
