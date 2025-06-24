#!/bin/bash

echo "--- Starting Local End-to-End Test Run ---"

python ./app/predict.py \
    --config_path ./app/config.yaml \
    --model_dir ./outputs \
    --input_path ./saisdata/LIP_data_test_A.pkl \
    --output_path ./saisresult/submit.csv

echo "--- Local Test Run Finished ---"
echo "Please check for 'submit.csv' inside the 'saisresult' folder."