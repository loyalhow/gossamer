#!/bin/bash

# ==============================================================================
#  比賽提交 - Docker 入口腳本 (run.sh)
# ==============================================================================
#
# 功能:
# 1. 設置所有必要的絕對路徑。
# 2. 打印日誌訊息，方便在比賽平台追蹤運行狀態。
# 3. 調用核心的 Python 預測腳本 (predict.py)，並將路徑作為參數傳入。
#
# 存放位置: /app/run.sh
#
# ==============================================================================

# --- 步驟 1: 設置所有檔案和目錄的路徑 ---
# 確保使用平台規定的絕對路徑

# 輸入檔案的路徑
# 比賽平台會將測試數據掛載到 /saisdata 目錄下
INPUT_DATA_PATH="/saisdata/LIP_data_test_A.pkl"

# 輸出檔案的路徑
# 我們的腳本需要將結果寫入到 /saisresult 目錄下
OUTPUT_CSV_PATH="/saisresult/submit.csv"

# 模型和設定檔在我們鏡像內部的位置
CONFIG_FILE="/app/config.yaml"
MODEL_DIR="/app/outputs"


# --- 步驟 2: 打印日誌，方便偵錯 ---
echo "=========================================================="
echo "           Starting Prediction Process"
echo "=========================================================="
echo "Using configuration file: ${CONFIG_FILE}"
echo "Loading models from: ${MODEL_DIR}"
echo "Reading input data from: ${INPUT_DATA_PATH}"
echo "Will write output to: ${OUTPUT_CSV_PATH}"
echo "----------------------------------------------------------"


# --- 步驟 3: 執行核心的 Python 預測腳本 ---
# 使用 python 命令，並通過命令列參數傳遞所有路徑
# 這樣做的好處是，我們的 Python 腳本不需要硬編碼任何路徑，非常靈活
python /app/predict.py \
    --config_path ${CONFIG_FILE} \
    --model_dir ${MODEL_DIR} \
    --input_path ${INPUT_DATA_PATH} \
    --output_path ${OUTPUT_CSV_PATH}


# --- 步驟 4: 檢查執行結果並打印結束日誌 ---
# $? 變數會保存上一條命令的退出碼，0 代表成功
if [ $? -eq 0 ]; then
    echo "----------------------------------------------------------"
    echo "Prediction script finished successfully."
    echo "Submission file should be at ${OUTPUT_CSV_PATH}"
    echo "=========================================================="
else
    echo "----------------------------------------------------------"
    echo "!!! Prediction script failed with an error. !!!"
    echo "=========================================================="
    # 返回一個非零的退出碼，告知平台運行失敗
    exit 1
fi