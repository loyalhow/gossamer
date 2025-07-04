
FROM sais-public-registry.cn-shanghai.cr.aliyuncs.com/sais-public/pytorch:2.0.0-py3.9.12-cuda11.8.0-u22.04

RUN apt-get update && \
    apt-get install -y curl dos2unix && \
    rm -rf /var/lib/apt/lists/*


WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY app/ /app/
COPY outputs/ /app/outputs/
COPY esm2_model_local/ /app/esm2_model_local/

RUN dos2unix /app/run.sh && chmod +x /app/run.sh

