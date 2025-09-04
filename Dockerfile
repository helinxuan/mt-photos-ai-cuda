FROM python:3.11-slim
USER root

# 更新apt源并安装必要的系统依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    patch \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libfontconfig1 \
    liblmdb-dev \
    libgomp1 \
    libopenblas-dev && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# RUN apt update && \
#     apt install -y wget libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libfontconfig  python3 pip && \
#     wget http://nz2.archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2.22_amd64.deb && \
#     dpkg -i libssl1.1_1.1.1f-1ubuntu2.22_amd64.deb && \
#     rm -rf /var/lib/apt/lists/* /tmp/* /var/log/*
# 如果 libssl1.1_1.1.1f-1ubuntu2.22_amd64.deb 404 not found
# 请打开 http://nz2.archive.ubuntu.com/ubuntu/pool/main/o/openssl/ 查找 libssl1.1_1.1.1f-1ubuntu2.22_amd64.deb 对应的新版本
WORKDIR /app

COPY requirements.txt .

# 先安装PyTorch CPU版本
RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 再安装其他依赖包
RUN pip3 install --no-cache-dir -r requirements.txt --index-url=https://pypi.tuna.tsinghua.edu.cn/simple/

ENV API_AUTH_KEY=mt_photos_ai_extra
ENV CLIP_MODEL=ViT-B-16
ENV DEVICE=cpu
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

# 模型会自动下载，无需手动复制
# PaddleOCR和CLIP模型都会自动下载到缓存目录

# 复制应用代码
COPY server.py .
COPY immich_adapter.py .
COPY machine-learning ./machine-learning

EXPOSE 3004

CMD [ "python3", "/app/server.py" ]
