# MT-Photos AI with Immich Integration

本项目将 [Immich](https://github.com/immich-app/immich) 的机器学习模块集成到 [MT-Photos AI](https://github.com/MT-Photos/mt-photos-ai) 中，提供更优秀的 CLIP 和人脸识别功能，并支持 CUDA 12.9 GPU 加速。

## 功能特性

- ✅ **CLIP 图像编码**: 使用 Immich 的高质量 CLIP 模型进行图像特征提取
- ✅ **CLIP 文本编码**: 支持多语言文本特征提取
- ✅ **人脸检测**: 基于 InsightFace 的高精度人脸检测
- ✅ **人脸识别**: 人脸特征提取和识别
- ✅ **OCR 文本识别**: 使用 PaddleOCR GPU 版本进行文本识别
- ✅ **API 兼容性**: 完全兼容原 MT-Photos AI 的 API 接口
- ✅ **GPU 加速**: 完整支持 CUDA 12.9 加速推理
- ✅ **集成显卡支持**: 支持 Intel 集成显卡通过 OpenVINO 加速推理

## 参考项目

- [Immich](https://github.com/immich-app/immich) - 高性能自托管照片和视频管理解决方案
- [MT-Photos AI](https://github.com/MT-Photos/mt-photos-ai) - MT Photos AI 识别服务
- [XiaoranQingxue/mt-ai](https://github.com/XiaoranQingxue/mt-ai) - 参考实现
- [MT-Photos/mt-photos-deepface](https://github.com/MT-Photos/mt-photos-deepface)

## 项目结构

```
e:\mt-photos-ai-cuda\
├── machine-learning/          # Immich 机器学习模块
├── immich_adapter.py         # Immich 适配器
├── server.py                 # 主服务文件 (已修改)
├── requirements.txt          # Python 依赖 (已更新)
├── .env                     # 环境配置文件
├── .env.example             # 配置模板
├── Dockerfile               # Docker 构建文件
└── cache/                   # 模型缓存目录
```

## 系统要求

### CUDA 版本（独立显卡）
#### 硬件要求
- NVIDIA GPU（支持 CUDA 12.9）
- 至少 8GB GPU 显存（推荐 16GB+）
- 至少 16GB 系统内存

#### 软件环境
- Windows 11 + WSL2 Ubuntu
- CUDA 12.9
- Python 3.8+
- Docker（可选）

### OpenVINO 版本（集成显卡）
#### 硬件要求
- Intel 集成显卡（支持 OpenCL）
- 至少 8GB 系统内存（推荐 16GB+）
- 支持的 Intel 处理器（第6代酷睿及以上）

#### 软件环境
- Windows 10/11 或 Linux
- Intel GPU 驱动程序
- Python 3.8+
- Docker（可选）

## 安装步骤

### CUDA 版本安装（独立显卡）

#### 1. 环境准备

确保已安装 CUDA 12.9 和相应的驱动程序：

```bash
# 检查 CUDA 版本
nvidia-smi
nvcc --version
```

### 2. 克隆项目

```bash
cd /opt
git clone <repository-url> mt-photos-ai-cuda
cd mt-photos-ai-cuda
```

### 3. 创建虚拟环境

```bash
python -m venv venv
source venv/bin/activate
```

### 4. 配置 pip 源（可选，提升下载速度）

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### 5. 安装 PyTorch（CUDA 12.9 版本）

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu129
```

### 6. 安装项目依赖

```bash
pip install -r requirements.txt
python -m pip install gunicorn
```

#### 7. 验证安装

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

### OpenVINO 版本安装（集成显卡）

#### 1. 使用 Docker 快速部署（推荐）

```bash
# Windows 用户
build-openvino.bat

# Linux/macOS 用户
chmod +x build-openvino.sh
./build-openvino.sh

# 启动服务
docker compose -f docker-compose.openvino.yml up -d
```

#### 2. 手动安装

```bash
# 克隆项目
git clone <repository-url> mt-photos-ai-cuda
cd mt-photos-ai-cuda

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate     # Windows

# 安装 OpenVINO 版本依赖
pip install -r requirements.openvino.txt

# 启动服务
python server.py
```

#### 3. 验证 Intel GPU 支持

```bash
# 检查 OpenCL 设备
python -c "import onnxruntime as ort; print('Available providers:', ort.get_available_providers())"

# 检查 Intel GPU 驱动
# Windows: 设备管理器 -> 显示适配器
# Linux: lspci | grep VGA
```

## 配置说明

### 环境变量配置

复制并编辑配置文件：

```bash
cp .env.example .env
```

主要配置项：

```bash
# 基本配置
API_AUTH_KEY=mt_photos_ai_extra
HTTP_PORT=3004

# CLIP 模型配置
CLIP_MODEL_NAME=nllb-clip-large-siglip__v1

# 人脸识别配置
FACE_MODEL_NAME=antelopev2
FACE_THRESHOLD=0.7

# 设备类型配置（OpenVINO 版本专用）
DEVICE=openvino  # 可选值: cpu, cuda, openvino

# 模型缓存目录
MACHINE_LEARNING_CACHE_FOLDER=/model-cache

# 性能配置
MODEL_TTL=300
AUTO_LOAD_TXT_MODAL=off

# 日志级别
LOG_LEVEL=INFO

# 服务重启时间间隔 (秒)
SERVER_RESTART_TIME=300
```

### OpenVINO 版本特殊配置

OpenVINO 版本支持以下额外配置：

```bash
# OpenVINO 设备配置
DEVICE=openvino

# Intel GPU 设备选择（可选）
# AUTO: 自动选择最佳设备
# GPU: 强制使用集成显卡
# CPU: 强制使用 CPU
OPENVINO_DEVICE=AUTO

# 性能优化（集成显卡推荐设置）
MODEL_TTL=600  # 增加模型缓存时间
MAX_BATCH_SIZE=1  # 降低批处理大小
```

### 模型选择

#### CLIP 模型选项：
- `nllb-clip-large-siglip__v1`（推荐，多语言支持）
- `XLM-Roberta-Large-Vit-B-16Plus`（默认，多语言支持）
- `ViT-B-32::openai`
- `ViT-B-16::openai`
- `nllb-clip-base-siglip__mrl`

#### 人脸识别模型选项：
- `antelopev2`（推荐，高精度）
- `buffalo_l`（较快速度）

## 启动服务

### 开发模式

```bash
source venv/bin/activate
python server.py
```

### 生产模式（使用 Gunicorn）

```bash
source venv/bin/activate
gunicorn -w 1 -k uvicorn.workers.UvicornWorker server:app --bind 0.0.0.0:3004
```

### Docker 部署（可选）

```bash
# 构建镜像
docker build -t mt-photos-ai-cuda .

# 运行容器
docker run -d \
  --name mt-photos-ai \
  --gpus all \
  -p 3004:3004 \
  -v $(pwd)/cache:/model-cache \
  -v $(pwd)/.env:/app/.env \
  mt-photos-ai-cuda
```

或使用 docker-compose：

```yaml
version: "3"
services:
  mtphotos_ai:
    build: .
    container_name: mt-ai-immich
    restart: always
    ports:
      - 3004:3004
    volumes:
      - ./model-cache:/cache
    environment:
      - API_AUTH_KEY=mt_photos_ai_extra
      - CLIP_MODEL_NAME=nllb-clip-large-siglip__v1
      - FACE_MODEL_NAME=antelopev2
      - FACE_THRESHOLD=0.7
      - MODEL_TTL=300
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## API 接口

### 原有接口 (保持兼容)

#### 1. 健康检查
```bash
curl -X POST "http://localhost:3004/check" \
  -H "api-key: mt_photos_ai_extra"
```

#### 2. OCR 文本识别
```bash
curl -X POST "http://localhost:3004/ocr" \
  -H "api-key: mt_photos_ai_extra" \
  -F "file=@image.jpg"
```

#### 3. CLIP 图像编码
```bash
curl -X POST "http://localhost:3004/clip/img" \
  -H "api-key: mt_photos_ai_extra" \
  -F "file=@image.jpg"
```

#### 4. CLIP 文本编码
```bash
curl -X POST "http://localhost:3004/clip/txt" \
  -H "Content-Type: application/json" \
  -H "api-key: mt_photos_ai_extra" \
  -d '{"text":"飞机"}'
```

### 新增接口

#### 5. 人脸检测和识别
```bash
curl -X POST "http://localhost:3004/face/detect" \
  -H "api-key: mt_photos_ai_extra" \
  -F "file=@image.jpg"
```

## 性能优化

### 内存管理
- 设置 `MODEL_TTL` 来自动卸载模型释放内存
- 设置 `AUTO_LOAD_TXT_MODAL=on` 来预加载文本模型（占用更多内存但响应更快）
- 根据 GPU 显存大小选择合适的模型

### GPU 支持
- 自动检测 CUDA 可用性
- 使用 `onnxruntime-gpu` 进行 GPU 加速
- 确保 PyTorch 与 CUDA 12.9 版本兼容

### 推理速度优化
- 确保使用 CUDA 加速
- 批量处理图像
- 合理设置并发数

## 与 MT-Photos 集成

在 MT-Photos 中配置 AI 服务地址：

1. 人脸置信度建议设置为 0.55-0.70 (对应 FACE_THRESHOLD 参数)
2. CLIP 向量长度根据模型自动调整

### 环境变量配置

- `FACE_THRESHOLD`: 人脸检测置信度阈值，默认 0.7

## 故障排除

### 常见问题

1. **CUDA 不可用**
   ```bash
   # 检查 CUDA 安装
   nvidia-smi
   # 检查 PyTorch CUDA 支持
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **模型下载失败**
   - 检查网络连接，模型会自动从 Hugging Face 下载
   - 手动下载模型到缓存目录
   - 使用代理或镜像源

3. **内存不足**
   - 调整 `MODEL_TTL` 设置或减少并发请求
   - 使用更小的模型
   - 减少并发数

4. **端口被占用**
   ```bash
   # 检查端口占用
   netstat -tulpn | grep 3004
   # 修改 .env 中的 HTTP_PORT
   ```

5. **GPU 不可用**
   - 检查 CUDA 安装和 `onnxruntime-gpu` 版本
   - 确保 PyTorch 与 CUDA 版本兼容

### 日志调试

设置 `LOG_LEVEL=DEBUG` 获取详细日志信息。

```bash
# 查看服务日志
tail -f /var/log/mt-photos-ai.log

# 调整日志级别
# 在 .env 中设置 LOG_LEVEL=DEBUG
```

## 性能对比

| 版本类型 | 硬件要求 | 推理速度 | 显存占用 | 适用场景 |
|---------|---------|---------|---------|----------|
| CUDA 版本 | NVIDIA 独立显卡 | 快 | 8GB+ | 高性能服务器、工作站 |
| OpenVINO 版本 | Intel 集成显卡 | 中等 | 系统内存 | 普通电脑、笔记本 |
| CPU 版本 | 仅 CPU | 慢 | 系统内存 | 无 GPU 设备 |

### 性能建议

- **CUDA 版本**: 适合有独立显卡的用户，性能最佳
- **OpenVINO 版本**: 适合只有集成显卡的用户，性能适中
- **CPU 版本**: 适合无 GPU 的服务器，性能较低但兼容性最好

## 注意事项

### OpenVINO 版本限制

1. **硬件兼容性**:
   - 需要第6代 Intel 酷睿处理器及以上
   - 需要支持 OpenCL 的集成显卡
   - 部分老旧集成显卡可能不支持

2. **性能特点**:
   - 推理速度比 CUDA 版本慢 2-3 倍
   - 内存占用相对较低
   - 首次加载模型时间较长

3. **功能限制**:
   - PaddleOCR 使用 CPU 版本（无 GPU 加速）
   - 批处理大小建议设为 1
   - 不支持某些高级 CUDA 特性

### 故障排除

#### OpenVINO 版本常见问题

1. **Intel GPU 驱动问题**:
   ```bash
   # Windows: 更新 Intel 显卡驱动
   # 访问 Intel 官网下载最新驱动
   
   # Linux: 安装 Intel GPU 驱动
   sudo apt install intel-opencl-icd
   ```

2. **OpenCL 不可用**:
   ```bash
   # 检查 OpenCL 设备
   clinfo
   
   # 如果没有 clinfo，安装它
   sudo apt install clinfo
   ```

3. **容器权限问题**:
   ```bash
   # 确保 Docker 容器有访问 GPU 的权限
   docker run --device /dev/dri:/dev/dri ...
   ```

## 开发说明

### 项目结构
```
mt-photos-ai-cuda/
├── server.py                    # 主服务文件
├── immich_adapter.py            # Immich 适配器
├── requirements.txt             # CUDA 版本依赖
├── requirements.openvino.txt    # OpenVINO 版本依赖
├── Dockerfile                   # CUDA 版本 Docker 文件
├── Dockerfile.openvino          # OpenVINO 版本 Docker 文件
├── docker-compose.openvino.yml  # OpenVINO 版本 Docker Compose
├── build-openvino.bat           # Windows 构建脚本
├── build-openvino.sh            # Linux 构建脚本
├── .env                         # 环境配置
└── machine-learning/            # Immich 机器学习模块
```

### 添加新功能
1. 在 `server.py` 中添加新的 API 端点
2. 实现相应的处理函数
3. 更新 `requirements.txt`
4. 添加相应的测试

## 许可证

本项目基于相关开源项目的许可证发布，请参考各个依赖项目的许可证要求。

## 贡献

欢迎提交 Issue 和 Pull Request 来改进本项目。

## 联系方式

如有问题，请通过 GitHub Issues 联系。