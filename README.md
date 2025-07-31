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

### 硬件要求
- NVIDIA GPU（支持 CUDA 12.9）
- 至少 8GB GPU 显存（推荐 16GB+）
- 至少 16GB 系统内存

### 软件环境
- Windows 11 + WSL2 Ubuntu
- CUDA 12.9
- Python 3.8+
- Docker（可选）

## 安装步骤

### 1. 环境准备

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

### 7. 验证安装

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}')"
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
2. 人物匹配差异值建议设置为 0.35-0.50 (对应 FACE_MAX_DISTANCE 参数)
3. CLIP 向量长度根据模型自动调整

### 环境变量配置

- `FACE_THRESHOLD`: 人脸检测置信度阈值，默认 0.7
- `FACE_MAX_DISTANCE`: 人脸识别最大距离阈值，默认 0.5

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

## 开发说明

### 项目结构
```
mt-photos-ai-cuda/
├── server.py              # 主服务文件
├── immich_adapter.py      # Immich 适配器
├── requirements.txt       # Python 依赖
├── .env                   # 环境配置
├── .env.example          # 配置模板
├── Dockerfile            # Docker 构建文件
├── machine-learning/     # Immich ML 模块
└── cache/               # 模型缓存目录
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