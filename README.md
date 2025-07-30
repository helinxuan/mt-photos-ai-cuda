# MT-Photos AI CUDA

基于 CUDA 的 MT-Photos AI 服务，提供图像识别、人脸检测和 OCR 功能，由于 Immich 效果好些，移植了人脸和clip过来。

## 项目介绍

本项目是一个高性能的 AI 服务，专为 MT-Photos 照片管理系统设计，集成了以下功能：

- **CLIP 图像特征提取**：支持多语言图像搜索
- **人脸检测与识别**：基于 Immich 的人脸识别技术
- **OCR 文字识别**：使用 RapidOCR 进行图像文字提取
- **GPU 加速**：完整支持 CUDA 12.9 加速推理

## 参考项目

- [XiaoranQingxue/mt-ai](https://github.com/XiaoranQingxue/mt-ai)
- [MT-Photos/mt-photos-ai](https://github.com/MT-Photos/mt-photos-ai)
- [MT-Photos/mt-photos-deepface](https://github.com/MT-Photos/mt-photos-deepface)
- [immich-app/immich](https://github.com/immich-app/immich)

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
# CLIP 模型配置
CLIP_MODEL_NAME=nllb-clip-large-siglip__v1

# 人脸识别模型
FACE_MODEL_NAME=antelopev2
FACE_THRESHOLD=0.7

# 模型缓存目录
MACHINE_LEARNING_CACHE_FOLDER=/model-cache

# 模型TTL（秒，0表示不自动卸载）
MODEL_TTL=300

# 日志级别
LOG_LEVEL=INFO

# 服务端口
HTTP_PORT=3004

# API 认证密钥
API_AUTH_KEY=mt_photos_ai_extra
```

### 模型选择

#### CLIP 模型选项：
- `nllb-clip-large-siglip__v1`（推荐，多语言支持）
- `XLM-Roberta-Large-Vit-B-16Plus`
- `XLM-Roberta-Large-ViT-H-14__frozen_laion5b_s13b_b90k`

#### 人脸识别模型选项：
- `antelopev2`（推荐）
- `buffalo_l`

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

## API 接口

### 健康检查
```
GET /health
```

### CLIP 图像特征提取
```
POST /clip/img
Content-Type: multipart/form-data
Authorization: Bearer <API_AUTH_KEY>

Body: image file
```

### CLIP 文本特征提取
```
POST /clip/txt
Content-Type: application/json
Authorization: Bearer <API_AUTH_KEY>

Body: {"text": "搜索文本"}
```

### 人脸检测
```
POST /face
Content-Type: multipart/form-data
Authorization: Bearer <API_AUTH_KEY>

Body: image file
```

### OCR 文字识别
```
POST /ocr
Content-Type: multipart/form-data
Authorization: Bearer <API_AUTH_KEY>

Body: image file
```

## 性能优化

### GPU 内存优化
- 调整 `MODEL_TTL` 参数控制模型缓存时间
- 根据 GPU 显存大小选择合适的模型
- 使用 `AUTO_LOAD_TXT_MODAL=off` 减少内存占用

### 推理速度优化
- 确保使用 CUDA 加速
- 批量处理图像
- 合理设置并发数

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
   - 检查网络连接
   - 手动下载模型到缓存目录
   - 使用代理或镜像源

3. **内存不足**
   - 减少并发数
   - 调整模型 TTL
   - 使用更小的模型

4. **端口被占用**
   ```bash
   # 检查端口占用
   netstat -tulpn | grep 3004
   # 修改 .env 中的 HTTP_PORT
   ```

### 日志查看

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