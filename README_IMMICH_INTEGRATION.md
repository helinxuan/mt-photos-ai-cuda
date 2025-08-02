# MT-Photos AI with Immich Integration

本项目将 [Immich](https://github.com/immich-app/immich) 的机器学习模块集成到 [MT-Photos AI](https://github.com/MT-Photos/mt-photos-ai) 中，提供更优秀的 CLIP 和人脸识别功能。

## 功能特性

- ✅ **CLIP 图像编码**: 使用 Immich 的高质量 CLIP 模型进行图像特征提取
- ✅ **CLIP 文本编码**: 支持多语言文本特征提取
- ✅ **人脸检测**: 基于 InsightFace 的高精度人脸检测
- ✅ **人脸识别**: 人脸特征提取和识别
- ✅ **OCR 文本识别**: 保持原有的 RapidOCR 功能
- ✅ **API 兼容性**: 完全兼容原 MT-Photos AI 的 API 接口

## 项目结构

```
e:\mt-photos-ai-cuda\
├── machine-learning/          # Immich 机器学习模块
├── immich_adapter.py         # Immich 适配器
├── server.py                 # 主服务文件 (已修改)
├── requirements.txt          # Python 依赖 (已更新)
├── .env.immich              # Immich 配置文件
└── README_IMMICH_INTEGRATION.md
```

## 安装和配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.immich` 到 `.env` 或直接设置环境变量：

```bash
# 基本配置
API_AUTH_KEY=mt_photos_ai_extra
HTTP_PORT=8060

# CLIP 模型配置
CLIP_MODEL_NAME=XLM-Roberta-Large-Vit-B-16Plus

# 人脸识别配置
FACE_MODEL_NAME=antelopev2
FACE_THRESHOLD=0.7

# 性能配置
MODEL_TTL=300
AUTO_LOAD_TXT_MODAL=off
```

### 3. 启动服务

```bash
python server.py
```

## API 接口

### 原有接口 (保持兼容)

#### 1. 健康检查
```bash
curl -X POST "http://localhost:8060/check" \
  -H "api-key: mt_photos_ai_extra"
```

#### 2. OCR 文本识别
```bash
curl -X POST "http://localhost:8060/ocr" \
  -H "api-key: mt_photos_ai_extra" \
  -F "file=@image.jpg"
```

#### 3. CLIP 图像编码
```bash
curl -X POST "http://localhost:8060/clip/img" \
  -H "api-key: mt_photos_ai_extra" \
  -F "file=@image.jpg"
```

#### 4. CLIP 文本编码
```bash
curl -X POST "http://localhost:8060/clip/txt" \
  -H "Content-Type: application/json" \
  -H "api-key: mt_photos_ai_extra" \
  -d '{"text":"飞机"}'
```

### 新增接口

#### 5. 人脸检测和识别
```bash
curl -X POST "http://localhost:8060/face/detect" \
  -H "api-key: mt_photos_ai_extra" \
  -F "file=@image.jpg"
```

## 模型配置

### CLIP 模型选项

- `XLM-Roberta-Large-Vit-B-16Plus` (默认，多语言支持)
- `ViT-B-32::openai`
- `ViT-B-16::openai`
- `nllb-clip-base-siglip__mrl`

### 人脸识别模型选项

- `antelopev2` (默认，高精度)
- `buffalo_l` (较快速度)

## 性能优化

### 内存管理
- 设置 `MODEL_TTL` 来自动卸载模型释放内存
- 设置 `AUTO_LOAD_TXT_MODAL=on` 来预加载文本模型（占用更多内存但响应更快）

### GPU 支持
- 自动检测 CUDA 可用性
- 使用 `onnxruntime-gpu` 进行 GPU 加速

## Docker 部署

可以参考原项目的 Docker 配置，需要确保：

1. 安装了 CUDA 支持（如果使用 GPU）
2. 挂载模型缓存目录
3. 设置正确的环境变量

```yaml
version: "3"
services:
  mtphotos_ai:
    build: .
    container_name: mt-ai-immich
    restart: always
    ports:
      - 3003:8060
    volumes:
      - ./model-cache:/cache
    environment:
      - API_AUTH_KEY=mt_photos_ai_extra
      - CLIP_MODEL_NAME=XLM-Roberta-Large-Vit-B-16Plus
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

## 与 MT-Photos 集成

在 MT-Photos 中配置 AI 服务地址：

1. 人脸置信度建议设置为 0.55-0.70
2. 人物匹配差异值建议设置为 0.35-0.50
3. CLIP 向量长度根据模型自动调整

## 故障排除

### 常见问题

1. **模型下载失败**: 检查网络连接，模型会自动从 Hugging Face 下载
2. **内存不足**: 调整 `MODEL_TTL` 设置或减少并发请求
3. **GPU 不可用**: 检查 CUDA 安装和 `onnxruntime-gpu` 版本

### 日志调试

设置 `LOG_LEVEL=DEBUG` 获取详细日志信息。

## 参考项目

- [Immich](https://github.com/immich-app/immich) - 高性能自托管照片和视频管理解决方案
- [MT-Photos AI](https://github.com/MT-Photos/mt-photos-ai) - MT Photos AI 识别服务
- [XiaoranQingxue/mt-ai](https://github.com/XiaoranQingxue/mt-ai) - 参考实现