#!/bin/bash

# MT-Photos AI OpenVINO版本构建脚本
# 支持Intel集成显卡加速

set -e

echo "=== MT-Photos AI OpenVINO版本构建脚本 ==="
echo "此版本支持Intel集成显卡加速，适用于没有独立显卡的设备"
echo ""

# 检查Docker是否安装
if ! command -v docker &> /dev/null; then
    echo "错误: Docker未安装，请先安装Docker"
    exit 1
fi

# 检查Docker Compose是否安装
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "错误: Docker Compose未安装，请先安装Docker Compose"
    exit 1
fi

echo "1. 构建OpenVINO版本的Docker镜像..."
docker build -f Dockerfile.openvino -t mt-photos-ai-openvino:latest .

if [ $? -eq 0 ]; then
    echo "✅ Docker镜像构建成功！"
else
    echo "❌ Docker镜像构建失败！"
    exit 1
fi

echo ""
echo "2. 创建必要的目录..."
mkdir -p cache
mkdir -p models

echo "✅ 目录创建完成！"
echo ""
echo "=== 构建完成 ==="
echo ""
echo "使用方法:"
echo "1. 启动服务: docker-compose -f docker-compose.openvino.yml up -d"
echo "2. 查看日志: docker-compose -f docker-compose.openvino.yml logs -f"
echo "3. 停止服务: docker-compose -f docker-compose.openvino.yml down"
echo ""
echo "注意事项:"
echo "- 确保您的系统支持Intel集成显卡"
echo "- 首次运行时会自动下载模型文件，请耐心等待"
echo "- 服务启动后可通过 http://localhost:8060 访问"
echo "- 集成显卡性能相比独立显卡较低，处理速度会慢一些"
echo ""