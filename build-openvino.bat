@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo === MT-Photos AI OpenVINO版本构建脚本 ===
echo 此版本支持Intel集成显卡加速，适用于没有独立显卡的设备
echo.

REM 检查Docker是否安装
docker --version >nul 2>&1
if errorlevel 1 (
    echo 错误: Docker未安装，请先安装Docker Desktop
    pause
    exit /b 1
)

REM 检查Docker Compose是否可用
docker compose version >nul 2>&1
if errorlevel 1 (
    docker-compose --version >nul 2>&1
    if errorlevel 1 (
        echo 错误: Docker Compose未安装，请先安装Docker Compose
        pause
        exit /b 1
    )
)

echo 1. 构建OpenVINO版本的Docker镜像...
docker build -f Dockerfile.openvino -t mt-photos-ai-openvino:latest .

if errorlevel 1 (
    echo ❌ Docker镜像构建失败！
    pause
    exit /b 1
)

echo ✅ Docker镜像构建成功！
echo.

echo 2. 创建必要的目录...
if not exist "cache" mkdir cache
if not exist "models" mkdir models

echo ✅ 目录创建完成！
echo.
echo === 构建完成 ===
echo.
echo 使用方法:
echo 1. 启动服务: docker compose -f docker-compose.openvino.yml up -d
echo 2. 查看日志: docker compose -f docker-compose.openvino.yml logs -f
echo 3. 停止服务: docker compose -f docker-compose.openvino.yml down
echo 4. 重新构建: docker compose -f docker-compose.openvino.yml up --build -d
echo.
echo 注意事项:
echo - 确保您的系统支持Intel集成显卡
echo - 首次运行时会自动下载模型文件，请耐心等待
echo - 服务启动后可通过 http://localhost:8060 访问
echo - 集成显卡性能相比独立显卡较低，处理速度会慢一些
echo.
pause