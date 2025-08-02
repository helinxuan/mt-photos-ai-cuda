@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo === MT-Photos AI OpenVINO版本启动脚本 ===
echo.

REM 检查Docker是否运行
docker info >nul 2>&1
if errorlevel 1 (
    echo 错误: Docker未运行，请先启动Docker Desktop
    pause
    exit /b 1
)

REM 检查镜像是否存在
docker images mt-photos-ai-openvino:latest --format "table {{.Repository}}" | findstr "mt-photos-ai-openvino" >nul
if errorlevel 1 (
    echo 镜像不存在，开始构建...
    call build-openvino.bat
    if errorlevel 1 (
        echo 构建失败，请检查错误信息
        pause
        exit /b 1
    )
)

echo 启动 OpenVINO 版本服务...
docker compose -f docker-compose.openvino.yml up -d

if errorlevel 1 (
    echo ❌ 服务启动失败！
    echo.
    echo 常见解决方案:
    echo 1. 确保Docker Desktop正在运行
    echo 2. 检查端口8060是否被占用
    echo 3. 运行: docker compose -f docker-compose.openvino.yml logs
    pause
    exit /b 1
)

echo ✅ 服务启动成功！
echo.
echo 服务信息:
echo - 访问地址: http://localhost:8060
echo - 健康检查: http://localhost:8060/health
echo.
echo 管理命令:
echo - 查看日志: docker compose -f docker-compose.openvino.yml logs -f
echo - 停止服务: docker compose -f docker-compose.openvino.yml down
echo - 重启服务: docker compose -f docker-compose.openvino.yml restart
echo.
echo 正在检查服务状态...
timeout /t 5 /nobreak >nul

REM 检查服务是否正常运行
curl -s http://localhost:8060/health >nul 2>&1
if errorlevel 1 (
    echo ⚠️  服务可能还在启动中，请稍等片刻后访问
) else (
    echo ✅ 服务运行正常！
)

echo.
pause