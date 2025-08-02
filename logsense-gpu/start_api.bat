@echo off
chcp 65001 >nul
echo 🚀 启动LogSense API服务器
echo ================================

REM 检查Python环境
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 未找到Python环境，请先安装Python
    pause
    exit /b 1
)

REM 检查依赖
echo 📦 检查依赖包...
pip install -r requirements.txt >nul 2>&1
if errorlevel 1 (
    echo ❌ 依赖安装失败
    pause
    exit /b 1
)

REM 启动API服务器
echo 🌐 启动API服务器...
echo 📍 服务器地址: http://localhost:5000
echo 📍 健康检查: http://localhost:5000/health
echo 📍 模型信息: http://localhost:5000/model/info
echo 📍 预测接口: http://localhost:5000/predict
echo.
echo ⚠️  按 Ctrl+C 停止服务器
echo ================================

python api_server.py

pause 