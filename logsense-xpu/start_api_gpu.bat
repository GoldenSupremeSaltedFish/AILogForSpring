@echo off
echo ========================================
echo 日志分类API服务器 - GPU模式
echo ========================================

REM 设置Python环境
set PYTHONPATH=%PYTHONPATH%;%CD%\core

REM 检查依赖
echo 检查依赖包...
python -c "import flask, flask_cors, lightgbm, joblib, numpy, pandas, torch" 2>nul
if errorlevel 1 (
    echo 安装依赖包...
    pip install flask flask-cors requests
    if errorlevel 1 (
        echo 依赖安装失败，请手动安装
        pause
        exit /b 1
    )
)

REM 检查模型文件
echo 检查模型文件...
if not exist "models\lightgbm_model_*.txt" (
    echo 错误: 未找到模型文件，请先运行训练脚本
    pause
    exit /b 1
)

echo 模型文件检查通过

echo.
echo 启动API服务器 (GPU模式)...
echo 服务器地址: http://localhost:5000
echo 设备模式: GPU (Intel Arc A750)
echo.
echo 按 Ctrl+C 停止服务器
echo.

python api_server_enhanced.py --device gpu

pause 