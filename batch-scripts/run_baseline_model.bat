@echo off
chcp 65001 >nul
echo ========================================
echo 小样本验证 + Base模型训练
echo ========================================
echo.

REM 切换到项目根目录
cd /d "%~dp0.."

REM 检查Python是否可用
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python未安装或不在PATH中
    pause
    exit /b 1
)

REM 检查必要的Python包
echo 🔍 检查依赖包...
python -c "import pandas, sklearn, matplotlib, seaborn" >nul 2>&1
if errorlevel 1 (
    echo ❌ 缺少必要的包，正在安装...
    pip install pandas scikit-learn matplotlib seaborn
    if errorlevel 1 (
        echo ❌ 安装依赖包失败
        pause
        exit /b 1
    )
)

REM 检查LightGBM（可选）
python -c "import lightgbm" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  LightGBM未安装，将使用GradientBoosting
    echo 如需使用LightGBM，请运行: pip install lightgbm
)

REM 检查PyTorch（用于GPU检测）
python -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  PyTorch未安装，无法检测GPU
    echo 如需GPU支持，请运行: pip install torch
)

echo ✅ 环境检查完成
echo.

REM 运行平台检测
echo 🔍 检测系统环境...
python logsense-gpu/utils/platform_utils.py

echo.

REM 运行baseline模型训练
echo 🚀 开始小样本验证实验...
python logsense-gpu/scripts/baseline_model.py --sample-size 500 --model-type gradient_boosting

echo.
echo ========================================
echo 实验完成！
echo ========================================
pause 