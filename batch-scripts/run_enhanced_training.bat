@echo off
chcp 65001 >nul
echo ========================================
echo 增强版模型训练
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
    pip install pandas scikit-learn matplotlib seaborn joblib
    if errorlevel 1 (
        echo ❌ 安装依赖包失败
        pause
        exit /b 1
    )
)

echo ✅ 环境检查完成
echo.

REM 运行增强训练
echo 🚀 开始增强版模型训练...
python logsense-gpu/scripts/enhanced_training.py --data-file DATA_OUTPUT/training_data/combined_dataset_20250802_131542.csv

echo.
echo ========================================
echo 增强训练完成！
echo ========================================
pause 