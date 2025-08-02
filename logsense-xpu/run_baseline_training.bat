@echo off
echo ========================================
echo Baseline模型训练脚本
echo ========================================

REM 设置Python环境
set PYTHONPATH=%PYTHONPATH%;%CD%\core

REM 检查Intel XPU环境
echo 检查Intel XPU环境...
python -c "import torch; print('PyTorch版本:', torch.__version__); import intel_extension_for_pytorch as ipex; print('Intel XPU可用:', torch.xpu.is_available())" 2>nul
if errorlevel 1 (
    echo 警告: Intel XPU环境未正确配置，将使用CPU模式
    set USE_XPU=
) else (
    echo Intel XPU环境已配置
    set USE_XPU=--use_xpu
)

echo.
echo 开始训练LightGBM模型...
python train_baseline.py --model_type lightgbm --data_dir ../../DATA_OUTPUT --output_dir results %USE_XPU%

echo.
echo 开始训练FastText模型...
python train_baseline.py --model_type fasttext --data_dir ../../DATA_OUTPUT --output_dir results %USE_XPU%

echo.
echo 训练完成！
pause 