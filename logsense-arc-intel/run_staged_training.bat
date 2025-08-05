@echo off
chcp 65001 >nul
echo 🎯 Intel Arc GPU 分阶段训练启动器
echo ==========================================

echo.
echo 📋 训练计划:
echo   阶段1: 小数据集快速验证 (3 epochs)
echo   阶段2: 完整数据集正式训练 (10 epochs)
echo.

echo 🔍 检查环境...
python quick_start.py

echo.
echo 📂 准备分阶段数据...
python scripts/prepare_data_staged.py

echo.
echo 🔬 阶段1: 小数据集快速验证...
python staged_training.py --skip_large

echo.
echo 🚀 阶段2: 完整数据集正式训练...
python staged_training.py --skip_small --skip_data_prep

echo.
echo ✅ 分阶段训练完成！
echo 📁 模型保存在:
echo    results/models_small/  (小数据集模型)
echo    results/models_large/  (完整数据集模型)
pause 