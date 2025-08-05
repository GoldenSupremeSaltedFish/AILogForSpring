@echo off
chcp 65001 >nul
echo 🚀 Intel Arc GPU 训练启动器
echo ==========================================

echo.
echo 🔍 检查环境...
python quick_start.py

echo.
echo 📂 准备训练数据...
python scripts/prepare_data.py

echo.
echo 🎯 开始训练...
python start_training.py

echo.
echo ✅ 训练完成！
pause 