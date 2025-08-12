@echo off
chcp 65001 >nul
echo 🚀 本地配置设置工具
echo ================================================
echo.

echo 📋 正在设置本地配置...
python local_config_setup.py

echo.
echo ================================================
echo ✅ 设置完成！
echo.
echo 📝 重要提醒：
echo 1. 所有配置文件已添加到 .gitignore
echo 2. 数据文件不会被上传到Git仓库
echo 3. 请设置环境变量 GITHUB_TOKEN
echo.
echo 按任意键退出...
pause >nul
