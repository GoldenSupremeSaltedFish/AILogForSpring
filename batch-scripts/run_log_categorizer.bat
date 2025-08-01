@echo off
chcp 65001 >nul
echo ========================================
echo 日志分类整理器
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
python -c "import pandas" >nul 2>&1
if errorlevel 1 (
    echo ❌ pandas包未安装，正在安装...
    pip install pandas
    if errorlevel 1 (
        echo ❌ 安装pandas失败
        pause
        exit /b 1
    )
)

echo ✅ 环境检查完成
echo.

REM 运行日志分类整理器
echo 🚀 开始处理日志文件...
python log-processing/log_categorizer.py

echo.
echo ========================================
echo 处理完成！
echo ========================================
pause 