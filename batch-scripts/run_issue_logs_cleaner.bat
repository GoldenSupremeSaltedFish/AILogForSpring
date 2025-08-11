@echo off
chcp 65001 >nul
echo ========================================
echo Issue日志数据清洗器
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

REM 运行Issue日志数据清洗器
echo 🚀 开始Issue日志数据清洗...
echo 📁 输入目录: issue-logs
echo 📁 输出目录: DATA_OUTPUT
echo.

python issue_logs_cleaner.py --combined --max-per-class 500

echo.
echo ========================================
echo Issue日志数据清洗完成！
echo ========================================
echo.
echo 💡 下一步建议:
echo   1. 检查DATA_OUTPUT目录中的清洗结果
echo   2. 使用log_reviewer.py进行人工标注
echo   3. 将清洗后的数据用于模型训练
echo.
pause
