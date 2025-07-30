@echo off
chcp 65001 > nul
echo ========================================
echo     增强日志预分类器
echo ========================================
echo.

cd /d "%~dp0.."

echo 选择处理模式：
echo 1. 批量处理 dataset-ready 目录
echo 2. 批量处理自定义目录
echo 3. 处理单个日志文件
echo 4. 显示帮助
echo.
set /p choice="请输入选项 (1-4): "

if "%choice%"=="1" (
    echo 正在批量处理 dataset-ready 目录...
    python log-processing\enhanced_pre_classifier.py batch --input-dir "dataset-ready" --output-dir "DATA_OUTPUT"
) else if "%choice%"=="2" (
    set /p input_dir="请输入日志目录路径: "
    python log-processing\enhanced_pre_classifier.py batch --input-dir "!input_dir!" --output-dir "DATA_OUTPUT"
) else if "%choice%"=="3" (
    set /p input_file="请输入日志文件路径: "
    python log-processing\enhanced_pre_classifier.py single --input-file "!input_file!" --output-dir "DATA_OUTPUT"
) else if "%choice%"=="4" (
    python log-processing\enhanced_pre_classifier.py --help
) else (
    echo 无效选项！
)

echo.
echo 按任意键退出...
pause > nul