@echo off
chcp 65001 >nul
echo ================================
echo 🚀 半自动日志标签辅助器
echo ================================
echo.

:: 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 错误：未找到Python，请先安装Python
    pause
    exit /b 1
)

:: 设置脚本目录
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."
set "LABELER_SCRIPT=%PROJECT_ROOT%\log-processing\auto_labeler.py"

:: 检查脚本是否存在
if not exist "%LABELER_SCRIPT%" (
    echo ❌ 错误：找不到auto_labeler.py脚本
    echo 路径：%LABELER_SCRIPT%
    pause
    exit /b 1
)

echo 📁 项目根目录：%PROJECT_ROOT%
echo 📄 标签器脚本：%LABELER_SCRIPT%
echo.

:: 显示选项菜单
echo 请选择操作模式：
echo [1] 运行演示示例
echo [2] 处理指定CSV文件（仅规则分类）
echo [3] 处理指定CSV文件（使用机器学习）
echo [4] 自动处理DATA_OUTPUT目录中的最新CSV文件
echo [0] 退出
echo.

set /p choice=请输入选择 (0-4): 

if "%choice%"=="0" goto :end
if "%choice%"=="1" goto :demo
if "%choice%"=="2" goto :rule_only
if "%choice%"=="3" goto :with_ml
if "%choice%"=="4" goto :auto_process

echo ❌ 无效选择，请重新运行
pause
exit /b 1

:demo
echo.
echo 🎯 运行演示示例...
cd /d "%PROJECT_ROOT%\log-processing"
python example_usage.py
goto :end

:rule_only
echo.
set /p input_file=请输入CSV文件路径: 
if "%input_file%"=="" (
    echo ❌ 错误：文件路径不能为空
    pause
    exit /b 1
)

echo.
echo 🔍 使用规则分类处理：%input_file%
cd /d "%PROJECT_ROOT%\log-processing"
python auto_labeler.py "%input_file%"
goto :end

:with_ml
echo.
set /p input_file=请输入CSV文件路径: 
if "%input_file%"=="" (
    echo ❌ 错误：文件路径不能为空
    pause
    exit /b 1
)

set /p train_file=请输入训练数据路径（可选，直接回车跳过）: 

echo.
if "%train_file%"=="" (
    echo 🤖 使用机器学习处理（无训练数据）：%input_file%
    cd /d "%PROJECT_ROOT%\log-processing"
    python auto_labeler.py "%input_file%" --use-ml
) else (
    echo 🤖 使用机器学习处理（有训练数据）：%input_file%
    cd /d "%PROJECT_ROOT%\log-processing"
    python auto_labeler.py "%input_file%" --use-ml --train-data "%train_file%"
)
goto :end

:auto_process
echo.
echo 🔍 扫描DATA_OUTPUT目录...

:: 查找最新的CSV文件
set "DATA_OUTPUT_DIR=%PROJECT_ROOT%\DATA_OUTPUT"
if not exist "%DATA_OUTPUT_DIR%" (
    echo ❌ 错误：DATA_OUTPUT目录不存在
    pause
    exit /b 1
)

:: 使用PowerShell查找最新的CSV文件
for /f "delims=" %%f in ('powershell -command "Get-ChildItem -Path '%DATA_OUTPUT_DIR%' -Recurse -Filter '*.csv' | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName"') do set "latest_csv=%%f"

if "%latest_csv%"=="" (
    echo ❌ 错误：DATA_OUTPUT目录中没有找到CSV文件
    pause
    exit /b 1
)

echo 📁 找到最新文件：%latest_csv%
echo.
echo 🔍 使用规则分类处理...
cd /d "%PROJECT_ROOT%\log-processing"
python auto_labeler.py "%latest_csv%"
goto :end

:end
echo.
echo ================================
echo ✅ 操作完成
echo ================================
echo.
echo 💡 提示：
echo - 检查生成的 *_labeled_*.csv 文件
echo - 查看 *_summary.txt 文件了解分类结果
echo - 人工校正后可用于机器学习训练
echo.
pause 