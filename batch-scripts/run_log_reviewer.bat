@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

set "PROJECT_ROOT=%~dp0.."
set "SCRIPT_PATH=%PROJECT_ROOT%\log-processing\log_reviewer.py"

echo ========================================
echo 📋 日志标签审查工具
echo ========================================
echo.

echo 请选择操作模式:
echo 1. 审查单个文件
echo 2. 批量审查目录
echo 3. 继续上次审查
echo.
set /p choice=请输入选择 (1-3): 

if "%choice%"=="1" goto :single_file
if "%choice%"=="2" goto :batch_review
if "%choice%"=="3" goto :continue_review
goto :invalid_choice

:single_file
echo.
set /p input_file=请输入CSV文件路径: 
if "%input_file%"=="" (
    echo ❌ 文件路径不能为空
    pause
    exit /b 1
)

echo.
echo 🚀 开始审查: %input_file%
cd /d "%PROJECT_ROOT%"
python "%SCRIPT_PATH%" "%input_file%"
goto :end

:batch_review
echo.
set /p input_dir=请输入包含标注文件的目录: 
if "%input_dir%"=="" (
    echo ❌ 目录路径不能为空
    pause
    exit /b 1
)

echo.
echo 🔍 扫描目录: %input_dir%
for %%f in ("%input_dir%\*_labeled_*.csv") do (
    echo 找到文件: %%f
    echo 开始审查...
    cd /d "%PROJECT_ROOT%"
    python "%SCRIPT_PATH%" "%%f"
    echo.
)
goto :end

:continue_review
echo.
echo 🔍 查找未完成的审查...
for /r "%PROJECT_ROOT%" %%f in (*_review_progress.json) do (
    echo 找到进度文件: %%f
    set "progress_file=%%f"
    set "csv_file=%%~dpnf"
    set "csv_file=!csv_file:_review_progress=!"
    set "csv_file=!csv_file!.csv"
    
    if exist "!csv_file!" (
        echo 继续审查: !csv_file!
        cd /d "%PROJECT_ROOT%"
        python "%SCRIPT_PATH%" "!csv_file!"
    )
)
goto :end

:invalid_choice
echo ❌ 无效选择
pause
exit /b 1

:end
echo.
echo ================================
echo ✅ 操作完成
echo ================================
echo.
echo 💡 提示:
echo - 审查结果保存在输入文件同目录
echo - 进度文件支持断点续传
echo - 使用 Ctrl+C 可安全退出
echo.
pause