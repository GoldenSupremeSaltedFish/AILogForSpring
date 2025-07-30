@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

set "PROJECT_ROOT=%~dp0.."
set "SCRIPT_PATH=%PROJECT_ROOT%\log-processing\quality_analyzer.py"

echo ========================================
echo 📊 日志分类质量分析工具
echo ========================================
echo.

echo 请选择分析模式:
echo 1. 单文件质量分析
echo 2. 双文件对比分析
echo 3. 批量分析DATA_OUTPUT目录
echo 4. 分析特定项目的分类结果
echo.
set /p choice=请输入选择 (1-4): 

if "%choice%"=="1" goto :single_analysis
if "%choice%"=="2" goto :compare_analysis
if "%choice%"=="3" goto :batch_analysis
if "%choice%"=="4" goto :project_analysis
goto :invalid_choice

:single_analysis
echo.
set /p input_file=请输入CSV文件路径: 
if "%input_file%"=="" (
    echo ❌ 文件路径不能为空
    pause
    exit /b 1
)

echo.
echo 🚀 开始分析: %input_file%
cd /d "%PROJECT_ROOT%"
python "%SCRIPT_PATH%" analyze --file "%input_file%"
goto :end

:compare_analysis
echo.
set /p file1=请输入第一个CSV文件路径: 
set /p file2=请输入第二个CSV文件路径: 
if "%file1%"=="" (
    echo ❌ 第一个文件路径不能为空
    pause
    exit /b 1
)
if "%file2%"=="" (
    echo ❌ 第二个文件路径不能为空
    pause
    exit /b 1
)

echo.
echo 🔄 开始对比分析...
cd /d "%PROJECT_ROOT%"
python "%SCRIPT_PATH%" compare --file1 "%file1%" --file2 "%file2%"
goto :end

:batch_analysis
echo.
echo 🔍 扫描DATA_OUTPUT目录下的分类文件...
set count=0
for /r "%PROJECT_ROOT%\DATA_OUTPUT" %%f in (*_classified.csv) do (
    set /a count+=1
    echo !count!. %%f
    echo 正在分析...
    cd /d "%PROJECT_ROOT%"
    python "%SCRIPT_PATH%" analyze --file "%%f"
    echo.
)
echo 批量分析完成，共处理 !count! 个文件
goto :end

:project_analysis
echo.
echo 🔍 扫描可用的项目分类结果...
set count=0
for /d %%d in ("%PROJECT_ROOT%\DATA_OUTPUT\*") do (
    set /a count+=1
    echo !count!. %%~nd
)

if !count!==0 (
    echo ❌ 未找到项目分类结果
    pause
    exit /b 1
)

echo.
set /p project_choice=请选择要分析的项目编号: 
set current_count=0
for /d %%d in ("%PROJECT_ROOT%\DATA_OUTPUT\*") do (
    set /a current_count+=1
    if !current_count!==!project_choice! (
        echo 分析项目: %%~nd
        for /r "%%d" %%f in (*_classified.csv) do (
            echo 正在分析: %%f
            cd /d "%PROJECT_ROOT%"
            python "%SCRIPT_PATH%" analyze --file "%%f"
        )
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
echo ✅ 分析完成
echo ================================
echo.
echo 💡 提示:
echo - 分析报告保存在输入文件同目录
echo - 包含文本报告、JSON数据和可视化图表
echo - 可用于质量评估和改进决策
echo.
pause