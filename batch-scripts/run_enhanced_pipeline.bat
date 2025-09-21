@echo off
REM 增强的日志半自动分类流水线批处理脚本
REM 使用方法: run_enhanced_pipeline.bat [模式] [输入文件/目录] [选项]

setlocal enabledelayedexpansion

REM 设置默认参数
set MODE=full
set INPUT_PATH=
set SKIP_HUMAN_REVIEW=
set SKIP_TEMPLATING=
set SKIP_FEATURE_ENGINEERING=
set SKIP_ML=
set SKIP_QUALITY_ANALYSIS=
set OUTPUT_DIR=

REM 解析命令行参数
:parse_args
if "%~1"=="" goto :run_pipeline
if "%~1"=="--mode" (
    set MODE=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--input-file" (
    set INPUT_PATH=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--input-dir" (
    set INPUT_PATH=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--output-dir" (
    set OUTPUT_DIR=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--skip-human-review" (
    set SKIP_HUMAN_REVIEW=--skip-human-review
    shift
    goto :parse_args
)
if "%~1"=="--skip-templating" (
    set SKIP_TEMPLATING=--skip-templating
    shift
    goto :parse_args
)
if "%~1"=="--skip-feature-engineering" (
    set SKIP_FEATURE_ENGINEERING=--skip-feature-engineering
    shift
    goto :parse_args
)
if "%~1"=="--skip-ml" (
    set SKIP_ML=--skip-ml
    shift
    goto :parse_args
)
if "%~1"=="--skip-quality-analysis" (
    set SKIP_QUALITY_ANALYSIS=--skip-quality-analysis
    shift
    goto :parse_args
)
if "%~1"=="--help" (
    goto :show_help
)
if "%~1"=="-h" (
    goto :show_help
)

REM 如果没有指定参数，尝试使用第一个参数作为输入路径
if "%INPUT_PATH%"=="" (
    set INPUT_PATH=%~1
    shift
    goto :parse_args
)

shift
goto :parse_args

:show_help
echo.
echo 增强的日志半自动分类流水线批处理脚本
echo ==========================================
echo.
echo 使用方法:
echo   run_enhanced_pipeline.bat [选项]
echo.
echo 选项:
echo   --mode MODE              运行模式: full, template-only, batch
echo   --input-file FILE        输入日志文件路径
echo   --input-dir DIR          输入目录路径（批量模式）
echo   --output-dir DIR         输出目录路径
echo   --skip-human-review      跳过人工审查步骤
echo   --skip-templating        跳过模板化步骤
echo   --skip-feature-engineering 跳过特征工程步骤
echo   --skip-ml                跳过机器学习分类
echo   --skip-quality-analysis  跳过质量分析
echo   --help, -h               显示此帮助信息
echo.
echo 示例:
echo   run_enhanced_pipeline.bat --input-file logs.csv --mode full
echo   run_enhanced_pipeline.bat --input-dir logs/ --mode batch --skip-human-review
echo   run_enhanced_pipeline.bat --input-file logs.csv --mode template-only
echo.
goto :end

:run_pipeline
echo.
echo ==========================================
echo 增强的日志半自动分类流水线
echo ==========================================
echo.

REM 检查Python是否可用
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python，请确保Python已安装并添加到PATH
    goto :end
)

REM 检查输入路径
if "%INPUT_PATH%"=="" (
    echo 错误: 未指定输入文件或目录
    echo 使用 --help 查看帮助信息
    goto :end
)

REM 检查输入路径是否存在
if not exist "%INPUT_PATH%" (
    echo 错误: 输入路径不存在: %INPUT_PATH%
    goto :end
)

REM 构建Python命令
set PYTHON_CMD=python log-processing\enhanced_pipeline.py --mode %MODE%

if "%MODE%"=="full" (
    set PYTHON_CMD=%PYTHON_CMD% --input-file "%INPUT_PATH%"
) else if "%MODE%"=="template-only" (
    set PYTHON_CMD=%PYTHON_CMD% --input-file "%INPUT_PATH%"
) else if "%MODE%"=="batch" (
    set PYTHON_CMD=%PYTHON_CMD% --input-dir "%INPUT_PATH%"
) else (
    echo 错误: 无效的运行模式: %MODE%
    echo 支持的模式: full, template-only, batch
    goto :end
)

if not "%OUTPUT_DIR%"=="" (
    set PYTHON_CMD=%PYTHON_CMD% --output-dir "%OUTPUT_DIR%"
)

if not "%SKIP_HUMAN_REVIEW%"=="" (
    set PYTHON_CMD=%PYTHON_CMD% %SKIP_HUMAN_REVIEW%
)

if not "%SKIP_TEMPLATING%"=="" (
    set PYTHON_CMD=%PYTHON_CMD% %SKIP_TEMPLATING%
)

if not "%SKIP_FEATURE_ENGINEERING%"=="" (
    set PYTHON_CMD=%PYTHON_CMD% %SKIP_FEATURE_ENGINEERING%
)

if not "%SKIP_ML%"=="" (
    set PYTHON_CMD=%PYTHON_CMD% %SKIP_ML%
)

if not "%SKIP_QUALITY_ANALYSIS%"=="" (
    set PYTHON_CMD=%PYTHON_CMD% %SKIP_QUALITY_ANALYSIS%
)

REM 显示执行信息
echo 执行信息:
echo   模式: %MODE%
echo   输入: %INPUT_PATH%
if not "%OUTPUT_DIR%"=="" (
    echo   输出: %OUTPUT_DIR%
)
echo   跳过步骤: %SKIP_HUMAN_REVIEW% %SKIP_TEMPLATING% %SKIP_FEATURE_ENGINEERING% %SKIP_ML% %SKIP_QUALITY_ANALYSIS%
echo.

REM 执行Python脚本
echo 正在执行流水线...
echo.
%PYTHON_CMD%

REM 检查执行结果
if errorlevel 1 (
    echo.
    echo 错误: 流水线执行失败
    echo 请检查错误信息并重试
) else (
    echo.
    echo 流水线执行完成！
    echo 请查看输出目录中的结果文件
)

:end
echo.
pause
