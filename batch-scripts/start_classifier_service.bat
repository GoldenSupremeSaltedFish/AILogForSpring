@echo off
REM 日志分类器服务启动脚本
REM 使用方法: start_classifier_service.bat [模式] [选项]

setlocal enabledelayedexpansion

REM 设置默认参数
set MODE=api
set INPUT_FILE=
set INPUT_DIR=
set OUTPUT_FILE=
set OUTPUT_DIR=
set CONFIG=
set HOST=0.0.0.0
set PORT=5000
set NO_ML=
set DEBUG=

REM 解析命令行参数
:parse_args
if "%~1"=="" goto :run_service
if "%~1"=="--mode" (
    set MODE=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--input-file" (
    set INPUT_FILE=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--input-dir" (
    set INPUT_DIR=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--output-file" (
    set OUTPUT_FILE=%~2
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
if "%~1"=="--config" (
    set CONFIG=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--host" (
    set HOST=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--port" (
    set PORT=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--no-ml" (
    set NO_ML=--no-ml
    shift
    goto :parse_args
)
if "%~1"=="--debug" (
    set DEBUG=--debug
    shift
    goto :parse_args
)
if "%~1"=="--help" (
    goto :show_help
)
if "%~1"=="-h" (
    goto :show_help
)

REM 如果没有指定参数，尝试使用第一个参数作为模式
if "%MODE%"=="api" (
    set MODE=%~1
    shift
    goto :parse_args
)

shift
goto :parse_args

:show_help
echo.
echo 日志分类器服务启动脚本
echo ==========================
echo.
echo 使用方法:
echo   start_classifier_service.bat [选项]
echo.
echo 运行模式:
echo   api          - 启动API服务 (默认)
echo   file         - 单文件分类
echo   batch        - 批量分类
echo   interactive  - 交互式分类
echo.
echo 选项:
echo   --mode MODE              运行模式: api, file, batch, interactive
echo   --input-file FILE        输入文件路径 (file模式)
echo   --input-dir DIR          输入目录路径 (batch模式)
echo   --output-file FILE       输出文件路径
echo   --output-dir DIR         输出目录路径
echo   --config FILE            配置文件路径
echo   --host HOST              API服务主机地址 (默认: 0.0.0.0)
echo   --port PORT              API服务端口 (默认: 5000)
echo   --no-ml                  不使用机器学习分类
echo   --debug                  调试模式
echo   --help, -h               显示此帮助信息
echo.
echo 示例:
echo   start_classifier_service.bat --mode api
echo   start_classifier_service.bat --mode file --input-file logs.csv
echo   start_classifier_service.bat --mode batch --input-dir logs/
echo   start_classifier_service.bat --mode interactive
echo.
goto :end

:run_service
echo.
echo ==========================================
echo 日志分类器服务
echo ==========================================
echo.

REM 检查Python是否可用
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python，请确保Python已安装并添加到PATH
    goto :end
)

REM 检查必要文件
if not exist "automated_log_classifier.py" (
    echo 错误: 未找到 automated_log_classifier.py
    goto :end
)

if not exist "start_classifier_service.py" (
    echo 错误: 未找到 start_classifier_service.py
    goto :end
)

REM 构建Python命令
set PYTHON_CMD=python start_classifier_service.py --mode %MODE%

if not "%INPUT_FILE%"=="" (
    set PYTHON_CMD=%PYTHON_CMD% --input-file "%INPUT_FILE%"
)

if not "%INPUT_DIR%"=="" (
    set PYTHON_CMD=%PYTHON_CMD% --input-dir "%INPUT_DIR%"
)

if not "%OUTPUT_FILE%"=="" (
    set PYTHON_CMD=%PYTHON_CMD% --output-file "%OUTPUT_FILE%"
)

if not "%OUTPUT_DIR%"=="" (
    set PYTHON_CMD=%PYTHON_CMD% --output-dir "%OUTPUT_DIR%"
)

if not "%CONFIG%"=="" (
    set PYTHON_CMD=%PYTHON_CMD% --config "%CONFIG%"
)

if not "%HOST%"=="0.0.0.0" (
    set PYTHON_CMD=%PYTHON_CMD% --host %HOST%
)

if not "%PORT%"=="5000" (
    set PYTHON_CMD=%PYTHON_CMD% --port %PORT%
)

if not "%NO_ML%"=="" (
    set PYTHON_CMD=%PYTHON_CMD% %NO_ML%
)

if not "%DEBUG%"=="" (
    set PYTHON_CMD=%PYTHON_CMD% %DEBUG%
)

REM 显示执行信息
echo 执行信息:
echo   模式: %MODE%
if not "%INPUT_FILE%"=="" (
    echo   输入文件: %INPUT_FILE%
)
if not "%INPUT_DIR%"=="" (
    echo   输入目录: %INPUT_DIR%
)
if not "%OUTPUT_FILE%"=="" (
    echo   输出文件: %OUTPUT_FILE%
)
if not "%OUTPUT_DIR%"=="" (
    echo   输出目录: %OUTPUT_DIR%
)
if not "%HOST%"=="0.0.0.0" (
    echo   主机: %HOST%
)
if not "%PORT%"=="5000" (
    echo   端口: %PORT%
)
echo   选项: %NO_ML% %DEBUG%
echo.

REM 执行Python脚本
echo 正在启动服务...
echo.
%PYTHON_CMD%

REM 检查执行结果
if errorlevel 1 (
    echo.
    echo 错误: 服务启动失败
    echo 请检查错误信息并重试
) else (
    echo.
    echo 服务已停止
)

:end
echo.
pause
