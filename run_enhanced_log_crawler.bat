@echo off
chcp 65001 >nul
echo.
echo ========================================
echo 🚀 Spring Boot 日志抓取工具 - 增强版
echo ========================================
echo.
echo 📋 功能特点:
echo   • 按类别抓取不同Spring Boot项目
echo   • 增强的日志识别算法
echo   • 支持多种应用场景
echo   • 详细的统计报告
echo.
echo 🎯 目标类别:
echo   • 电商/业务系统
echo   • 微服务架构
echo   • 安全/权限系统
echo   • 数据处理/ETL
echo   • 消息队列集成
echo   • 数据库驱动集成
echo   • 分布式事务
echo.
echo ⏰ 开始时间: %date% %time%
echo.

python issue-helper-enhanced.py

echo.
echo ⏰ 结束时间: %date% %time%
echo.
echo 💡 提示:
echo   • 检查 issue-logs 目录中的CSV文件
echo   • 使用 log_reviewer.py 进行日志标注
echo   • 将标注数据用于模型训练
echo.
pause
