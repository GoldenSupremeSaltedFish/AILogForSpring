@echo off
chcp 65001 >nul
echo.
echo ========================================
echo 🧪 增强版日志抓取工具 - 功能测试
echo ========================================
echo.
echo 📋 测试内容:
echo   • 配置文件加载测试
echo   • 日志识别功能测试
echo   • 文本清理功能测试
echo   • 输出目录权限测试
echo.
echo ⏰ 开始时间: %date% %time%
echo.

python test_enhanced_crawler.py

echo.
echo ⏰ 结束时间: %date% %time%
echo.
echo 💡 测试结果说明:
echo   • 如果所有测试都通过，可以运行完整抓取
echo   • 如果有测试失败，请检查相关配置
echo.
pause
