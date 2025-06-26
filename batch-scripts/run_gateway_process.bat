@echo off
echo 正在处理 gate_way_logs 目录下的日志文件...
echo 输出将保存到 DATA_OUTPUT 目录
cd /d "C:\Users\30871\Desktop\AILogForSpring\logsense-xpu\filter"
python clean_and_filter_logs.py --dir "..\data\gate_way_logs" --output-dir "..\..\DATA_OUTPUT"
pause 