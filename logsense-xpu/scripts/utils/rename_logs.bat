@echo off
echo 开始重命名gateway日志文件...
cd /d "data\gate_way_logs"

echo 重命名以数字结尾的文件为.log后缀：

for %%f in (*.1) do (
    if not exist "%%f.log" (
        ren "%%f" "%%f.log"
        echo 重命名: %%f -^> %%f.log
    ) else (
        echo 跳过: %%f ^(目标文件已存在^)
    )
)

for %%f in (*.2) do (
    if not exist "%%f.log" (
        ren "%%f" "%%f.log"
        echo 重命名: %%f -^> %%f.log
    ) else (
        echo 跳过: %%f ^(目标文件已存在^)
    )
)

for %%f in (*.3) do (
    if not exist "%%f.log" (
        ren "%%f" "%%f.log"
        echo 重命名: %%f -^> %%f.log
    ) else (
        echo 跳过: %%f ^(目标文件已存在^)
    )
)

for %%f in (*.4) do (
    if not exist "%%f.log" (
        ren "%%f" "%%f.log"
        echo 重命名: %%f -^> %%f.log
    ) else (
        echo 跳过: %%f ^(目标文件已存在^)
    )
)

for %%f in (*.5) do (
    if not exist "%%f.log" (
        ren "%%f" "%%f.log"
        echo 重命名: %%f -^> %%f.log
    ) else (
        echo 跳过: %%f ^(目标文件已存在^)
    )
)

for %%f in (*.6) do (
    if not exist "%%f.log" (
        ren "%%f" "%%f.log"
        echo 重命名: %%f -^> %%f.log
    ) else (
        echo 跳过: %%f ^(目标文件已存在^)
    )
)

for %%f in (*.7) do (
    if not exist "%%f.log" (
        ren "%%f" "%%f.log"
        echo 重命名: %%f -^> %%f.log
    ) else (
        echo 跳过: %%f ^(目标文件已存在^)
    )
)

for %%f in (*.8) do (
    if not exist "%%f.log" (
        ren "%%f" "%%f.log"
        echo 重命名: %%f -^> %%f.log
    ) else (
        echo 跳过: %%f ^(目标文件已存在^)
    )
)

for %%f in (*.9) do (
    if not exist "%%f.log" (
        ren "%%f" "%%f.log"
        echo 重命名: %%f -^> %%f.log
    ) else (
        echo 跳过: %%f ^(目标文件已存在^)
    )
)

for %%f in (*.10) do (
    if not exist "%%f.log" (
        ren "%%f" "%%f.log"
        echo 重命名: %%f -^> %%f.log
    ) else (
        echo 跳过: %%f ^(目标文件已存在^)
    )
)

echo.
echo 重命名完成！
pause 