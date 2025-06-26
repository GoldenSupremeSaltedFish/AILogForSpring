# 批量重命名gateway日志文件脚本
# 将数字后缀的文件改为.log后缀

$logDir = "data\gate_way_logs"
Write-Host "🚀 开始处理目录: $logDir" -ForegroundColor Green

# 获取所有需要重命名的文件（数字后缀的文件）
$filesToRename = Get-ChildItem -Path $logDir -File | Where-Object { 
    $_.Name -match '\.(\d+)$' -and $_.Extension -ne '.log'
}

Write-Host "📊 找到 $($filesToRename.Count) 个需要重命名的文件" -ForegroundColor Yellow

if ($filesToRename.Count -eq 0) {
    Write-Host "✅ 没有需要重命名的文件" -ForegroundColor Green
    exit 0
}

# 显示将要重命名的文件
Write-Host "`n📋 将要重命名的文件:" -ForegroundColor Cyan
$filesToRename | ForEach-Object {
    $newName = $_.Name + ".log"
    Write-Host "  $($_.Name) -> $newName" -ForegroundColor White
}

# 确认重命名
Write-Host "`n❓ 确认重命名这些文件吗? (y/N): " -NoNewline -ForegroundColor Yellow
$confirmation = Read-Host

if ($confirmation -eq 'y' -or $confirmation -eq 'Y') {
    Write-Host "`n🔄 开始重命名..." -ForegroundColor Green
    
    $successCount = 0
    $errorCount = 0
    
    foreach ($file in $filesToRename) {
        try {
            $newName = $file.Name + ".log"
            $newPath = Join-Path $file.Directory $newName
            
            # 检查目标文件是否已存在
            if (Test-Path $newPath) {
                Write-Host "⚠️  跳过 $($file.Name): 目标文件 $newName 已存在" -ForegroundColor Yellow
                continue
            }
            
            # 重命名文件
            Rename-Item -Path $file.FullName -NewName $newName
            Write-Host "✅ $($file.Name) -> $newName" -ForegroundColor Green
            $successCount++
        }
        catch {
            Write-Host "❌ 重命名失败 $($file.Name): $($_.Exception.Message)" -ForegroundColor Red
            $errorCount++
        }
    }
    
    Write-Host "`n📊 重命名完成!" -ForegroundColor Green
    Write-Host "✅ 成功: $successCount 个文件" -ForegroundColor Green
    if ($errorCount -gt 0) {
        Write-Host "❌ 失败: $errorCount 个文件" -ForegroundColor Red
    }
}
else {
    Write-Host "`n❌ 操作已取消" -ForegroundColor Red
} 