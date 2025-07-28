import requests
import time
import re
import json
import os
import csv
from datetime import datetime
from tqdm import tqdm

# 加载配置
def load_config():
    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ 配置文件 config.json 不存在，请先创建配置文件")
        return None
    except json.JSONDecodeError:
        print("❌ 配置文件格式错误")
        return None

# 判断是否是日志的代码段
def is_log_text(text):
    log_keywords = [
        'Exception', 'Caused by', 'at ', 'ERROR', 'WARN', 'INFO', 'DEBUG',
        'TRACE', 'FATAL', 'Stack trace', 'java.lang.', 'org.springframework.',
        'Stacktrace:', 'Error:', 'Failed to', 'Cannot', 'Unable to',
        'NullPointerException', 'IllegalArgumentException', 'RuntimeException'
    ]
    return any(k in text for k in log_keywords)

# 清理和标准化日志文本
def clean_log_text(text):
    # 移除代码块标记
    text = re.sub(r'^```[\w]*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n?```$', '', text, flags=re.MULTILINE)
    
    # 移除多余的空行
    text = re.sub(r'\n\s*\n', '\n', text)
    
    # 移除首尾空白
    text = text.strip()
    
    return text

# 获取所有 issue 列表
def fetch_issues(owner_repo, headers, max_pages, state="all"):
    issues = []
    print(f"\n📥 正在爬取仓库: {owner_repo}")
    print(f"🔍 目标页数: {max_pages} (每页最多100个issues)")
    
    for page in tqdm(range(1, max_pages + 1), desc=f"获取 {owner_repo} issues"):
        url = f"https://api.github.com/repos/{owner_repo}/issues?page={page}&per_page=100&state={state}"
        
        print(f"\n📡 请求第 {page} 页: {url}")
        
        try:
            res = requests.get(url, headers=headers, timeout=30)
        except requests.exceptions.Timeout:
            error_msg = f"❌ 请求超时: {url}"
            print(error_msg)
            raise Exception(f"网络超时 - {owner_repo} 第{page}页")
        except requests.exceptions.ConnectionError:
            error_msg = f"❌ 连接错误: 无法连接到GitHub API"
            print(error_msg)
            raise Exception(f"网络连接失败 - {owner_repo}")
        except requests.exceptions.RequestException as e:
            error_msg = f"❌ 请求异常: {str(e)}"
            print(error_msg)
            raise Exception(f"请求失败 - {owner_repo}: {str(e)}")
        
        if res.status_code == 200:
            print(f"✅ 第 {page} 页请求成功")
        elif res.status_code == 403:
            error_msg = f"❌ API限流 (HTTP 403): {owner_repo}"
            print(error_msg)
            print(f"📊 剩余请求次数: {res.headers.get('X-RateLimit-Remaining', 'Unknown')}")
            print(f"🕐 重置时间: {res.headers.get('X-RateLimit-Reset', 'Unknown')}")
            raise Exception(f"API限流 - {owner_repo}")
        elif res.status_code == 404:
            error_msg = f"❌ 仓库不存在 (HTTP 404): {owner_repo}"
            print(error_msg)
            raise Exception(f"仓库不存在 - {owner_repo}")
        elif res.status_code == 401:
            error_msg = f"❌ 认证失败 (HTTP 401): 请检查GitHub Token"
            print(error_msg)
            raise Exception(f"认证失败 - 请检查Token")
        else:
            error_msg = f"❌ HTTP错误 {res.status_code}: {owner_repo}"
            print(error_msg)
            try:
                error_detail = res.json().get('message', '未知错误')
                print(f"📝 错误详情: {error_detail}")
            except:
                print(f"📝 响应内容: {res.text[:200]}...")
            raise Exception(f"HTTP {res.status_code} - {owner_repo}")
            
        try:
            data = res.json()
        except json.JSONDecodeError:
            error_msg = f"❌ JSON解析失败: {owner_repo} 第{page}页"
            print(error_msg)
            raise Exception(f"JSON解析失败 - {owner_repo}")
            
        if not data:
            print(f"📄 第 {page} 页无数据，停止获取")
            break
            
        print(f"✅ 第 {page} 页获取到 {len(data)} 个issues")
        issues.extend(data)
        
        # 显示API限制信息
        remaining = res.headers.get('X-RateLimit-Remaining')
        if remaining:
            print(f"📊 API剩余请求次数: {remaining}")
            if int(remaining) < 10:
                print(f"⚠️  API请求次数即将耗尽，建议稍后再试")
        
        time.sleep(0.5)  # 避免限流
        
    print(f"\n🎯 {owner_repo} 总共获取到 {len(issues)} 个issues")
    return issues

# 抽取日志代码片段
def extract_logs_from_body(body, issue_info):
    logs = []
    
    if not body:
        return logs

    print(f"  🔍 分析Issue #{issue_info['number']}: {issue_info['title'][:50]}...")
    
    try:
        # 匹配 ```xxx``` 代码块
        code_blocks = re.findall(r"```[\s\S]*?```", body, re.MULTILINE)
        print(f"    📝 找到 {len(code_blocks)} 个代码块")
        
        log_blocks_found = 0
        for block in code_blocks:
            if is_log_text(block):
                cleaned_log = clean_log_text(block)
                if cleaned_log and len(cleaned_log) > 20:  # 过滤太短的内容
                    logs.append({
                        'timestamp': datetime.now().isoformat(),
                        'message': cleaned_log,
                        'label': 'unknown',  # 待标注
                        'source': 'github_issue',
                        'repository': issue_info['repository'],
                        'issue_number': issue_info['number'],
                        'issue_title': issue_info['title'],
                        'issue_url': issue_info['url']
                    })
                    log_blocks_found += 1
        
        if log_blocks_found > 0:
            print(f"    ✅ 从代码块中提取到 {log_blocks_found} 条日志")

        # 匹配 inline 报错信息（不在 ``` 中）
        lines = body.split("\n")
        current_log = []
        inline_logs_found = 0
        
        for line in lines:
            line = line.strip()
            if is_log_text(line):
                current_log.append(line)
            elif current_log and line == "":
                # 空行可能是日志的一部分
                current_log.append(line)
            elif current_log:
                # 非日志行，保存当前日志
                log_text = "\n".join(current_log).strip()
                if len(log_text) > 20:
                    logs.append({
                        'timestamp': datetime.now().isoformat(),
                        'message': log_text,
                        'label': 'unknown',
                        'source': 'github_issue',
                        'repository': issue_info['repository'],
                        'issue_number': issue_info['number'],
                        'issue_title': issue_info['title'],
                        'issue_url': issue_info['url']
                    })
                    inline_logs_found += 1
                current_log = []
        
        # 处理最后一个日志
        if current_log:
            log_text = "\n".join(current_log).strip()
            if len(log_text) > 20:
                logs.append({
                    'timestamp': datetime.now().isoformat(),
                    'message': log_text,
                    'label': 'unknown',
                    'source': 'github_issue',
                    'repository': issue_info['repository'],
                    'issue_number': issue_info['number'],
                    'issue_title': issue_info['title'],
                    'issue_url': issue_info['url']
                })
                inline_logs_found += 1
        
        if inline_logs_found > 0:
            print(f"    ✅ 从内联文本中提取到 {inline_logs_found} 条日志")
        
        total_logs = log_blocks_found + inline_logs_found
        if total_logs == 0:
            print(f"    ⚪ Issue #{issue_info['number']} 未发现日志内容")
        else:
            print(f"    🎯 Issue #{issue_info['number']} 总共提取到 {total_logs} 条日志")

    except Exception as e:
        error_msg = f"❌ 处理Issue #{issue_info['number']} 时出错: {str(e)}"
        print(error_msg)
        # 不抛出异常，继续处理其他issues
        
    return logs

# 保存数据集
def save_dataset(logs, repository, output_dir):
    if not logs:
        print(f"⚠️  {repository}: 没有日志数据需要保存")
        return
        
    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 创建输出目录: {output_dir}")
        
        # 生成文件名
        repo_name = repository.replace('/', '_')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_filename = f"{repo_name}_logs_{timestamp}.csv"
        csv_path = os.path.join(output_dir, csv_filename)
        
        print(f"💾 正在保存到文件: {csv_filename}")
        
        # 保存为CSV格式
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['timestamp', 'message', 'label', 'source', 'repository', 
                         'issue_number', 'issue_title', 'issue_url']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for log in logs:
                writer.writerow(log)
        
        print(f"✅ {repository}: 成功保存 {len(logs)} 条日志到 {csv_filename}")
        print(f"📊 文件大小: {os.path.getsize(csv_path) / 1024:.2f} KB")
        
    except PermissionError:
        error_msg = f"❌ 权限错误: 无法写入文件 {csv_path}"
        print(error_msg)
        raise Exception(f"文件写入权限错误 - {repository}")
    except OSError as e:
        error_msg = f"❌ 文件系统错误: {str(e)}"
        print(error_msg)
        raise Exception(f"文件系统错误 - {repository}: {str(e)}")
    except Exception as e:
        error_msg = f"❌ 保存文件时出错: {str(e)}"
        print(error_msg)
        raise Exception(f"文件保存失败 - {repository}: {str(e)}")

# 处理单个仓库
def process_repository(repo, headers, config):
    print(f"\n{'='*60}")
    print(f"🚀 开始处理仓库: {repo}")
    print(f"{'='*60}")
    
    try:
        issues = fetch_issues(repo, headers, config['max_pages'])
    except Exception as e:
        error_msg = f"❌ 获取issues失败: {str(e)}"
        print(error_msg)
        raise Exception(f"获取issues失败 - {repo}: {str(e)}")
    
    all_logs = []
    
    print(f"\n📋 开始分析 {len(issues)} 个issues...")
    
    for i, issue in enumerate(tqdm(issues, desc=f"处理 {repo} issues"), 1):
        try:
            issue_info = {
                'repository': repo,
                'number': issue.get('number'),
                'title': issue.get('title', ''),
                'url': issue.get('html_url', '')
            }
            
            print(f"\n📄 [{i}/{len(issues)}] 处理Issue #{issue_info['number']}")
            print(f"📝 标题: {issue_info['title']}")
            print(f"🔗 链接: {issue_info['url']}")
            
            # 提取issue正文中的日志
            logs = extract_logs_from_body(issue.get("body", ""), issue_info)
            all_logs.extend(logs)

            # 提取评论中的日志
            comments_url = issue.get("comments_url")
            if comments_url:
                print(f"  💬 获取评论: {comments_url}")
                try:
                    res = requests.get(comments_url, headers=headers, timeout=30)
                    if res.status_code == 200:
                        comments = res.json()
                        print(f"  📨 找到 {len(comments)} 条评论")
                        
                        comment_logs_total = 0
                        for j, comment in enumerate(comments, 1):
                            print(f"    💬 分析评论 {j}/{len(comments)}")
                            comment_logs = extract_logs_from_body(comment.get("body", ""), issue_info)
                            all_logs.extend(comment_logs)
                            comment_logs_total += len(comment_logs)
                        
                        if comment_logs_total > 0:
                            print(f"  ✅ 从评论中总共提取到 {comment_logs_total} 条日志")
                        else:
                            print(f"  ⚪ 评论中未发现日志内容")
                    else:
                        print(f"  ❌ 获取评论失败: HTTP {res.status_code}")
                except requests.exceptions.RequestException as e:
                    print(f"  ❌ 获取评论时网络错误: {str(e)}")
                    # 继续处理，不中断整个流程
                
                time.sleep(0.3)  # 避免限流
            else:
                print(f"  ⚪ 该Issue无评论")
            
            print(f"📊 当前累计日志数量: {len(all_logs)}")
            
        except Exception as e:
            print(f"❌ 处理Issue #{issue.get('number', 'Unknown')} 时出错: {str(e)}")
            # 继续处理下一个issue，不中断整个流程
            continue

    # 保存该仓库的数据集
    print(f"\n💾 开始保存 {repo} 的数据集...")
    try:
        save_dataset(all_logs, repo, config['output_directory'])
    except Exception as e:
        error_msg = f"❌ 保存数据集失败: {str(e)}"
        print(error_msg)
        raise Exception(f"保存数据集失败 - {repo}: {str(e)}")
    
    print(f"\n🎉 {repo} 处理完成！")
    print(f"📊 最终统计: {len(all_logs)} 条日志")
    
    return len(all_logs)

# 主函数
def main():
    print("🚀 GitHub Issue 日志爬取工具 - 处理剩余仓库")
    print("=" * 50)
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 加载配置
    print("\n📋 加载配置文件...")
    config = load_config()
    if not config:
        return
    
    print(f"✅ 配置加载成功")
    print(f"🎯 目标仓库数量: {len(config['repositories'])}")
    print(f"📁 输出目录: {config['output_directory']}")
    print(f"📄 每个仓库最大页数: {config['max_pages']}")
    
    # 显示要爬取的仓库列表
    print("\n📋 剩余未处理的仓库:")
    for i, repo in enumerate(config['repositories'], 1):
        print(f"  {i}. {repo}")
    
    # 设置请求头
    headers = {"Authorization": f"token {config['github_token']}"}
    print(f"\n🔑 使用GitHub Token: {config['github_token'][:10]}...")
    
    total_logs = 0
    successful_repos = 0
    failed_repos = []
    error_details = []
    
    # 处理每个仓库
    for i, repo in enumerate(config['repositories'], 1):
        print(f"\n🔄 [{i}/{len(config['repositories'])}] 开始处理仓库: {repo}")
        try:
            log_count = process_repository(repo, headers, config)
            total_logs += log_count
            successful_repos += 1
            print(f"✅ {repo} 处理成功，获取 {log_count} 条日志")
        except Exception as e:
            error_msg = f"❌ 处理仓库 {repo} 时出错: {str(e)}"
            print(error_msg)
            failed_repos.append(repo)
            error_details.append(f"{repo}: {str(e)}")
            continue
    
    # 最终统计
    print("\n" + "=" * 60)
    print("🎉 爬取任务完成！")
    print("=" * 60)
    print(f"⏰ 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 总体统计:")
    print(f"  🎯 总日志数量: {total_logs} 条")
    print(f"  ✅ 成功处理仓库: {successful_repos}/{len(config['repositories'])}")
    print(f"  📁 输出目录: {config['output_directory']}")
    
    if failed_repos:
        print(f"\n❌ 失败的仓库 ({len(failed_repos)}个):")
        for i, repo in enumerate(failed_repos, 1):
            print(f"  {i}. {repo}")
        
        print(f"\n📝 详细错误信息:")
        for error in error_details:
            print(f"  • {error}")
    
    print("\n💡 下一步建议:")
    if successful_repos > 0:
        print("1. 检查输出目录中的CSV文件")
        print("2. 使用 log_reviewer.py 工具来标注和验证日志")
        print("3. 将标注好的数据用于训练日志分类模型")
    
    if failed_repos:
        print("4. 检查失败仓库的错误信息")
        print("5. 修复问题后重新运行脚本")
    
    if total_logs > 0:
        print(f"\n🎊 恭喜！成功获取了 {total_logs} 条日志数据")
    else:
        print("\n⚠️  未获取到任何日志数据，请检查错误信息")

if __name__ == "__main__":
    main()
