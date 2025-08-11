import requests
import time
import re
import json
import os
import csv
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict

# 加载配置
def load_config(config_file='config_extended.json'):
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ 配置文件 {config_file} 不存在，请先创建配置文件")
        return None
    except json.JSONDecodeError:
        print(f"❌ 配置文件 {config_file} 格式错误")
        return None

# 增强的日志识别
def is_log_text_enhanced(text, log_keywords):
    # 基础日志关键词检查
    if any(k.lower() in text.lower() for k in log_keywords):
        return True
    
    # 更精确的日志模式匹配
    log_patterns = [
        r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}',  # 时间戳
        r'\[\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\]',  # 带方括号的时间戳
        r'ERROR|WARN|INFO|DEBUG|TRACE|FATAL',  # 日志级别
        r'Exception\s+in\s+thread',  # 线程异常
        r'Caused\s+by:',  # 异常链
        r'at\s+[\w\.$]+\([^)]*\)',  # 堆栈跟踪
        r'java\.lang\.',  # Java异常
        r'org\.springframework\.',  # Spring异常
        r'org\.hibernate\.',  # Hibernate异常
        r'org\.apache\.',  # Apache异常
        r'com\.mysql\.',  # MySQL异常
        r'redis\.',  # Redis异常
        r'kafka\.',  # Kafka异常
        r'rabbitmq\.',  # RabbitMQ异常
        r'Connection\s+refused',  # 连接错误
        r'Connection\s+timeout',  # 连接超时
        r'SQL\s+state',  # SQL状态
        r'ORA-\d+',  # Oracle错误
        r'MySQL\s+error',  # MySQL错误
        r'PostgreSQL\s+error',  # PostgreSQL错误
        r'Authentication\s+failed',  # 认证失败
        r'Authorization\s+denied',  # 授权拒绝
        r'Validation\s+failed',  # 验证失败
        r'Transaction\s+rollback',  # 事务回滚
        r'Deadlock\s+found',  # 死锁
        r'Lock\s+wait\s+timeout',  # 锁等待超时
        r'OutOfMemoryError',  # 内存溢出
        r'StackOverflowError',  # 栈溢出
        r'ClassNotFoundException',  # 类未找到
        r'NoSuchMethodException',  # 方法未找到
        r'IllegalStateException',  # 非法状态异常
        r'BeanCreationException',  # Bean创建异常
        r'NoSuchBeanDefinitionException',  # Bean定义未找到
        r'CircularDependencyException',  # 循环依赖异常
        r'DataIntegrityViolationException',  # 数据完整性违反异常
        r'JdbcSQLException',  # JDBC SQL异常
        r'HttpMessageNotReadableException',  # HTTP消息不可读异常
        r'MethodArgumentNotValidException',  # 方法参数验证异常
        r'BindException',  # 绑定异常
        r'TypeMismatchException',  # 类型不匹配异常
        r'MissingServletRequestParameterException',  # 缺少请求参数异常
        r'HttpRequestMethodNotSupportedException',  # HTTP方法不支持异常
        r'HttpMediaTypeNotSupportedException',  # HTTP媒体类型不支持异常
        r'NoHandlerFoundException',  # 处理器未找到异常
        r'AsyncRequestTimeoutException',  # 异步请求超时异常
        r'ResponseStatusException',  # 响应状态异常
        r'AccessDeniedException',  # 访问拒绝异常
        r'BadCredentialsException',  # 凭据错误异常
        r'UsernameNotFoundException',  # 用户名未找到异常
        r'AccountExpiredException',  # 账户过期异常
        r'LockedException',  # 账户锁定异常
        r'DisabledException',  # 账户禁用异常
        r'CredentialsExpiredException',  # 凭据过期异常
        r'InvalidTokenException',  # 无效令牌异常
        r'TokenExpiredException',  # 令牌过期异常
        r'JwtException',  # JWT异常
        r'OAuth2AuthenticationException',  # OAuth2认证异常
        r'InvalidGrantException',  # 无效授权异常
        r'RedirectMismatchException',  # 重定向不匹配异常
        r'UnsupportedGrantTypeException',  # 不支持的授权类型异常
        r'InvalidClientException',  # 无效客户端异常
        r'InvalidScopeException',  # 无效作用域异常
        r'InsufficientScopeException',  # 作用域不足异常
        r'InvalidRequestException',  # 无效请求异常
        r'UnsupportedResponseTypeException',  # 不支持的响应类型异常
        r'UserDeniedAuthorizationException',  # 用户拒绝授权异常
    ]
    
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in log_patterns)

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
def extract_logs_from_body(body, issue_info, log_keywords):
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
            if is_log_text_enhanced(block, log_keywords):
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
                        'issue_url': issue_info['url'],
                        'category': issue_info.get('category', 'unknown')
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
            if is_log_text_enhanced(line, log_keywords):
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
                        'issue_url': issue_info['url'],
                        'category': issue_info.get('category', 'unknown')
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
                    'issue_url': issue_info['url'],
                    'category': issue_info.get('category', 'unknown')
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
def save_dataset(logs, repository, output_dir, category='unknown'):
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
        csv_filename = f"{repo_name}_logs_{timestamp}_{category}.csv"
        csv_path = os.path.join(output_dir, csv_filename)
        
        print(f"💾 正在保存到文件: {csv_filename}")
        
        # 保存为CSV格式
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['timestamp', 'message', 'label', 'source', 'repository', 
                         'issue_number', 'issue_title', 'issue_url', 'category']
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
def process_repository(repo, headers, config, category='unknown'):
    print(f"\n{'='*60}")
    print(f"🚀 开始处理仓库: {repo} (类别: {category})")
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
                'url': issue.get('html_url', ''),
                'category': category
            }
            
            print(f"\n📄 [{i}/{len(issues)}] 处理Issue #{issue_info['number']}")
            print(f"📝 标题: {issue_info['title']}")
            print(f"🔗 链接: {issue_info['url']}")
            
            # 提取issue正文中的日志
            logs = extract_logs_from_body(issue.get("body", ""), issue_info, config['log_keywords'])
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
                            comment_logs = extract_logs_from_body(comment.get("body", ""), issue_info, config['log_keywords'])
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
        save_dataset(all_logs, repo, config['output_directory'], category)
    except Exception as e:
        error_msg = f"❌ 保存数据集失败: {str(e)}"
        print(error_msg)
        raise Exception(f"保存数据集失败 - {repo}: {str(e)}")
    
    print(f"\n🎉 {repo} 处理完成！")
    print(f"📊 最终统计: {len(all_logs)} 条日志")
    
    return len(all_logs)

# 按类别处理仓库
def process_by_category(config, headers):
    category_stats = defaultdict(lambda: {'total_logs': 0, 'successful_repos': 0, 'failed_repos': []})
    
    for category, repos in config['categories'].items():
        print(f"\n{'='*80}")
        print(f"🎯 开始处理类别: {category}")
        print(f"📋 包含仓库: {', '.join(repos)}")
        print(f"{'='*80}")
        
        for repo in repos:
            try:
                log_count = process_repository(repo, headers, config, category)
                category_stats[category]['total_logs'] += log_count
                category_stats[category]['successful_repos'] += 1
                print(f"✅ {repo} 处理成功，获取 {log_count} 条日志")
            except Exception as e:
                error_msg = f"❌ 处理仓库 {repo} 时出错: {str(e)}"
                print(error_msg)
                category_stats[category]['failed_repos'].append(repo)
                continue
    
    return category_stats

# 主函数
def main():
    print("🚀 GitHub Issue 日志爬取工具 - 增强版")
    print("=" * 60)
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 加载配置
    print("\n📋 加载配置文件...")
    config = load_config('config_extended.json')
    if not config:
        return
    
    print(f"✅ 配置加载成功")
    print(f"🎯 目标仓库数量: {len(config['repositories'])}")
    print(f"📁 输出目录: {config['output_directory']}")
    print(f"📄 每个仓库最大页数: {config['max_pages']}")
    
    # 显示类别信息
    print("\n📋 按类别组织的仓库:")
    for category, repos in config['categories'].items():
        print(f"  🏷️  {category}: {len(repos)} 个仓库")
        for repo in repos:
            print(f"    • {repo}")
    
    # 设置请求头
    headers = {"Authorization": f"token {config['github_token']}"}
    print(f"\n🔑 使用GitHub Token: {config['github_token'][:10]}...")
    
    # 按类别处理
    category_stats = process_by_category(config, headers)
    
    # 最终统计
    print("\n" + "=" * 80)
    print("🎉 爬取任务完成！")
    print("=" * 80)
    print(f"⏰ 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    total_logs = 0
    total_successful = 0
    total_failed = 0
    
    print(f"\n📊 按类别统计:")
    for category, stats in category_stats.items():
        print(f"\n🏷️  {category}:")
        print(f"  📊 日志数量: {stats['total_logs']} 条")
        print(f"  ✅ 成功仓库: {stats['successful_repos']} 个")
        if stats['failed_repos']:
            print(f"  ❌ 失败仓库: {len(stats['failed_repos'])} 个")
            for repo in stats['failed_repos']:
                print(f"    • {repo}")
        
        total_logs += stats['total_logs']
        total_successful += stats['successful_repos']
        total_failed += len(stats['failed_repos'])
    
    print(f"\n📊 总体统计:")
    print(f"  🎯 总日志数量: {total_logs} 条")
    print(f"  ✅ 成功处理仓库: {total_successful} 个")
    print(f"  ❌ 失败仓库: {total_failed} 个")
    print(f"  📁 输出目录: {config['output_directory']}")
    
    print("\n💡 下一步建议:")
    if total_logs > 0:
        print("1. 检查输出目录中的CSV文件")
        print("2. 使用 log_reviewer.py 工具来标注和验证日志")
        print("3. 将标注好的数据用于训练日志分类模型")
        print("4. 分析不同类别的日志分布，优化数据平衡")
    
    if total_failed > 0:
        print("5. 检查失败仓库的错误信息")
        print("6. 修复问题后重新运行脚本")
    
    if total_logs > 0:
        print(f"\n🎊 恭喜！成功获取了 {total_logs} 条日志数据")
        print("📈 数据覆盖了多种Spring Boot应用场景，将显著提升模型性能！")
    else:
        print("\n⚠️  未获取到任何日志数据，请检查错误信息")

if __name__ == "__main__":
    main()
