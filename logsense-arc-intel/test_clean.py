#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试数据清理
"""

import pandas as pd
import re

def clean_log_content(text):
    if pd.isna(text) or text == '':
        return ''
    
    text = str(text)
    
    # 移除GitHub元数据
    patterns = [
        r'github\.com/[^,\s]+',
        r'https://github\.com/[^,\s]+',
        r'github_issue',
        r'unknown,github_issue',
        r'[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]+',
        r'[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+,\d+',
        r'https://github\.com/[^,\s]+/issues/\d+',
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # 移除HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'&[a-zA-Z]+;', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    
    # 清理多余字符
    text = re.sub(r'^,+|,+$', '', text)
    text = re.sub(r'^"+|"+$', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def is_valid_log(text):
    if pd.isna(text) or text == '':
        return False
    
    text = str(text).lower()
    log_keywords = ['error', 'warn', 'info', 'debug', 'exception', 'java', 'spring', 'at ', 'caused by']
    has_keyword = any(keyword in text for keyword in log_keywords)
    has_length = len(text) > 20
    
    return has_keyword or has_length

# 测试清理函数
test_texts = [
    '2025-07-25T01:29:29.341834,provider.setHideUserNotFoundExceptions(false); // <-- it\'s quite a long way to set this property,unknown,github_issue,spring-projects/spring-security,17209,Why isn\'t there a simple way to set `org.springframework.security.authentication.dao.AbstractUserDetailsAuthenticationProvider#hideUserNotFoundExceptions` to `false`?,https://github.com/spring-projects/spring-security/issues/17209',
    'at org.springframework.boot.SpringApplication.run(SpringApplication.java:1361) ~[spring-boot-3.4.4.jar:3.4.4]',
    'java.lang.Exception: Connection timeout',
    'github_issue,spring-projects/spring-security,16496,AbstractUserDetailsAuthenticationProvider should not swallow UsernameNotFoundException'
]

print('清理测试:')
for i, text in enumerate(test_texts):
    cleaned = clean_log_content(text)
    valid = is_valid_log(cleaned)
    print(f'{i+1}. 原始: {text[:100]}...')
    print(f'   清理: {cleaned[:100]}...')
    print(f'   有效: {valid}')
    print() 