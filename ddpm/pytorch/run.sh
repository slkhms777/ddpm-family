#!/bin/bash

# 定义日志目录
Log_dir="./Logs"

# 创建日志目录（如果不存在）
mkdir -p "$Log_dir"

# 获取当前日期和时间，格式为 YYYY-MM-DD_HH-MM-SS
Current_datetime=$(date +%Y-%m-%d_%H-%M-%S)

# 定义日志文件路径，包含日期和时间
Log_file="$Log_dir/${Current_datetime}.log"

# 执行 Python 脚本，并将标准输出和标准错误都重定向
# 使用 tee 命令将输出同时发送到控制台和日志文件
# 2>&1 表示将标准错误重定向到标准输出，这样它们都会被 tee 处理
python main.py 2>&1 | tee "$Log_file"

echo "脚本执行完毕，日志已保存到 $Log_file"