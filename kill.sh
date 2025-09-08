#!/bin/bash

# 燃烧室分析程序终止脚本
# 使用方法: bash kill.sh

echo "🔍 查找正在运行的 homework 进程..."

# 方法1: 通过进程ID文件终止
if [ -f ".homework_pid" ]; then
    PID=$(cat .homework_pid)
    echo "📝 找到PID文件: $PID"
    
    if kill -0 $PID 2>/dev/null; then
        echo "⚡ 正在终止进程 $PID..."
        kill -TERM $PID
        sleep 2
        
        # 如果进程仍然存在，强制终止
        if kill -0 $PID 2>/dev/null; then
            echo "💥 强制终止进程 $PID..."
            kill -KILL $PID
        fi
        
        echo "✅ 进程 $PID 已终止"
        rm -f .homework_pid
    else
        echo "⚠️  进程 $PID 不存在，清理PID文件"
        rm -f .homework_pid
    fi
else
    echo "📄 未找到PID文件"
fi

# 方法2: 通过进程名查找并终止
echo ""
echo "🔍 查找所有 Python homework 相关进程..."

PIDS=$(pgrep -f "python.*main.py" 2>/dev/null)

if [ -n "$PIDS" ]; then
    echo "📋 找到以下进程:"
    ps aux | grep -E "python.*main.py" | grep -v grep
    
    echo ""
    echo "⚡ 终止这些进程..."
    
    for PID in $PIDS; do
        echo "  - 终止进程 $PID"
        kill -TERM $PID 2>/dev/null
    done
    
    sleep 3
    
    # 检查是否还有残留进程，强制终止
    REMAINING=$(pgrep -f "python.*main.py" 2>/dev/null)
    if [ -n "$REMAINING" ]; then
        echo "💥 强制终止残留进程..."
        for PID in $REMAINING; do
            kill -KILL $PID 2>/dev/null
            echo "  - 强制终止进程 $PID"
        done
    fi
    
    echo "✅ 所有相关进程已终止"
else
    echo "ℹ️  未找到正在运行的 homework 进程"
fi

# 方法3: 清理可能的僵尸进程
echo ""
echo "🧹 清理环境..."

# 清理临时文件
rm -f .homework_pid
rm -f homework_process.log.lock

# 检查是否还有相关进程
FINAL_CHECK=$(pgrep -f "python.*main.py" 2>/dev/null)
if [ -n "$FINAL_CHECK" ]; then
    echo "⚠️  警告: 仍有进程在运行:"
    ps aux | grep -E "python.*main.py" | grep -v grep
    echo ""
    echo "💡 如果需要强制终止，请运行: sudo kill -9 $FINAL_CHECK"
else
    echo "✨ 环境清理完成"
fi

echo ""
echo "🏁 终止脚本执行完成"

# 显示系统资源情况
echo ""
echo "📊 当前系统资源:"
echo "内存使用情况:"
if command -v free >/dev/null 2>&1; then
    free -h
elif command -v vm_stat >/dev/null 2>&1; then
    # macOS 系统
    vm_stat | head -5
fi

echo ""
echo "CPU负载:"
if command -v uptime >/dev/null 2>&1; then
    uptime
fi