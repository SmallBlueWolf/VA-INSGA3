#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
运行脚本，用于分别执行原始算法和改进算法，并比较结果
"""

import os
import sys
import time
import subprocess
import shutil
import pickle
import numpy as np
import datetime

# 创建输出记录类，同时将输出发送到控制台和文件
class TeeOutput:
    def __init__(self, file_path):
        self.file = open(file_path, 'w', encoding='utf-8')
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self
    
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
        
    def flush(self):
        self.file.flush()
        self.stdout.flush()
    
    def close(self):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.file.close()

def run_algorithm(improved=False):
    """运行特定版本的算法"""
    # 备份原始配置
    shutil.copy("config.py", "config_backup.py")
    
    try:
        # 修改配置文件中的USE_IMPROVED_ALGORITHM参数
        with open("config.py", "r", encoding="utf-8") as f:
            config_content = f.read()
        
        # 找到并替换USE_IMPROVED_ALGORITHM参数的值
        if "USE_IMPROVED_ALGORITHM = True" in config_content and not improved:
            config_content = config_content.replace("USE_IMPROVED_ALGORITHM = True", "USE_IMPROVED_ALGORITHM = False")
        elif "USE_IMPROVED_ALGORITHM = False" in config_content and improved:
            config_content = config_content.replace("USE_IMPROVED_ALGORITHM = False", "USE_IMPROVED_ALGORITHM = True")
        
        # 写回配置文件
        with open("config.py", "w", encoding="utf-8") as f:
            f.write(config_content)
        
        # 执行优化算法
        print(f"开始执行{'改进' if improved else '原始'}算法...")
        start_time = time.time()
        
        # 使用subprocess执行，以便捕获输出
        result = subprocess.run([sys.executable, "main.py"], 
                              capture_output=True, text=True, check=True)
        
        # 打印输出，但忽略进度条输出以减少冗余
        for line in result.stdout.split("\n"):
            if not ("[" in line and "]" in line and "%" in line):  # 简单过滤进度条行
                print(line)
        
        end_time = time.time()
        print(f"{'改进' if improved else '原始'}算法执行完成，总耗时: {end_time - start_time:.2f}秒")
        
        return True
    except Exception as e:
        print(f"执行{'改进' if improved else '原始'}算法时出错: {e}")
        return False
    finally:
        # 恢复配置文件
        shutil.copy("config_backup.py", "config.py")
        # 删除备份
        os.remove("config_backup.py")

def check_results_exist():
    """检查两种算法的结果是否都存在"""
    origin_exists = os.path.exists("origin/pareto_data.pkl")
    improv_exists = os.path.exists("improv/pareto_data.pkl")
    
    return origin_exists and improv_exists

def main():
    """主函数，运行两种算法并比较结果"""
    # 创建日志文件
    log_file_path = f"log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    tee = TeeOutput(log_file_path)
    
    try:
        print("=" * 60)
        print("A2ACMOP优化算法对比测试")
        print(f"开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # 检查是否已有结果
        if check_results_exist():
            print("检测到两种算法的结果已存在。")
            proceed = input("是否要重新运行优化算法？(y/n): ").strip().lower()
            if proceed != 'y':
                print("跳过优化过程，直接进行结果比较...")
                # 调用metric.py进行结果比较
                import metric
                metric.compare_results()
                return
        
        # 运行原始算法
        print("\n=== 执行原始NSGA-III算法 ===\n")
        success_origin = run_algorithm(improved=False)
        
        if not success_origin:
            print("原始算法执行失败，终止程序")
            return
        
        # 运行改进算法
        print("\n=== 执行改进NSGA-III算法 ===\n")
        success_improv = run_algorithm(improved=True)
        
        if not success_improv:
            print("改进算法执行失败，终止程序")
            return
        
        # 如果两种算法都成功运行，则调用metric.py进行比较
        if success_origin and success_improv:
            print("\n=== 执行结果对比分析 ===\n")
            import metric
            metric.compare_results()
            
        print(f"\n执行完成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"日志已保存至: {log_file_path}")
    finally:
        # 关闭输出重定向
        tee.close()

if __name__ == "__main__":
    main() 