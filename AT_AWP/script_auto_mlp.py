import subprocess
import sys

# 配置部分
SCRIPT_NAME = "D:\\Users\\Jchu\\Desktop\\RAWPFT250217\\RAWPFT250217\\AT_AWP\\train_mlp_layer.py"
ARG_COMBINATIONS = [
    ["--lr-max", "0.0005", "--awp-gamma", "0.002","--dataset","AMusic"],
    ["--lr-max", "0.0006", "--awp-gamma", "0.002","--dataset","AMusic"],
    ["--lr-max", "0.0005", "--awp-gamma", "0.003","--dataset","AMusic"],
    ["--lr-max", "0.0006", "--awp-gamma", "0.003","--dataset","AMusic"],
    ["--lr-max", "0.0005", "--awp-gamma", "0.004","--dataset","AMusic"],
    ["--lr-max", "0.0006", "--awp-gamma", "0.004","--dataset","AMusic"],
    ["--lr-max", "0.0005", "--awp-gamma", "0.005","--dataset","AMusic"],
    ["--lr-max", "0.0006", "--awp-gamma", "0.005","--dataset","AMusic"],

    ["--lr-max", "0.0001", "--awp-gamma", "0.002","--dataset","ml-1m"],
    ["--lr-max", "0.0002", "--awp-gamma", "0.002","--dataset","ml-1m"],
    ["--lr-max", "0.0001", "--awp-gamma", "0.003","--dataset","ml-1m"],
    ["--lr-max", "0.0002", "--awp-gamma", "0.003","--dataset","ml-1m"],
    ["--lr-max", "0.0001", "--awp-gamma", "0.004","--dataset","ml-1m"],
    ["--lr-max", "0.0002", "--awp-gamma", "0.004","--dataset","ml-1m"],


    # ["--lr-max", "0.0004", "--awp-gamma", "0.007","--dataset","AMusic"],
    # ["--lr-max", "0.0005", "--awp-gamma", "0.007","--dataset","AMusic"],
    # ["--lr-max", "0.0006", "--awp-gamma", "0.007","--dataset","AMusic"],

    # ["--lr-max", "0.0004", "--awp-gamma", "0.008","--dataset","AMusic"],
    # ["--lr-max", "0.0005", "--awp-gamma", "0.008","--dataset","AMusic"],
    # ["--lr-max", "0.0006", "--awp-gamma", "0.008","--dataset","AMusic"],
    
    # ["--lr-max", "0.0004", "--awp-gamma", "0.009","--dataset","AMusic"],
    # ["--lr-max", "0.0005", "--awp-gamma", "0.009","--dataset","AMusic"],
    # ["--lr-max", "0.0006", "--awp-gamma", "0.009","--dataset","AMusic"],

    # ["--lr-max", "0.0001", "--awp-gamma", "0.007","--dataset","ml-1m"],
    # ["--lr-max", "0.0002", "--awp-gamma", "0.007","--dataset","ml-1m"],

    # ["--lr-max", "0.0001", "--awp-gamma", "0.008","--dataset","ml-1m"],
    # ["--lr-max", "0.0002", "--awp-gamma", "0.008","--dataset","ml-1m"],

    # ["--lr-max", "0.0001", "--awp-gamma", "0.009","--dataset","ml-1m"],
    # ["--lr-max", "0.0002", "--awp-gamma", "0.009","--dataset","ml-1m"],
]
OUTPUT_PREFIX = "output_auto_log"

def main():
    for idx, args in enumerate(ARG_COMBINATIONS, 1):
        output_file = f"{OUTPUT_PREFIX}_{idx}.txt"
        cmd = [sys.executable, SCRIPT_NAME] + args
        
        try:
            with open(output_file, "w") as f:
                # 关键修改：合并stdout和stderr
                result = subprocess.run(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,  # 将标准错误合并到标准输出
                    text=True,
                    check=True
                )
            print(f"✅ 运行成功 {idx} | 日志文件: {output_file}")
            
        except subprocess.CalledProcessError as e:
            # 错误信息已自动写入文件
            print(f"❌ 运行失败 {idx} | 错误码 {e.returncode}")
            print(f"📁 完整日志请查看: {output_file}")

if __name__ == "__main__":
    main()