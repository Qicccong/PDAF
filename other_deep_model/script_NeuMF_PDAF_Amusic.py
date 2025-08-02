import subprocess
import sys
import os

# 配置部分
SCRIPT_NAME = "AT_AWP/other_deep_model/train_NeuMF_PDAF.py"
ARG_COMBINATIONS = [
   

    ["--lr", "0.01", "--awp-gamma", "0.008", "--data_set", "AMusic", "--model_path", "AT_AWP/param_file/other_model/NeuMF/AMusic/AMusic_lr-0.0001_g-0.005_NeuMF_3896.pth"],
    ["--lr", "0.1", "--awp-gamma", "0.008", "--data_set", "AMusic", "--model_path", "AT_AWP/param_file/other_model/NeuMF/AMusic/AMusic_lr-0.0001_g-0.005_NeuMF_3896.pth"],
    ["--lr", "0.001", "--awp-gamma", "0.008", "--data_set", "AMusic", "--model_path", "AT_AWP/param_file/other_model/NeuMF/AMusic/AMusic_lr-0.0001_g-0.005_NeuMF_3896.pth"],

]
OUTPUT_PREFIX2 = "AT_AWP/output/other_model"

def generate_unique_filename(path):
    """生成带递增序号的文件名以避免覆盖"""
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    n = 1
    while True:
        new_path = f"{base}({n}){ext}"
        if not os.path.exists(new_path):
            return new_path
        n += 1

def main():
    for idx, args in enumerate(ARG_COMBINATIONS, 1):
        pre = OUTPUT_PREFIX2
    
        lr_value = None
        dataset_value = None
        gamma = None
        
        # 遍历 args 找到 --lr 和 --dataset 的值
        for i in range(len(args)):
            if (args[i] == "--data_set" or args[i] == "--dataset") and i + 1 < len(args):
                dataset_value = args[i + 1]
            if args[i] == "--lr" and i + 1 < len(args):
                lr_value = args[i + 1]
            if args[i] == "--awp-gamma" and i + 1 < len(args):
                gamma = args[i + 1]
        if dataset_value is None:
            dataset_value = "none"
        if gamma is None:
            # gamma = 0.005
            gamma = 0.005


        base_output = f"{pre}/{dataset_value}_lr-{lr_value}_g-{gamma}_NeuMF_PDAF.txt"
        output_file = generate_unique_filename(base_output)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

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