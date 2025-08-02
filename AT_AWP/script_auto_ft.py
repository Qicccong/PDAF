import subprocess
import sys

# 配置部分
SCRIPT_NAME = "AT_AWP/RItrain.py"
ARG_COMBINATIONS = [

    ["--lr", "0.005", "--awp-gamma", "0.005", "--dataset", "ml-1m", "--model_path", "AT_AWP/param_file/RAWP/80_ml-1m_MLP_init_1713513910.3174512.pth"],
    ["--lr", "0.005", "--awp-gamma", "0.001", "--dataset", "ml-1m", "--model_path", "AT_AWP/param_file/RAWP/80_ml-1m_MLP_init_1713513910.3174512.pth"],
    ["--lr", "0.001", "--awp-gamma", "0.001", "--dataset", "ml-1m", "--model_path", "AT_AWP/param_file/RAWP/80_ml-1m_MLP_init_1713513910.3174512.pth"],
    
    
]
OUTPUT_PREFIX2 = "AT_AWP/output/PDAF/FT"

def main():
    for idx, args in enumerate(ARG_COMBINATIONS, 1):
        pre = OUTPUT_PREFIX2
    
        # 提取 lr 和 dataset
        lr_value = None
        dataset_value = None
        gamma = None
        epoch = 5
        
        # 遍历 args 找到 --lr 和 --dataset 的值
        for i in range(len(args)):
            if args[i] == "--lr" and i + 1 < len(args):
                lr_value = args[i + 1]
            if args[i] == "--dataset" and i + 1 < len(args):
                dataset_value = args[i + 1]
            if args[i] == "--awp-gamma" and i + 1 < len(args):
                gamma = args[i + 1]
            if args[i] == "--epochs" and i + 1 < len(args):
                epoch = args[i + 1]
        if lr_value is None:
            lr_value = 0.0
        if dataset_value is None:
            dataset_value = "none"
        if gamma is None:
            gamma = 0.008

        exp = "Ablation"

        output_file = f"{pre}_{dataset_value}_lr-{lr_value}_gamma-{gamma}.txt"


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