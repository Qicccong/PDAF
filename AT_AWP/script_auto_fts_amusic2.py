import subprocess
import sys

# 配置部分
SCRIPT_NAME = "AT_AWP/RItrain_sele_qc_ri.py"
ARG_COMBINATIONS = [

    # ["--lr", "0.1", "--awp-gamma", "0.005", "--dataset", "AMusic", "--epochs", "5", "--model_path", "AT_AWP/param_file/RAWP/11_AMusic_MLP_init_1714030591.7578497.pth"],
    # ["--lr", "0.1", "--awp-gamma", "0.002", "--dataset", "AMusic", "--epochs", "5", "--model_path", "AT_AWP/param_file/RAWP/11_AMusic_MLP_init_1714030591.7578497.pth"],
    # ["--lr", "0.1", "--awp-gamma", "0.008", "--dataset", "AMusic", "--epochs", "5", "--model_path", "AT_AWP/param_file/RAWP/11_AMusic_MLP_init_1714030591.7578497.pth"],
    
    # ["--lr", "0.1", "--awp-gamma", "0.005", "--dataset", "AMusic", "--epochs", "10", "--model_path", "AT_AWP/param_file/RAWP/11_AMusic_MLP_init_1714030591.7578497.pth"],
    ["--lr", "0.1", "--awp-gamma", "0.002", "--dataset", "AMusic", "--epochs", "10", "--model_path", "AT_AWP/param_file/RAWP/11_AMusic_MLP_init_1714030591.7578497.pth"],
    ["--lr", "0.1", "--awp-gamma", "0.010", "--dataset", "AMusic", "--epochs", "10", "--model_path", "AT_AWP/param_file/RAWP/11_AMusic_MLP_init_1714030591.7578497.pth"],
    
    # ["--lr", "0.1", "--awp-gamma", "0.012", "--dataset", "AMusic", "--model_path", "D:\\Users\\Jchu\\Desktop\\RAWPFT250217\\RAWPFT250217\\AT_AWP\\param_file\\RAWP\\11_AMusic_MLP_init_1714030591.7578497.pth"],
    # ["--lr", "0.1", "--awp-gamma", "0.016", "--dataset", "AMusic", "--model_path", "D:\\Users\\Jchu\\Desktop\\RAWPFT250217\\RAWPFT250217\\AT_AWP\\param_file\\RAWP\\11_AMusic_MLP_init_1714030591.7578497.pth"],
    # ["--lr", "0.1", "--awp-gamma", "0.020", "--dataset", "AMusic", "--model_path", "D:\\Users\\Jchu\\Desktop\\RAWPFT250217\\RAWPFT250217\\AT_AWP\\param_file\\RAWP\\11_AMusic_MLP_init_1714030591.7578497.pth"],

    
]
OUTPUT_PREFIX2 = "AT_AWP/output/PDAF/PDAF"

def main():
    for idx, args in enumerate(ARG_COMBINATIONS, 1):
        pre = OUTPUT_PREFIX2
    
        # 提取 lr 和 dataset
        lr_value = None
        dataset_value = None
        gamma = None
        epoch = 0
        
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

        output_file = f"{pre}_{dataset_value}_lr-{lr_value}_gamma-{gamma}_e-{epoch}.txt"


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