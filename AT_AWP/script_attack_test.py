import subprocess
import sys

# 配置部分
SCRIPT_NAME = "AT_AWP/Attack_test_testattack.py"
ARG_COMBINATIONS = [
   
    # ["--dataset", "lastfm", "--attack", "fgsm", "--epsilon" , "0.008", "--model_path", "pretrained/AMusic/lastfm_0.05_0.008_PDAF_Inter.pth"],
    ["--dataset", "lastfm", "--attack", "bim", "--epsilon" , "0.008", "--model_path", "pretrained/AMusic/lastfm_0.05_0.008_PDAF_Inter.pth"],  
    ["--dataset", "lastfm", "--attack", "pgd", "--epsilon" , "0.008", "--model_path", "pretrained/AMusic/lastfm_0.05_0.008_PDAF_Inter.pth"],  
    ["--dataset", "lastfm", "--attack", "mim", "--epsilon" , "0.008", "--model_path", "pretrained/AMusic/lastfm_0.05_0.008_PDAF_Inter.pth"],
    # ["--dataset", "lastfm", "--attack", "none", "--epsilon" , "0.008", "--model_path", "pretrained/AMusic/lastfm_0.05_0.008_PDAF_Inter.pth"],       
    # ["--dataset", "ml-1m", "--attack", "fgsm", "--epsilon" , "0.020", "--model_path", "AT_AWP/param_file/FTS/ML-1M/ml-1m_FTS_lr-0.005_gamma-0.005.pth"],

    # ["--dataset", "ml-1m", "--attack", "fgsm", "--epsilon" , "0.008", "--model_path", "results/MLP_clean/FT/RAWP_ml-1m/checkpoint/rift_refine_layer_robust_lastfm_SGDM_lr=0.005_g=0.005_wd=0.0005_epochs=20_None/finetune_0.500_params.pth"],   
    # ["--dataset", "ml-1m", "--attack", "none", "--epsilon" , "0.008", "--model_path", "results/MLP_clean/FT/RAWP_ml-1m/checkpoint/rift_refine_layer_robust_lastfm_SGDM_lr=0.005_g=0.005_wd=0.0005_epochs=20_None/finetune_0.500_params.pth"],   
    # ["--dataset", "ml-1m", "--attack", "pgd", "--epsilon" , "0.008", "--model_path", "results/MLP_clean/FT/RAWP_ml-1m/checkpoint/rift_refine_layer_robust_lastfm_SGDM_lr=0.005_g=0.005_wd=0.0005_epochs=20_None/finetune_0.500_params.pth"],   
    # ["--dataset", "ml-1m", "--attack", "mim", "--epsilon" , "0.008", "--model_path", "results/MLP_clean/FT/RAWP_ml-1m/checkpoint/rift_refine_layer_robust_lastfm_SGDM_lr=0.005_g=0.005_wd=0.0005_epochs=20_None/finetune_0.500_params.pth"],      
    
    # ["--dataset", "ml-1m", "--attack", "bim", "--epsilon" , "0.008", "--model_path", "AT_AWP/param_file/RAWP-FT/ml-1m_RAWP_80_FineTurning_1715970106.2888427.pth"],   
    # ["--dataset", "ml-1m", "--attack", "pgd", "--epsilon" , "0.008", "--model_path", "AT_AWP/param_file/RAWP-FT/ml-1m_RAWP_80_FineTurning_1715970106.2888427.pth"],   
    # ["--dataset", "ml-1m", "--attack", "mim", "--epsilon" , "0.008", "--model_path", "AT_AWP/param_file/RAWP-FT/ml-1m_RAWP_80_FineTurning_1715970106.2888427.pth"],      
    
]

OUTPUT_PREFIX = "AT_AWP/output/"

def main():
    for idx, args in enumerate(ARG_COMBINATIONS, 1):
        pre = OUTPUT_PREFIX

        dataset_value = None
        gamma = None
        attack = None
        
        # 遍历 args 找到 --lr 和 --dataset 的值
        for i in range(len(args)):
            if args[i] == "--dataset" and i + 1 < len(args):
                dataset_value = args[i + 1]
            if args[i] == "--epsilon" and i + 1 < len(args):
                gamma = args[i + 1]
            if args[i] == "--attack" and i + 1 < len(args):
                attack = args[i + 1]

        output_file = f"{pre}{dataset_value}_attack-{attack}_e-{gamma}_FT.txt"
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