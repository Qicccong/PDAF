import subprocess
import sys

# 配置部分
SCRIPT_NAME = "AT_AWP/other_deep_model/Attack_test_testattack.py"
ARG_COMBINATIONS = [

    # ["--dataset", "lastfm", "--attack", "none", "--epsilon" , "0.008", "--model", "NeuMF", "--model_path", "pretrain/NeuMF_PDAF/lastfm_0.001_0.008/lastfm_lr-0.001_g-0.008_NeuMF_PDAF.pth"],   
    # ["--dataset", "lastfm", "--attack", "none", "--epsilon" , "0.008", "--model", "ConvNCF", "--model_path", "AT_AWP/param_file/other_model/ConvNCF/lastfm/lastfm_lr-0.01_g-0.008_8283.pth"],   
    
    # ["--dataset", "AMusic", "--attack", "none", "--epsilon" , "0.008", "--model", "ConvNCF", "--model_path", "pretrain/ConvNCF_PDAF/AMusic_0.01_0.008/AMusic_lr-0.01_g-0.008_ConvNCF_PDAF.pth"],   
    # ["--dataset", "AMusic", "--attack", "none", "--epsilon" , "0.008", "--model", "ConvNCF", "--model_path", "AT_AWP/param_file/other_model/ConvNCF/AMusic/AMusic_ConNCF_RAT_3294.pth"],   
    # ["--dataset", "AMusic", "--attack", "none", "--epsilon" , "0.000", "--model", "ConvNCF", "--model_path", "AT_AWP/param_file/other_model/ConvNCF/AMusic/AMusic_lr-0.001_g-0.005_FT_3367.pth"],   
    # ["--dataset", "AMusic", "--attack", "none", "--epsilon" , "0.000", "--model", "ConvNCF", "--model_path", "AT_AWP/param_file/other_model/ConvNCF/AMusic/AMusic_lr-0.002_g-0.005_PDAF_3418.pth"],   
    
   
    # ["--dataset", "ml-1m", "--attack", "fgsm", "--epsilon" , "0.008", "--model", "NeuMF", "--model_path", "pretrain/NeuMF_PDAF/ml-1m_0.001_0.001/ml-1m_lr-0.001_g-0.001_NeuMF_PDAF.pth"],   
    ["--dataset", "ml-1m", "--attack", "none", "--epsilon" , "0.008", "--model", "NeuMF", "--model_path", "pretrain/NeuMF_PDAF/ml-1m_0.0001_0.001/ml-1m_lr-0.0001_g-0.001_NeuMF_PDAF.pth"],   


]
OUTPUT_PREFIX = "AT_AWP/output/other_model/robust/"

def extract_filename_without_txt(path):
    # 移除末尾的斜杠（如果有）
    path = path.rstrip('/')
    # 提取文件名
    filename = path.split('/')[-1]
    # 移除 .txt 后缀（如果存在）
    if filename.endswith('.pth'):
        filename = filename[:-4]  # 移除最后 4 个字符（.txt）
    return filename

def main():
    for idx, args in enumerate(ARG_COMBINATIONS, 1):
        pre = OUTPUT_PREFIX

        dataset_value = None
        gamma = 0.0
        attack = None
        model_path = None
        model = None
        
        # 遍历 args 找到 --lr 和 --dataset 的值
        for i in range(len(args)):
            if args[i] == "--dataset" and i + 1 < len(args):
                dataset_value = args[i + 1]
            if args[i] == "--epsilon" and i + 1 < len(args):
                gamma = args[i + 1]
            if args[i] == "--attack" and i + 1 < len(args):
                attack = args[i + 1]
            if args[i] == "--model_path" and i + 1 < len(args):
                model_path = args[i + 1]
            if args[i] == "--model" and i + 1 < len(args):
                model = args[i + 1]
        model_path = extract_filename_without_txt(model_path)

        output_file = f"{pre}{model}_{dataset_value}_{model_path}_{gamma}.txt"
        if attack == "none":
            output_file = f"{pre}{model}_{dataset_value}_{model_path}_clean.txt"
        # output_file = f"{pre}FTS_gamma-{0.008}_attack-fgsm.txt"
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