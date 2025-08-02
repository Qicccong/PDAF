import subprocess
import sys

# 配置部分
SCRIPT_NAME = "D:\\Users\\Jchu\\Desktop\\RAWPFT250217\\RAWPFT250217\\AT_AWP\\Attack_test_testattack.py"
ARG_COMBINATIONS = [
    # # RAWP
    # ["--attack", "none", "--model_path", "D:\\Users\\Jchu\\Desktop\\RAWPFT250217\\RAWPFT250217\\AT_AWP\\param_file\\RAWP\\79_lastfm_MLP_RAWP_1718355934.5297291.pth"],
    # ["--dataset", "AMusic", "--attack", "none", "--model_path", "D:\\Users\\Jchu\\Desktop\\RAWPFT250217\\RAWPFT250217\\AT_AWP\\param_file\\RAWP\\11_AMusic_MLP_init_1714030591.7578497.pth"],   
    # ["--dataset", "ml-1m", "--attack", "none", "--model_path", "D:\\Users\\Jchu\\Desktop\\RAWPFT250217\\RAWPFT250217\\AT_AWP\\param_file\\RAWP\\80_ml-1m_MLP_init_1713513910.3174512.pth"],
    # ["--dataset", "yelp", "--attack", "none", "--alpha", "0.0008", "--num_steps", "10", "--model_path", "AT_AWP/param_file/RAWP/13_yelp_MLP_adv_1712984667.276178.pth"],

    # # RAWP-FT
    # ["--attack", "none", "--model_path", "D:\\Users\\Jchu\\Desktop\\RAWPFT250217\\RAWPFT250217\\AT_AWP\\param_file\\RAWP-FT\\lastfm_RAWP_79_FineTurning_1718535631.2096848.pth"],
    # ["--dataset", "AMusic", "--attack", "none", "--alpha", "0.0008", "--num_steps", "10", "--model_path", "D:\\Users\\Jchu\\Desktop\\RAWPFT250217\\RAWPFT250217\\AT_AWP\\param_file\\RAWP-FT\\AMusic_RAWP_11_FineTurning_1716879487.5125744.pth"],
    # ["--dataset", "ml-1m", "--attack", "none", "--alpha", "0.0008", "--num_steps", "10", "--model_path", "D:\\Users\\Jchu\\Desktop\\RAWPFT250217\\RAWPFT250217\\AT_AWP\\param_file\\RAWP-FT\\ml-1m_RAWP_80_FineTurning_1715970106.2888427.pth"],
    # ["--dataset", "yelp", "--attack", "none", "--alpha", "0.0008", "--num_steps", "10", "--model_path", "AT_AWP/param_file/RAWP-FT/FT_YELP_1_13_yelp_RAWP_FineTurning_1731292337.0358045.pth"],

    # # RAT
    # ["--attack", "none", "--model_path", "D:\\Users\\Jchu\\Desktop\\RAWPFT250217\\RAWPFT250217\\AT_AWP\\param_file\\RAT\\l2_01_lastfm_0.0005_fgsm_MLP_robust.pth"],
    # ["--dataset", "AMusic", "--attack", "none", "--alpha", "0.0008", "--num_steps", "10", "--model_path", "D:\\Users\\Jchu\\Desktop\\RAWPFT250217\\RAWPFT250217\\AT_AWP\\param_file\\RAT\\07_AMusic_0.0005_fgsm_MLP_robust.pth"],
    # ["--dataset", "ml-1m", "--attack", "none", "--alpha", "0.0008", "--num_steps", "10", "--model_path", "D:\\Users\\Jchu\\Desktop\\RAWPFT250217\\RAWPFT250217\\AT_AWP\\param_file\\RAT\\ml-1m_RAT_0.0005_fgsm_L2_robust_0529_1.pth"],
    # ["--dataset", "yelp", "--attack", "none", "--alpha", "0.0008", "--num_steps", "10", "--model_path", "AT_AWP/param_file/RAT/7_yelp_0.0005_fgsm_MLP_robust.pth"],

    # # MLP
    # ["--attack", "none", "--alpha", "0.0008", "--num_steps", "10", "--model_path", "D:\\Users\\Jchu\\Desktop\\RAWPFT250217\\RAWPFT250217\\AT_AWP\\param_file\\MLP\\65_clean_lastfm_MLP_1710746943.2880716.pth"],
    # ["--dataset", "AMusic", "--attack", "none", "--alpha", "0.0008", "--num_steps", "10", "--model_path", "D:\\Users\\Jchu\\Desktop\\RAWPFT250217\\RAWPFT250217\\AT_AWP\\param_file\\MLP\\01_AMusic_MLP_clean_1713258839.9874997.pth"],
    # ["--dataset", "ml-1m", "--attack", "none", "--alpha", "0.0008", "--num_steps", "10", "--model_path", "D:\\Users\\Jchu\\Desktop\\RAWPFT250217\\RAWPFT250217\\AT_AWP\\param_file\\MLP\\93_ml-1m_0.0001_fgsm_MLP_clean_robust_0521_1.pth"],
    # ["--dataset", "yelp", "--attack", "none", "--alpha", "0.0008", "--num_steps", "10", "--model_path", "AT_AWP/param_file/MLP/0_1_yelp_0.0005_fgsm_MLP_robust.pth"],

    # FTS
    # ["--dataset", "lastfm", "--attack", "none", "--alpha", "0.0008", "--num_steps", "10", "--model_path", "AT_AWP\param_file\FTS\lastfm\lastfm_FTS_lr-0.05_gamma-0.008.pth"],
    # ["--dataset", "AMusic", "--attack", "none", "--alpha", "0.0008", "--num_steps", "10", "--model_path", "AT_AWP\param_file\FTS\AMusic\AMusic_FTS_lr-0.1_gamma-0.005.pth"],
    # ["--dataset", "ml-1m", "--attack", "none", "--alpha", "0.0008", "--num_steps", "10", "--model_path", "AT_AWP\param_file\FTS\ML-1M\ml-1m_FTS_lr-0.005_gamma-0.001.pth"],
    # ["--dataset", "yelp", "--attack", "none", "--alpha", "0.0008", "--num_steps", "10", "--model_path", ""],
    
    # 消融
    # ["--dataset", "ml-1m", "--attack", "none", "--alpha", "0.0008", "--num_steps", "10", "--model_path", "AT_AWP/param_file/best_model/ml-1m_FTS_lr-0.005_gamma-0.001_layer_frozen.pth"],
    # ["--dataset", "ml-1m", "--attack", "none", "--alpha", "0.0008", "--num_steps", "10", "--model_path", "AT_AWP/param_file/best_model/ml-1m_FTS_lr-0.005_gamma-0.001_gamma_frozen.pth"],
    ["--dataset", "ml-1m", "--attack", "none", "--alpha", "0.0008", "--num_steps", "10", "--model_path", "AT_AWP/param_file/FTS/ml-1m_lr-0.005_g-0.001_random_gamma.pth"],
    

    # lr
    # ["--dataset", "lastfm", "--attack", "fgsm", "--epsilon", "0.008", "--num_steps", "10", "--model_path", "AT_AWP\param_file\FTS\lastfm\lastfm_FTS_lr-0.01_gamma-0.008.pth"],
    # ["--dataset", "lastfm", "--attack", "fgsm", "--epsilon", "0.008", "--num_steps", "10", "--model_path", "AT_AWP\param_file\FTS\lastfm\lastfm_FTS_lr-0.02_gamma-0.008.pth"],
    # ["--dataset", "lastfm", "--attack", "fgsm", "--epsilon", "0.008", "--num_steps", "10", "--model_path", "AT_AWP\param_file\FTS\lastfm\lastfm_FTS_lr-0.05_gamma-0.008.pth"],
    # ["--dataset", "lastfm", "--attack", "fgsm", "--epsilon", "0.008", "--num_steps", "10", "--model_path", "AT_AWP\param_file\FTS\lastfm\lastfm_FTS_lr-0.1_gamma-0.008.pth"],
    # ["--dataset", "lastfm", "--attack", "fgsm", "--epsilon", "0.008", "--num_steps", "10", "--model_path", "AT_AWP\param_file\FTS\lastfm\lastfm_FTS_lr-0.2_gamma-0.008.pth"],
    # ["--dataset", "lastfm", "--attack", "fgsm", "--epsilon", "0.008", "--num_steps", "10", "--model_path", "AT_AWP\param_file\FTS\lastfm\lastfm_FTS_lr-0.5_gamma-0.008.pth"],
    
    # ["--dataset", "AMusic", "--attack", "fgsm", "--epsilon", "0.008", "--num_steps", "10", "--model_path", "AT_AWP\param_file\FTS\AMusic\AMusic_FTS_lr-0.01_gamma-0.005.pth"],
    # ["--dataset", "AMusic", "--attack", "fgsm", "--epsilon", "0.008", "--num_steps", "10", "--model_path", "AT_AWP\param_file\FTS\AMusic\AMusic_FTS_lr-0.02_gamma-0.005.pth"],
    # ["--dataset", "AMusic", "--attack", "fgsm", "--epsilon", "0.008", "--num_steps", "10", "--model_path", "AT_AWP\param_file\FTS\AMusic\AMusic_FTS_lr-0.05_gamma-0.005.pth"],
    # ["--dataset", "AMusic", "--attack", "fgsm", "--epsilon", "0.008", "--num_steps", "10", "--model_path", "AT_AWP\param_file\FTS\AMusic\AMusic_FTS_lr-0.1_gamma-0.005.pth"],
    # ["--dataset", "AMusic", "--attack", "fgsm", "--epsilon", "0.008", "--num_steps", "10", "--model_path", "AT_AWP\param_file\FTS\AMusic\AMusic_FTS_lr-0.2_gamma-0.005.pth"],
    # ["--dataset", "AMusic", "--attack", "fgsm", "--epsilon", "0.008", "--num_steps", "10", "--model_path", "AT_AWP\param_file\FTS\AMusic\AMusic_FTS_lr-0.5_gamma-0.005.pth"],
    
    # ["--dataset", "ml-1m", "--attack", "fgsm", "--epsilon", "0.008", "--num_steps", "10", "--model_path", "AT_AWP\param_file\FTS\ML-1M\ml-1m_FTS_lr-0.0005_gamma-0.001.pth"],
    # ["--dataset", "ml-1m", "--attack", "fgsm", "--epsilon", "0.008", "--num_steps", "10", "--model_path", "AT_AWP\param_file\FTS\ML-1M\ml-1m_FTS_lr-0.001_gamma-0.001.pth"],
    # ["--dataset", "ml-1m", "--attack", "fgsm", "--epsilon", "0.008", "--num_steps", "10", "--model_path", "AT_AWP\param_file\FTS\ML-1M\ml-1m_FTS_lr-0.002_gamma-0.001.pth"],
    # ["--dataset", "ml-1m", "--attack", "fgsm", "--epsilon", "0.008", "--num_steps", "10", "--model_path", "AT_AWP\param_file\\best_model\ml-1m_FTS_lr-0.005_gamma-0.001_Best.pth"],
    # ["--dataset", "ml-1m", "--attack", "fgsm", "--epsilon", "0.008", "--num_steps", "10", "--model_path", "AT_AWP\param_file\FTS\ML-1M\ml-1m_FTS_lr-0.01_gamma-0.001.pth"],
    # ["--dataset", "ml-1m", "--attack", "fgsm", "--epsilon", "0.008", "--num_steps", "10", "--model_path", "AT_AWP\param_file\FTS\ML-1M\ml-1m_FTS_lr-0.02_gamma-0.001.pth"],
    
    
]
OUTPUT_PREFIX1 = "AT_AWP\\output\\RAWP_clean"
OUTPUT_PREFIX2 = "AT_AWP\\output\\RAWP-FT_clean"
OUTPUT_PREFIX3 = "AT_AWP\\output\\RAT_clean"
OUTPUT_PREFIX4 = "AT_AWP\\output\\MLP_clean"
OUTPUT_PREFIX5 = "AT_AWP\\output\\FTS-clean\\"
OUTPUT_PREFIX = "AT_AWP\\output\\FTS\\"

def main():
    for idx, args in enumerate(ARG_COMBINATIONS, 1):
        # if idx <= 1:
        #     pre = OUTPUT_PREFIX1
        # elif idx <= 2:
        #     pre = OUTPUT_PREFIX2
        # elif idx <= 3:
        #     pre = OUTPUT_PREFIX3
        # else: 
        #     pre = OUTPUT_PREFIX4
        # pre = OUTPUT_PREFIX5
        pre = OUTPUT_PREFIX5

        # 提取 lr 和 dataset
        lr_value = None
        dataset_value = None
        gamma = None
        model_name = None

        if idx <= 1:
            model_name = "RAWP"
        elif idx <= 2:
            model_name = "RAWP-FT"
        elif idx <= 3:
            model_name = "RAT"
        else: 
            model_name = "MLP"
        model_name = "FTS"
        
        # 遍历 args 找到 --lr 和 --dataset 的值
        for i in range(len(args)):
            if args[i] == "--dataset" and i + 1 < len(args):
                dataset_value = args[i + 1]

        if dataset_value is None:
            dataset_value = "none"
        
        if idx <= 1:
            fix = "layer_frozen"
        elif idx <= 2:
            fix = "gamma_frozen"
        else:
            assert "错误"

        # output_file = f"{pre}{model_name}_robust_{dataset_value}.txt"
        output_file = f"{pre}FTS_clean_{dataset_value}.txt"
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