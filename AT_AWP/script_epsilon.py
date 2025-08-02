import subprocess
import sys

# 配置部分
SCRIPT_NAME = "D:\\Users\\Jchu\\Desktop\\RAWPFT250217\\RAWPFT250217\\AT_AWP\\Attack_test_testattack.py"
ARG_COMBINATIONS = [
    # RAWP 
    # ["--epsilon", "0.005", "--alpha", "0.0005", "--dataset", "ml-1m", "--attack", "fgsm", "--model_path", "D:\\Users\\Jchu\\Desktop\\RAWPFT250217\\RAWPFT250217\\AT_AWP\\param_file\\RAWP\\80_ml-1m_MLP_init_1713513910.3174512.pth"],
    # ["--epsilon", "0.008", "--alpha", "0.0008", "--dataset", "ml-1m", "--attack", "fgsm", "--model_path", "D:\\Users\\Jchu\\Desktop\\RAWPFT250217\\RAWPFT250217\\AT_AWP\\param_file\\RAWP\\80_ml-1m_MLP_init_1713513910.3174512.pth"],
    # ["--epsilon", "0.010", "--alpha", "0.0010", "--dataset", "ml-1m", "--attack", "fgsm", "--model_path", "D:\\Users\\Jchu\\Desktop\\RAWPFT250217\\RAWPFT250217\\AT_AWP\\param_file\\RAWP\\80_ml-1m_MLP_init_1713513910.3174512.pth"],
    # ["--epsilon", "0.015", "--alpha", "0.0015", "--dataset", "ml-1m", "--attack", "fgsm", "--model_path", "D:\\Users\\Jchu\\Desktop\\RAWPFT250217\\RAWPFT250217\\AT_AWP\\param_file\\RAWP\\80_ml-1m_MLP_init_1713513910.3174512.pth"],
    # ["--epsilon", "0.020", "--alpha", "0.0020", "--dataset", "ml-1m", "--attack", "fgsm", "--model_path", "D:\\Users\\Jchu\\Desktop\\RAWPFT250217\\RAWPFT250217\\AT_AWP\\param_file\\RAWP\\80_ml-1m_MLP_init_1713513910.3174512.pth"],
    # ["--epsilon", "0.030", "--alpha", "0.0030", "--dataset", "ml-1m", "--attack", "fgsm", "--model_path", "D:\\Users\\Jchu\\Desktop\\RAWPFT250217\\RAWPFT250217\\AT_AWP\\param_file\\RAWP\\80_ml-1m_MLP_init_1713513910.3174512.pth"],
    # ["--epsilon", "0.050", "--alpha", "0.0050", "--dataset", "ml-1m", "--attack", "fgsm", "--model_path", "D:\\Users\\Jchu\\Desktop\\RAWPFT250217\\RAWPFT250217\\AT_AWP\\param_file\\RAWP\\80_ml-1m_MLP_init_1713513910.3174512.pth"],
    # ["--epsilon", "0.070", "--alpha", "0.0070", "--dataset", "ml-1m", "--attack", "fgsm", "--model_path", "D:\\Users\\Jchu\\Desktop\\RAWPFT250217\\RAWPFT250217\\AT_AWP\\param_file\\RAWP\\80_ml-1m_MLP_init_1713513910.3174512.pth"],
    # ["--epsilon", "0.100", "--alpha", "0.0100", "--dataset", "ml-1m", "--attack", "fgsm", "--model_path", "D:\\Users\\Jchu\\Desktop\\RAWPFT250217\\RAWPFT250217\\AT_AWP\\param_file\\RAWP\\80_ml-1m_MLP_init_1713513910.3174512.pth"],
    
    # ["--epsilon", "0.007", "--alpha", "0.0007", "--dataset", "ml-1m", "--attack", "fgsm", "--model_path", "D:\\Users\\Jchu\\Desktop\\RAWPFT250217\\RAWPFT250217\\AT_AWP\\param_file\\RAWP\\80_ml-1m_MLP_init_1713513910.3174512.pth"],
    # ["--epsilon", "0.008", "--alpha", "0.0008", "--dataset", "ml-1m", "--attack", "fgsm", "--model_path", "D:\\Users\\Jchu\\Desktop\\RAWPFT250217\\RAWPFT250217\\AT_AWP\\param_file\\RAWP\\80_ml-1m_MLP_init_1713513910.3174512.pth"],
    # ["--epsilon", "0.009", "--alpha", "0.0009", "--dataset", "ml-1m", "--attack", "fgsm", "--model_path", "D:\\Users\\Jchu\\Desktop\\RAWPFT250217\\RAWPFT250217\\AT_AWP\\param_file\\RAWP\\80_ml-1m_MLP_init_1713513910.3174512.pth"],
    # ["--epsilon", "0.010", "--alpha", "0.0010", "--dataset", "ml-1m", "--attack", "fgsm", "--model_path", "D:\\Users\\Jchu\\Desktop\\RAWPFT250217\\RAWPFT250217\\AT_AWP\\param_file\\RAWP\\80_ml-1m_MLP_init_1713513910.3174512.pth"],

    # ["--epsilon", "0.008", "--alpha", "0.0008", "--dataset", "lastfm", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/RAWP/79_lastfm_MLP_RAWP_1718355934.5297291.pth"],
    # ["--epsilon", "0.015", "--alpha", "0.0015", "--dataset", "lastfm", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/RAWP/79_lastfm_MLP_RAWP_1718355934.5297291.pth"],
    # ["--epsilon", "0.030", "--alpha", "0.0030", "--dataset", "lastfm", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/RAWP/79_lastfm_MLP_RAWP_1718355934.5297291.pth"],
    # ["--epsilon", "0.050", "--alpha", "0.0050", "--dataset", "lastfm", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/RAWP/79_lastfm_MLP_RAWP_1718355934.5297291.pth"],
    
    # ["--epsilon", "0.008", "--alpha", "0.0008", "--dataset", "AMusic", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/RAWP/11_AMusic_MLP_init_1714030591.7578497.pth"],
    # ["--epsilon", "0.015", "--alpha", "0.0015", "--dataset", "AMusic", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/RAWP/11_AMusic_MLP_init_1714030591.7578497.pth"],
    # ["--epsilon", "0.030", "--alpha", "0.0030", "--dataset", "AMusic", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/RAWP/11_AMusic_MLP_init_1714030591.7578497.pth"],
    # ["--epsilon", "0.050", "--alpha", "0.0050", "--dataset", "AMusic", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/RAWP/11_AMusic_MLP_init_1714030591.7578497.pth"],
    
    # ["--epsilon", "0.008", "--alpha", "0.0008", "--dataset", "yelp", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/RAWP/13_yelp_MLP_adv_1712984667.276178.pth"],
    # ["--epsilon", "0.015", "--alpha", "0.0015", "--dataset", "yelp", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/RAWP/13_yelp_MLP_adv_1712984667.276178.pth"],
    # ["--epsilon", "0.030", "--alpha", "0.0030", "--dataset", "yelp", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/RAWP/13_yelp_MLP_adv_1712984667.276178.pth"],
    # ["--epsilon", "0.050", "--alpha", "0.0050", "--dataset", "yelp", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/RAWP/13_yelp_MLP_adv_1712984667.276178.pth"],
    

    # # RAWP-FT
    # ["--epsilon", "0.005", "--alpha", "0.0005", "--dataset", "ml-1m", "--attack", "fgsm", "--model_path", "D:\\Users\\Jchu\\Desktop\\RAWPFT250217\\RAWPFT250217\\AT_AWP\\param_file\\RAWP-FT\\ml-1m_RAWP_80_FineTurning_1715970106.2888427.pth"],
    # ["--epsilon", "0.008", "--alpha", "0.0008", "--dataset", "ml-1m", "--attack", "fgsm", "--model_path", "D:\\Users\\Jchu\\Desktop\\RAWPFT250217\\RAWPFT250217\\AT_AWP\\param_file\\RAWP-FT\\ml-1m_RAWP_80_FineTurning_1715970106.2888427.pth"],
    # ["--epsilon", "0.010", "--alpha", "0.0010", "--dataset", "ml-1m", "--attack", "fgsm", "--model_path", "D:\\Users\\Jchu\\Desktop\\RAWPFT250217\\RAWPFT250217\\AT_AWP\\param_file\\RAWP-FT\\ml-1m_RAWP_80_FineTurning_1715970106.2888427.pth"],
    # ["--epsilon", "0.015", "--alpha", "0.0015", "--dataset", "ml-1m", "--attack", "fgsm", "--model_path", "D:\\Users\\Jchu\\Desktop\\RAWPFT250217\\RAWPFT250217\\AT_AWP\\param_file\\RAWP-FT\\ml-1m_RAWP_80_FineTurning_1715970106.2888427.pth"],
    # ["--epsilon", "0.020", "--alpha", "0.0020", "--dataset", "ml-1m", "--attack", "fgsm", "--model_path", "D:\\Users\\Jchu\\Desktop\\RAWPFT250217\\RAWPFT250217\\AT_AWP\\param_file\\RAWP-FT\\ml-1m_RAWP_80_FineTurning_1715970106.2888427.pth"],
    # ["--epsilon", "0.030", "--alpha", "0.0030", "--dataset", "ml-1m", "--attack", "fgsm", "--model_path", "D:\\Users\\Jchu\\Desktop\\RAWPFT250217\\RAWPFT250217\\AT_AWP\\param_file\\RAWP-FT\\ml-1m_RAWP_80_FineTurning_1715970106.2888427.pth"],
    # ["--epsilon", "0.050", "--alpha", "0.0050", "--dataset", "ml-1m", "--attack", "fgsm", "--model_path", "D:\\Users\\Jchu\\Desktop\\RAWPFT250217\\RAWPFT250217\\AT_AWP\\param_file\\RAWP-FT\\ml-1m_RAWP_80_FineTurning_1715970106.2888427.pth"],
    # ["--epsilon", "0.070", "--alpha", "0.0070", "--dataset", "ml-1m", "--attack", "fgsm", "--model_path", "D:\\Users\\Jchu\\Desktop\\RAWPFT250217\\RAWPFT250217\\AT_AWP\\param_file\\RAWP-FT\\ml-1m_RAWP_80_FineTurning_1715970106.2888427.pth"],
    # ["--epsilon", "0.100", "--alpha", "0.0100", "--dataset", "ml-1m", "--attack", "fgsm", "--model_path", "D:\\Users\\Jchu\\Desktop\\RAWPFT250217\\RAWPFT250217\\AT_AWP\\param_file\\RAWP-FT\\ml-1m_RAWP_80_FineTurning_1715970106.2888427.pth"],
    
    # ["--epsilon", "0.007", "--alpha", "0.0007", "--dataset", "ml-1m", "--attack", "fgsm", "--model_path", "D:\\Users\\Jchu\\Desktop\\RAWPFT250217\\RAWPFT250217\\AT_AWP\\param_file\\RAWP-FT\\ml-1m_RAWP_80_FineTurning_1715970106.2888427.pth"],
    # ["--epsilon", "0.008", "--alpha", "0.0008", "--dataset", "ml-1m", "--attack", "fgsm", "--model_path", "D:\\Users\\Jchu\\Desktop\\RAWPFT250217\\RAWPFT250217\\AT_AWP\\param_file\\RAWP-FT\\ml-1m_RAWP_80_FineTurning_1715970106.2888427.pth"],
    # ["--epsilon", "0.009", "--alpha", "0.0009", "--dataset", "ml-1m", "--attack", "fgsm", "--model_path", "D:\\Users\\Jchu\\Desktop\\RAWPFT250217\\RAWPFT250217\\AT_AWP\\param_file\\RAWP-FT\\ml-1m_RAWP_80_FineTurning_1715970106.2888427.pth"],
    # ["--epsilon", "0.010", "--alpha", "0.0010", "--dataset", "ml-1m", "--attack", "fgsm", "--model_path", "D:\\Users\\Jchu\\Desktop\\RAWPFT250217\\RAWPFT250217\\AT_AWP\\param_file\\RAWP-FT\\ml-1m_RAWP_80_FineTurning_1715970106.2888427.pth"],

    # ["--epsilon", "0.008", "--alpha", "0.0008", "--dataset", "lastfm", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/RAWP-FT/lastfm_RAWP_79_FineTurning_1718535631.2096848.pth"],
    # ["--epsilon", "0.015", "--alpha", "0.0015", "--dataset", "lastfm", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/RAWP-FT/lastfm_RAWP_79_FineTurning_1718535631.2096848.pth"],
    # ["--epsilon", "0.030", "--alpha", "0.0030", "--dataset", "lastfm", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/RAWP-FT/lastfm_RAWP_79_FineTurning_1718535631.2096848.pth"],
    # ["--epsilon", "0.050", "--alpha", "0.0050", "--dataset", "lastfm", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/RAWP-FT/lastfm_RAWP_79_FineTurning_1718535631.2096848.pth"],
    
    # ["--epsilon", "0.008", "--alpha", "0.0008", "--dataset", "AMusic", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/RAWP-FT/AMusic_RAWP_11_FineTurning_1716879487.5125744.pth"],
    # ["--epsilon", "0.015", "--alpha", "0.0015", "--dataset", "AMusic", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/RAWP-FT/AMusic_RAWP_11_FineTurning_1716879487.5125744.pth"],
    # ["--epsilon", "0.030", "--alpha", "0.0030", "--dataset", "AMusic", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/RAWP-FT/AMusic_RAWP_11_FineTurning_1716879487.5125744.pth"],
    # ["--epsilon", "0.050", "--alpha", "0.0050", "--dataset", "AMusic", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/RAWP-FT/AMusic_RAWP_11_FineTurning_1716879487.5125744.pth"],
    
    # ["--epsilon", "0.008", "--alpha", "0.0008", "--dataset", "yelp", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/RAWP-FT/FT_YELP_1_13_yelp_RAWP_FineTurning_1731292337.0358045.pth"],
    # ["--epsilon", "0.015", "--alpha", "0.0015", "--dataset", "yelp", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/RAWP-FT/FT_YELP_1_13_yelp_RAWP_FineTurning_1731292337.0358045.pth"],
    # ["--epsilon", "0.030", "--alpha", "0.0030", "--dataset", "yelp", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/RAWP-FT/FT_YELP_1_13_yelp_RAWP_FineTurning_1731292337.0358045.pth"],
    # ["--epsilon", "0.050", "--alpha", "0.0050", "--dataset", "yelp", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/RAWP-FT/FT_YELP_1_13_yelp_RAWP_FineTurning_1731292337.0358045.pth"],
    

    # # RAT
    # ["--epsilon", "0.007", "--alpha", "0.0007", "--dataset", "ml-1m", "--attack", "fgsm", "--model_path", "D:\\Users\\Jchu\\Desktop\\RAWPFT250217\\RAWPFT250217\\AT_AWP\\param_file\\RAT\\ml-1m_RAT_0.0005_fgsm_L2_robust_0529_1.pth"],
    # ["--epsilon", "0.008", "--alpha", "0.0008", "--dataset", "ml-1m", "--attack", "fgsm", "--model_path", "D:\\Users\\Jchu\\Desktop\\RAWPFT250217\\RAWPFT250217\\AT_AWP\\param_file\\RAT\\ml-1m_RAT_0.0005_fgsm_L2_robust_0529_1.pth"],
    # ["--epsilon", "0.009", "--alpha", "0.0009", "--dataset", "ml-1m", "--attack", "fgsm", "--model_path", "D:\\Users\\Jchu\\Desktop\\RAWPFT250217\\RAWPFT250217\\AT_AWP\\param_file\\RAT\\ml-1m_RAT_0.0005_fgsm_L2_robust_0529_1.pth"],
    # ["--epsilon", "0.010", "--alpha", "0.0010", "--dataset", "ml-1m", "--attack", "fgsm", "--model_path", "D:\\Users\\Jchu\\Desktop\\RAWPFT250217\\RAWPFT250217\\AT_AWP\\param_file\\RAT\\ml-1m_RAT_0.0005_fgsm_L2_robust_0529_1.pth"],

    # ["--epsilon", "0.008", "--alpha", "0.0008", "--dataset", "lastfm", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/RAT/l2_01_lastfm_0.0005_fgsm_MLP_robust.pth"],
    # ["--epsilon", "0.015", "--alpha", "0.0015", "--dataset", "lastfm", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/RAT/l2_01_lastfm_0.0005_fgsm_MLP_robust.pth"],
    # ["--epsilon", "0.030", "--alpha", "0.0030", "--dataset", "lastfm", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/RAT/l2_01_lastfm_0.0005_fgsm_MLP_robust.pth"],
    # ["--epsilon", "0.050", "--alpha", "0.0050", "--dataset", "lastfm", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/RAT/l2_01_lastfm_0.0005_fgsm_MLP_robust.pth"],
    
    # ["--epsilon", "0.008", "--alpha", "0.0008", "--dataset", "AMusic", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/RAT/07_AMusic_0.0005_fgsm_MLP_robust.pth"],
    # ["--epsilon", "0.015", "--alpha", "0.0015", "--dataset", "AMusic", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/RAT/07_AMusic_0.0005_fgsm_MLP_robust.pth"],
    # ["--epsilon", "0.030", "--alpha", "0.0030", "--dataset", "AMusic", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/RAT/07_AMusic_0.0005_fgsm_MLP_robust.pth"],
    # ["--epsilon", "0.050", "--alpha", "0.0050", "--dataset", "AMusic", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/RAT/07_AMusic_0.0005_fgsm_MLP_robust.pth"],
    
    # ["--epsilon", "0.008", "--alpha", "0.0008", "--dataset", "yelp", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/RAT/7_yelp_0.0005_fgsm_MLP_robust.pth"],
    # ["--epsilon", "0.015", "--alpha", "0.0015", "--dataset", "yelp", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/RAT/7_yelp_0.0005_fgsm_MLP_robust.pth"],
    # ["--epsilon", "0.030", "--alpha", "0.0030", "--dataset", "yelp", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/RAT/7_yelp_0.0005_fgsm_MLP_robust.pth"],
    # ["--epsilon", "0.050", "--alpha", "0.0050", "--dataset", "yelp", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/RAT/7_yelp_0.0005_fgsm_MLP_robust.pth"],
    

    # # MLP
    # ["--epsilon", "0.007", "--alpha", "0.0007", "--dataset", "ml-1m", "--attack", "fgsm", "--model_path", "D:\\Users\\Jchu\\Desktop\\RAWPFT250217\\RAWPFT250217\\AT_AWP\\param_file\\MLP\\93_ml-1m_0.0001_fgsm_MLP_clean_robust_0521_1.pth"],
    # ["--epsilon", "0.008", "--alpha", "0.0008", "--dataset", "ml-1m", "--attack", "fgsm", "--model_path", "D:\\Users\\Jchu\\Desktop\\RAWPFT250217\\RAWPFT250217\\AT_AWP\\param_file\\MLP\\93_ml-1m_0.0001_fgsm_MLP_clean_robust_0521_1.pth"],
    # ["--epsilon", "0.009", "--alpha", "0.0009", "--dataset", "ml-1m", "--attack", "fgsm", "--model_path", "D:\\Users\\Jchu\\Desktop\\RAWPFT250217\\RAWPFT250217\\AT_AWP\\param_file\\MLP\\93_ml-1m_0.0001_fgsm_MLP_clean_robust_0521_1.pth"],
    # ["--epsilon", "0.010", "--alpha", "0.0010", "--dataset", "ml-1m", "--attack", "fgsm", "--model_path", "D:\\Users\\Jchu\\Desktop\\RAWPFT250217\\RAWPFT250217\\AT_AWP\\param_file\\MLP\\93_ml-1m_0.0001_fgsm_MLP_clean_robust_0521_1.pth"],

    # ["--epsilon", "0.008", "--alpha", "0.0008", "--dataset", "lastfm", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/MLP/65_clean_lastfm_MLP_1710746943.2880716.pth"],
    # ["--epsilon", "0.015", "--alpha", "0.0015", "--dataset", "lastfm", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/MLP/65_clean_lastfm_MLP_1710746943.2880716.pth"],
    # ["--epsilon", "0.030", "--alpha", "0.0030", "--dataset", "lastfm", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/MLP/65_clean_lastfm_MLP_1710746943.2880716.pth"],
    # ["--epsilon", "0.050", "--alpha", "0.0050", "--dataset", "lastfm", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/MLP/65_clean_lastfm_MLP_1710746943.2880716.pth"],
    
    # ["--epsilon", "0.008", "--alpha", "0.0008", "--dataset", "AMusic", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/MLP/01_AMusic_MLP_clean_1713258839.9874997.pth"],
    # ["--epsilon", "0.015", "--alpha", "0.0015", "--dataset", "AMusic", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/MLP/01_AMusic_MLP_clean_1713258839.9874997.pth"],
    # ["--epsilon", "0.030", "--alpha", "0.0030", "--dataset", "AMusic", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/MLP/01_AMusic_MLP_clean_1713258839.9874997.pth"],
    # ["--epsilon", "0.050", "--alpha", "0.0050", "--dataset", "AMusic", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/MLP/01_AMusic_MLP_clean_1713258839.9874997.pth"],
    
    # ["--epsilon", "0.008", "--alpha", "0.0008", "--dataset", "yelp", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/MLP/0_1_yelp_0.0005_fgsm_MLP_robust.pth"],
    # ["--epsilon", "0.015", "--alpha", "0.0015", "--dataset", "yelp", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/MLP/0_1_yelp_0.0005_fgsm_MLP_robust.pth"],
    # ["--epsilon", "0.030", "--alpha", "0.0030", "--dataset", "yelp", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/MLP/0_1_yelp_0.0005_fgsm_MLP_robust.pth"],
    # ["--epsilon", "0.050", "--alpha", "0.0050", "--dataset", "yelp", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/MLP/0_1_yelp_0.0005_fgsm_MLP_robust.pth"],
    

    # FTS
    ["--epsilon", "0.008", "--alpha", "0.0008", "--dataset", "ml-1m", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/FTS/ml-1m_lr-0.005_g-0.001_random_gamma.pth"],
    # ["--epsilon", "0.008", "--alpha", "0.0008", "--dataset", "ml-1m", "--attack", "fgsm", "--model_path", ""],
    # ["--epsilon", "0.010", "--alpha", "0.0010", "--dataset", "ml-1m", "--attack", "fgsm", "--model_path", ""],
    # ["--epsilon", "0.015", "--alpha", "0.0015", "--dataset", "ml-1m", "--attack", "fgsm", "--model_path", ""],
    # ["--epsilon", "0.020", "--alpha", "0.0020", "--dataset", "ml-1m", "--attack", "fgsm", "--model_path", ""],
    # ["--epsilon", "0.030", "--alpha", "0.0030", "--dataset", "ml-1m", "--attack", "fgsm", "--model_path", ""],
    # ["--epsilon", "0.050", "--alpha", "0.0050", "--dataset", "ml-1m", "--attack", "fgsm", "--model_path", ""],
    # ["--epsilon", "0.070", "--alpha", "0.0070", "--dataset", "ml-1m", "--attack", "fgsm", "--model_path", ""],
    # ["--epsilon", "0.100", "--alpha", "0.0100", "--dataset", "ml-1m", "--attack", "fgsm", "--model_path", ""],

    # ["--epsilon", "0.008", "--alpha", "0.0008", "--dataset", "lastfm", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/FTS/lastfm/lastfm_FTS_lr-0.05_gamma-0.008.pth"],
    # ["--epsilon", "0.015", "--alpha", "0.0015", "--dataset", "lastfm", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/FTS/lastfm/lastfm_FTS_lr-0.05_gamma-0.008.pth"],
    # ["--epsilon", "0.030", "--alpha", "0.0030", "--dataset", "lastfm", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/FTS/lastfm/lastfm_FTS_lr-0.05_gamma-0.008.pth"],
    # ["--epsilon", "0.050", "--alpha", "0.0050", "--dataset", "lastfm", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/FTS/lastfm/lastfm_FTS_lr-0.05_gamma-0.008.pth"],
    
    # ["--epsilon", "0.008", "--alpha", "0.0008", "--dataset", "AMusic", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/FTS/AMusic/AMusic_FTS_lr-0.1_gamma-0.005.pth"],
    # ["--epsilon", "0.015", "--alpha", "0.0015", "--dataset", "AMusic", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/FTS/AMusic/AMusic_FTS_lr-0.1_gamma-0.005.pth"],
    # ["--epsilon", "0.030", "--alpha", "0.0030", "--dataset", "AMusic", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/FTS/AMusic/AMusic_FTS_lr-0.1_gamma-0.005.pth"],
    # ["--epsilon", "0.050", "--alpha", "0.0050", "--dataset", "AMusic", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/FTS/AMusic/AMusic_FTS_lr-0.1_gamma-0.005.pth"],
    
    # ["--epsilon", "0.008", "--alpha", "0.0008", "--dataset", "yelp", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/FTS/AMusic/AMusic_FTS_lr-0.1_gamma-0.005.pth"],
    # ["--epsilon", "0.015", "--alpha", "0.0015", "--dataset", "yelp", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/FTS/AMusic/AMusic_FTS_lr-0.1_gamma-0.005.pth"],
    # ["--epsilon", "0.030", "--alpha", "0.0030", "--dataset", "yelp", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/FTS/AMusic/AMusic_FTS_lr-0.1_gamma-0.005.pth"],
    # ["--epsilon", "0.050", "--alpha", "0.0050", "--dataset", "yelp", "--attack", "fgsm", "--model_path", "AT_AWP/param_file/FTS/AMusic/AMusic_FTS_lr-0.1_gamma-0.005.pth"],
    
]
OUTPUT_PREFIX1 = "AT_AWP\\output\\epsilon\\RAWP"
OUTPUT_PREFIX2 = "AT_AWP\\output\\epsilon\\RAWP-FT"
OUTPUT_PREFIX3 = "AT_AWP\\output\\epsilon\\FTS"
OUTPUT_PREFIX4 = "AT_AWP\\output\\epsilon\\MLP"
OUTPUT_PREFIX5 = "AT_AWP\\output\\epsilon\\RAT"
OUTPUT_PREFIX6 = "AT_AWP\\output\\FTS"

def main():
    for idx, args in enumerate(ARG_COMBINATIONS, 1):
        if idx <= 4:
            pre = OUTPUT_PREFIX1
        elif idx <= 8:
            pre = OUTPUT_PREFIX2
        elif idx <= 12:
            pre = OUTPUT_PREFIX5
        elif idx <= 16:
            pre = OUTPUT_PREFIX4
        elif idx <= 24:
            pre = OUTPUT_PREFIX3
        pre = OUTPUT_PREFIX6
        
        dataset_value = None
        gamma = None
        
        # 遍历 args 找到 --lr 和 --dataset 的值
        for i in range(len(args)):
            if args[i] == "--dataset" and i + 1 < len(args):
                dataset_value = args[i + 1]
            if args[i] == "--epsilon" and i + 1 < len(args):
                gamma = args[i + 1]
        if dataset_value is None:
            dataset_value = "none"
        if gamma is None:
            gamma = 0.008

        output_file = f"{pre}/{dataset_value}_epsilon-{gamma}.txt"
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