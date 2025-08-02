import re
import os
from pathlib import Path

def parse_log_file(log_file):
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 提取基础信息
    result = {
        "pretrained_model": None,
        "lr": None,
        "gamma": None,
        "epsilon": None,
        "initial_metrics": {},
        "best_model": {
            "hr10": 0,
            "ndcg10": 0,
            "robust_hr10": 0,
            "robust_ndcg10": 0,
            "model_path": None
        }
    }

    # 提取预训练模型路径
    model_path_match = re.search(r"model_path='(.*?)'", content)
    if model_path_match:
        result["pretrained_model"] = model_path_match.group(1)

    # 从Namespace提取参数
    namespace_match = re.search(r"Namespace\((.*?)\)", content)
    if namespace_match:
        namespace_str = namespace_match.group(1)
        result["lr"] = re.search(r"lr=([\d\.]+)", namespace_str).group(1)
        result["gamma"] = re.search(r"awp_gamma=([\d\.]+)", namespace_str).group(1)
        result["epsilon"] = re.search(r"epsilon=([\d\.]+)", namespace_str).group(1)

    # 提取初始指标
    init_match = re.search(
        r"init: HR10=([\d\.]+), NDCG10=([\d\.]+),.*?robust hit10: ([\d\.]+)",
        content
    )
    if init_match:
        result["initial_metrics"] = {
            "hr10": float(init_match.group(1)) * 100,
            "ndcg10": float(init_match.group(2)) * 100,
            "robust_hr10": float(init_match.group(3)) * 100
        }

    # 提取所有epoch的指标
    epochs = re.findall(
        r"==> hits10: ([\d\.]+)%\s+robust_hits10: ([\d\.]+)%",
        content
    )
    
    # 找到最佳robust hr10的epoch
    max_robust = 0
    best_epoch = None
    for epoch in epochs:
        robust = float(epoch[1])
        if robust > max_robust:
            max_robust = robust
            best_epoch = epoch

    # 提取最佳模型指标
    if best_epoch:
        result["best_model"]["hr10"] = float(best_epoch[0])
        result["best_model"]["robust_hr10"] = float(best_epoch[1])
        
    # 提取NDCG指标（需要日志中包含NDCG信息）
    ndcg_match = re.search(
        r"Finetune hits10: [\d\.]+%, robust hit10: [\d\.]+.*?NDCG10=([\d\.]+)",
        content,
        re.DOTALL
    )
    if ndcg_match:
        result["best_model"]["ndcg10"] = float(ndcg_match.group(1)) * 100
        result["best_model"]["robust_ndcg10"] = result["best_model"]["ndcg10"]  # 根据实际情况调整

    # 提取模型保存路径
    model_save_match = re.search(r"save:.*?\n.*?'(.*?)'", content)
    if model_save_match:
        result["best_model"]["model_path"] = model_save_match.group(1)

    return result

def process_logs(input_dir, output_file):
    results = []
    for file in Path(input_dir).glob("*.log"):
        print(f"Processing {file.name}...")
        results.append((file.name, parse_log_file(file)))

    with open(output_file, 'w') as f:
        for filename, data in results:
            f.write(f"File: {filename}\n")
            f.write(f"Pretrained Model: {data['pretrained_model']}\n")
            f.write(f"Parameters: lr={data['lr']}, gamma={data['gamma']}, epsilon={data['epsilon']}\n")
            f.write("Initial Metrics:\n")
            f.write(f"  HR@10: {data['initial_metrics'].get('hr10', 0):.2f}%\n")
            f.write(f"  NDCG@10: {data['initial_metrics'].get('ndcg10', 0):.2f}%\n")
            f.write(f"  Robust HR@10: {data['initial_metrics'].get('robust_hr10', 0):.2f}%\n")
            f.write("\nBest Model:\n")
            f.write(f"  HR@10: {data['best_model']['hr10']:.2f}%\n")
            f.write(f"  NDCG@10: {data['best_model']['ndcg10']:.2f}%\n")
            f.write(f"  Robust HR@10: {data['best_model']['robust_hr10']:.2f}%\n")
            f.write(f"  Robust NDCG@10: {data['best_model']['robust_ndcg10']:.2f}%\n")
            f.write(f"Model Path: {data['best_model']['model_path']}\n")
            f.write("-"*50 + "\n")

if __name__ == "__main__":
    input_directory = "./pretrained"  # 修改为日志目录
    output_file = "results_fts.txt"
    process_logs(input_directory, output_file)
    print(f"Processing completed. Results saved to {output_file}")