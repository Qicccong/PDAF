import re

def parse_log(log_file):
    # 初始化数据结构
    init_hr10 = None
    namespace_params = {'lr_max': None, 'awp_gamma': None, 'dataset': None}
    seed_data = []
    epoch_info = {}
    save_files = {}
    current_epoch = None
    last_save_epoch = None

    # 预定义阈值字典（示例值，需根据实际情况修改）
    thresholds = {
        # (lr_max, dataset_name) : 阈值配置
        (0.0001, 'AMusic'): {'HR10': 0.4071, 'NDCG10': 0.2423},
        (0.0005, 'AMusic'): {'HR10': 0.4127, 'NDCG10': 0.2467},
        (0.0001, 'ml-1m'): {'HR10': 0.7154, 'NDCG10': 0.4352},
        (0.0005, 'ml-1m'): {'HR10': 0.7111, 'NDCG10': 0.4341},
    }

    with open(log_file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.strip()
            
            # 解析Namespace参数
            if not namespace_params['lr_max']:
                # 匹配Namespace行
                namespace_match = re.search(r'Namespace\(.*\)', line)
                if namespace_match:
                    # 提取Namespace中的参数
                    namespace_str = namespace_match.group(0)
                    # 去掉Namespace(和)
                    namespace_str = namespace_str[10:-1]
                    # 按逗号分隔参数
                    params = namespace_str.split(',')
                    # 解析每个参数
                    for param in params:
                        key_value = param.strip().split('=')
                        if len(key_value) == 2:
                            key = key_value[0].strip()
                            value = key_value[1].strip()
                            if key == 'lr_max':
                                namespace_params['lr_max'] = float(value)
                            elif key == 'awp_gamma':
                                namespace_params['awp_gamma'] = float(value)
                            elif key == 'dataset':
                                # 去掉可能的引号
                                namespace_params['dataset'] = value.strip("'")
                    continue

            seed_pattern = re.compile(
                r'^\[([\d,\s]+)\]'  # 匹配数组部分
                r'\s+([\d.]+(?:e-?\d+)?)'  # 匹配科学计数法数值
                r'\s+(\d+)$'  # 匹配最后的整数
            )
            # 在文件解析循环中
            if seed_pattern.match(line):
                match = seed_pattern.match(line)
                seed_values = list(map(int, match.group(1).split(',')))
                actual_awp_gamma = float(match.group(2))
                seed_flag = int(match.group(3))
                seed_data.append( (seed_values, actual_awp_gamma) )
                continue

            # 解析Epoch信息
            epoch_match = re.search(
                r'Epoch (\d+):.*HR10=([0-9.]+).*NDCG10=([0-9.]+)',
                line
            )
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                epoch_info[current_epoch] = {
                    'normal_hr10': float(epoch_match.group(2)),
                    'ndcg10': float(epoch_match.group(3)),
                    'robust_hr10': None,
                    'robust_ndcg10': None,
                    'save_file': None
                }
                continue

            # 解析Robust信息
            robust_match = re.search(
                r'Robust: HR10=([0-9.]+).*NDCG10=([0-9.]+)',
                line
            )
            if robust_match and current_epoch:
                epoch_info[current_epoch].update({
                    'robust_hr10': float(robust_match.group(1)),
                    'robust_ndcg10': float(robust_match.group(2))
                })

            # 解析保存文件
            if 'save:=================================' in line:
                if i+1 < len(lines):
                    file_match = re.search(
                        r'pretrained/(.+\.pth)',
                        lines[i+1].strip()
                    )
                    if file_match:
                        # 获取当前保存的 epoch
                        save_epoch = current_epoch
                        if save_epoch is not None:
                            save_files[save_epoch] = file_match.group(1)
                            last_save_epoch = save_epoch  # 更新最后保存的 epoch

    # 如果没有保存文件，返回错误信息
    if last_save_epoch is None:
        return "未找到保存文件"

    # 获取最后一个保存的 epoch 的性能指标
    last_save_performance = epoch_info.get(last_save_epoch, None)
    if last_save_performance is None:
        return "未找到最后一个保存 epoch 的性能指标"

    # 获取对应数据
    save_file = save_files.get(last_save_epoch, "未找到保存文件")
    seed_entry = seed_data[last_save_epoch-1] if last_save_epoch <= len(seed_data) else None
    if seed_entry:
        seed_str = '[' + ','.join(map(str, seed_entry[0])) + ']'
        actual_awp = seed_entry[1]
    else:
        seed_str = "N/A"
        actual_awp = "N/A"

    result = [
        f"数据集: {namespace_params['dataset']}",
        f"最后一个保存的 Epoch: {last_save_epoch}",
        f"普通指标 - HR@10: {last_save_performance['normal_hr10']:.4f}, NDCG@10: {last_save_performance['ndcg10']:.4f}",
        f"鲁棒指标 - HR@10: {last_save_performance['robust_hr10']:.4f}, NDCG@10: {last_save_performance['robust_ndcg10']:.4f}",
        f"模型文件: {save_file}",
        f"超参数 - lr_max: {namespace_params['lr_max']}, awp_gamma: {namespace_params['awp_gamma']}",
        f"第{last_save_epoch}行种子数据:",
        f"  • Seed: {seed_str}",
        f"  • Actual AWP Gamma: {actual_awp}",
        "="*50
    ]

    return '\n'.join(result)

def batch_process(file_list, output_file='results.txt'):
    with open(output_file, 'w', encoding='utf-8') as f:  # 指定编码为 utf-8
        for log_path in file_list:
            try:
                result = parse_log(log_path)
                if isinstance(result, str):
                    result = [result]  # 将字符串包装成列表
                f.write(f"分析文件: {log_path}\n")
                f.write('\n'.join(result) + '\n\n')
            except Exception as e:
                f.write(f"处理 {log_path} 时出错: {str(e)}\n\n")

# 使用示例
files_to_process = [
    'output_auto_log_1.txt',
    'output_auto_log_2.txt',
    'output_auto_log_3.txt',
    'output_auto_log_4.txt',
    'output_auto_log_5.txt',
    'output_auto_log_6.txt',
    'output_auto_log_7.txt',
    'output_auto_log_8.txt',
    'output_auto_log_9.txt',
    'output_auto_log_10.txt',
    'output_auto_log_11.txt',
    'output_auto_log_12.txt',
    'output_auto_log_13.txt',
    'output_auto_log_14.txt',
]
batch_process(files_to_process)