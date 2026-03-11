import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import re
import argparse
import sys

# 确定eval.py路径
try:
    # 引入 eval.py 中的核心评估函数 eval_scene
    from src.eval import eval_scene
except ImportError:
    print("FATAL ERROR: Could not import 'eval_scene'. Please ensure your working directory is the ReSpace root.")
    sys.exit(1)

# ================= ！！！！评估参数 (与生成和 eval.py 保持一致) ===========================
VOXEL_SIZE = 0.05
TOTAL_LOSS_THRESHOLD = 0.1

# ================= 主评估函数 ===========================

def run_evaluation_on_generated_scenes(input_root_dir, output_csv_filename="statistical_evaluation.csv"):
    print("\n--- STARTING GEOMETRIC AND SEMANTIC EVALUATION (Reading JSONs directly) ---")

    input_path = Path(input_root_dir)
    print(input_path)
    if not input_path.is_dir():
        print(f"Error: Input directory not found at {input_root_dir}")
        return

    all_results = []
    json_files = []
    # 遍历所有子文件夹 (living room_15_objects)
    # 使用 glob 查找所有 JSON 文件，以便处理嵌套的目录结构
    for case_dir in input_path.iterdir():
        if case_dir.is_dir():
            json_files.extend(list(case_dir.glob('*.json')))

    if not json_files:
        print("Error: No JSON scene files found in the specified directory structure.")
        print(f"Directory checked: {input_root_dir}. Please ensure JSONs exist in subfolders.")
        return

    # 逐个处理 JSON 文件并计算指标
    for scene_path in tqdm(json_files, desc="Processing Scenes"):

        try:
            # 从文件名解析元数据
            # 假设文件名格式: {i}_{room}_{N_actual}obj.json
            match = re.search(r'(\d+)_([a-zA-Z]+ [a-zA-Z]+)_(\d+)obj\.json', scene_path.name)

            # 如果文件名不匹配，从父目录获取 room_type 和 target_N
            if match:
                # actual_N = int(match.group(3)) # 实际生成的数量
                # room_type_name = match.group(2)
                pass  # 使用 JSON 内部的 room_type

            # 读取场景 JSON 文件
            with open(scene_path, 'r') as f:
                scene_data = json.load(f)

            # 从父目录名和 JSON 内部提取元数据
            # 父目录格式: room_count_objects
            dir_name_parts = scene_path.parent.name.split('_')
            target_count = int(dir_name_parts[1]) if len(dir_name_parts) > 1 else len(scene_data.get("objects", []))

            actual_count = len(scene_data.get("objects", []))
            room_type = scene_data.get("room_type", "unknown")

            # 计算指标
            metrics = eval_scene(
                scene_data,
                is_debug=False,
                voxel_size=VOXEL_SIZE,
                total_loss_threshold=TOTAL_LOSS_THRESHOLD,
                do_pms_full_scene=True
            )


            # 汇总数据
            all_results.append({
                "scene_path": str(scene_path),
                "room_type": room_type,
                "target_N": target_count,
                "actual_N": actual_count,
                # "runtime_s": scene_meta["time_s"], # 无法直接获取

                "MBL_loss": metrics.get('total_mbl_loss', 0.0),
                "OOB_loss": metrics.get('total_oob_loss', 0.0),
                "PBL_valid": metrics.get('is_valid_scene_pbl', False),
                "PMS_score": metrics.get('txt_pms_sampled_score', 0.0),
            })

        except Exception as e:
            print(f"Error evaluating scene {scene_path.name}: {e}. Skipping.")

    if not all_results:
        print("No valid scene evaluation results collected.")
        return

    df_results = pd.DataFrame(all_results)

    # 计算分组平均值 (按 room_type 和 target_N 分组)
    grouped_stats = df_results.groupby(['room_type', 'target_N']).agg({
        'MBL_loss': ['mean', 'std'],
        'OOB_loss': ['mean', 'std'],
        'PBL_valid': 'mean',
        'PMS_score': 'mean',
        'actual_N': 'mean'
    }).reset_index()

    # 清理列名
    grouped_stats.columns = [
        'room_type', 'target_N',
        'MBL_mean', 'MBL_std',
        'OOB_mean', 'OOB_std',
        'PBL_valid_mean',
        'PMS_mean',
        'Actual_N_mean'
    ]

    output_csv_path = Path("/mnt/e/Respace_output") / output_csv_filename
    grouped_stats.to_csv(output_csv_path, index=False)

    print("\n--- EVALUATION COMPLETE ---")
    print(f"Statistical results saved to: {output_csv_path.resolve()}")

    return grouped_stats


if __name__ == "__main__":
    # 目标输入目录
    INPUT_DIRECTORY = "/mnt/e/Respace_output/Multi_Scene_Generation_Test"

    # 目标输出文件名
    OUTPUT_FILE_NAME = "final_mbl_oob_stats.csv"

    # 运行评估
    run_evaluation_on_generated_scenes(INPUT_DIRECTORY, OUTPUT_FILE_NAME)