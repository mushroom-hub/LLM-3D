import os
import json
import numpy as np
from pathlib import Path
import sys
import traceback
from datetime import datetime
from dotenv import load_dotenv

try:
    from src.eval import eval_scene
except ImportError:
    current_dir = Path(__file__).resolve().parent
    sys.path.append(str(current_dir / "src"))
    from eval import eval_scene
except Exception as e:
    print(f"FATAL ERROR: 无法导入 eval 模块。请确保 eval.py 在 src/ 目录下。错误: {e}")
    sys.exit(1)

# ======================= !!!!配置参数 =======================

SCENE_FILE_PATH = "/mnt/e/Respace_output/Multi_Scene_Generation_Test/living room_20_objects/4_living room_21obj.json"

OUTPUT_DIRECTORY = "/mnt/e/Respace_output/Eval_Scene_Results"
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

load_dotenv(".env")
# 几何评估参数
VOXEL_SIZE = 0.05
TOTAL_LOSS_THRESHOLD = 0.1


# ======================= 主测试函数 =======================

def test_single_scene(file_path):
    print(f"--- 开始评估场景: {file_path} ---")
    file_path = Path(file_path)

    if not file_path.exists():
        print(f"错误: 文件未找到: {file_path}")
        return

    full_metrics = {}

    try:
        # 1. 读取场景 JSON 文件
        with open(file_path, 'r') as f:
            scene_data = json.load(f)

        if not scene_data.get("objects"):
            print("场景中不包含物体，跳过评估。")
            return

        # 2. 调用 eval_scene 计算指标 (用于 Full Scene)
        metrics = eval_scene(
            scene_data,
            is_debug=False,
            voxel_size=VOXEL_SIZE,
            total_loss_threshold=TOTAL_LOSS_THRESHOLD,
            do_pms_full_scene=True
        )

        # 3. 整合所有指标和元数据
        full_metrics = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "file_path": str(file_path),
            "room_type": scene_data.get('room_type', 'N/A'),
            "object_count": len(scene_data['objects']),

            "GEOMETRY_VBL": {
                "Total_OOB_Loss_m3": metrics.get('total_oob_loss', 0.0),
                "Total_MBL_Loss_m3": metrics.get('total_mbl_loss', 0.0),
                "Total_PBL_Loss_m3": metrics.get('total_pbl_loss', 0.0),
                "PBL_Validity": metrics.get('is_valid_scene_pbl', False),
            },

            "SEMANTICS": {
                "PMS_Sampled_Score": metrics.get('txt_pms_sampled_score', 0.0),
                "PMS_Raw_Score": metrics.get('txt_pms_score', 0.0),  # 通常在 eval.py 中计算
            },

            "DIAGNOSTICS": {
                "Voxel_Size": VOXEL_SIZE,
                "Loss_Threshold": TOTAL_LOSS_THRESHOLD,
            },

            # 原始场景数据 (可选，用于完整记录)
            # "scene_data_preview": scene_data
        }

        # 4. 输出关键指标到终端
        print("\n================== 评估结果 ==================")
        print(f"房间类型: {full_metrics['room_type']}")
        print(f"物体数量: {full_metrics['object_count']}")
        print("------------------------------------------")
        print(f"总 OOB 损失 (越界体积): {full_metrics['GEOMETRY_VBL']['Total_OOB_Loss_m3']:.6f}")
        print(f"总 MBL 损失 (重叠体积): {full_metrics['GEOMETRY_VBL']['Total_MBL_Loss_m3']:.6f}")
        print(f"PBL 是否有效: {full_metrics['GEOMETRY_VBL']['PBL_Validity']}")
        print(f"PMS (Prompt 匹配度): {full_metrics['SEMANTICS']['PMS_Sampled_Score']:.4f}")
        print("============================================\n")

        # 5. 保存结果到 JSON 文件
        output_filename = f"{Path(SCENE_FILE_PATH).stem}_metrics.json"
        output_path = Path(OUTPUT_DIRECTORY) / output_filename

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(full_metrics, f, indent=4)

        print(f"详细评估数据已保存到: {output_path}")

        return full_metrics

    except json.JSONDecodeError:
        print("错误: JSON 文件格式错误，无法解析。")
    except Exception as e:
        print(f"评估过程中发生未知错误: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    # 打印环境诊断信息 (保持在 if __name__ 中)
    asset_root = os.getenv("PTH_3DFUTURE_ASSETS")
    print(f"DIAGNOSIS: PTH_3DFUTURE_ASSETS value: {asset_root}")
    print(f"DIAGNOSIS: Path exists? {os.path.exists(asset_root)}")

    test_single_scene(SCENE_FILE_PATH)