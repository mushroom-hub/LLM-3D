import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path
import json
from src.respace import ReSpace
import time
import numpy as np

# ================== 准备 ==============================
print("Initializing ReSpace module...")
respace = ReSpace()

# 定义输出根路径 (WSL 格式的 E 盘)
OUTPUT_ROOT = Path("/mnt/e/Respace_output/Multi_Scene_Generation_Test")
os.makedirs(OUTPUT_ROOT, exist_ok=True)
print(f"Generation output directory set to: {OUTPUT_ROOT}")

# ===================== 参数 ====================================
TEST_CASES = [
    # --- Living Room Cases ---
    # 少物体 (低上下文压力)
    {"room": "living room", "count": 5, "style": "industrial"},
    # 中物体 (中等上下文压力)
    {"room": "living room", "count": 8, "style": "classic"},
    # 中物体 (中等上下文压力)
    {"room": "living room", "count": 10, "style": "classic"},
    # 多物体 (高上下文压力 - 挑战 ReSpace 规划能力)
    {"room": "living room", "count": 15, "style": "eclectic"},
    # 多物体 (高上下文压力 - 挑战 ReSpace 规划能力)
    {"room": "living room", "count": 20, "style": "eclectic"},
]

# 每个案例重复生成的次数（以确保数据统计的可靠性）
N_SAMPLES_PER_CASE = 5


# ================= 生成 ===============================
def run_multi_scene_generation():
    all_generated_scenes = []

    for case in TEST_CASES:
        room = case["room"]
        count = case["count"]
        style = case["style"]

        # 为每个测试案例创建一个子文件夹
        case_dir = OUTPUT_ROOT / f"{room}_{count}_objects"
        os.makedirs(case_dir, exist_ok=True)

        print(f"\n--- Starting Generation for: {room} with {count} objects ---")

        for i in range(N_SAMPLES_PER_CASE):

            # Zero-shot LLM Prompt
            prompt = f"create a {style} {room} with exactly {count} objects."

            start_time = time.time()
            # !!!
            new_scene, is_success, token_list = respace.handle_prompt(prompt)  # 使用适中温度进行探索性生成
            end_time = time.time()

            # --- 结果处理与保存 ---

            if is_success:
                n_generated = len(new_scene.get('objects', []))

                # 强制采样资产 (MBL/OOB 计算和渲染的基础)
                try:
                    new_scene_with_assets = respace.sampling_engine.sample_all_assets(new_scene,
                                                                                      is_greedy_sampling=True)
                except Exception as e:
                    print(f"Warning: Asset sampling failed for scene {i} (N={n_generated}): {e}")
                    continue

                # --- 保存原始 JSON 和渲染 ---

                filename_base = f"{i}_{room}_{n_generated}obj"

                # 保存原始 JSON 文件
                with open(case_dir / f"{filename_base}.json", 'w') as f:
                    json.dump(new_scene_with_assets, f, indent=4)

                # 渲染单帧
                respace.render_scene_frame(new_scene_with_assets,
                                           filename=f"{filename_base}_frame",
                                           pth_viz_output=case_dir)
                respace.render_scene_360video(new_scene_with_assets,
                                              filename=f"{filename_base}_video-360_full",
                                              pth_viz_output=case_dir)
                mean_tokens = np.mean(token_list) if token_list else 0
                # 收集用于评估的元数据
                all_generated_scenes.append({
                    "path": case_dir / f"{filename_base}.json",
                    "target_count": count,
                    "actual_count": n_generated,
                    "room_type": room,
                    "time_s": end_time - start_time
                })

                print(
                    f"Sample {i + 1} SUCCESS. Generated: {n_generated}/{count} objects. Time: {end_time - start_time:.2f}s")
                print(f"Avg Tokens: {mean_tokens:.1f}. Time: {end_time - start_time:.2f}s")

            else:
                print(f"Sample {i + 1} FAILED to generate scene.")

    # 打印最终总结，指导下一步的 eval
    print("\n\n--- GENERATION PHASE COMPLETE ---")
    print("Next Step: Run MBL/OOB evaluation on the generated JSON files using eval.py.")

    return all_generated_scenes


if __name__ == "__main__":
    generated_data = run_multi_scene_generation()
    # 你现在可以对 generated_data 进行进一步的分析和处理