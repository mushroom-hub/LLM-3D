import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYOPENGL_PLATFORM"] = "egl"

from src.respace import ReSpace
from pathlib import Path
import json

# 完整场景
respace = ReSpace(do_rag=True,do_cot=True)
output_path_gen = Path("/mnt/e/cby/output1119/2")
output_path_gen.mkdir(parents=True, exist_ok=True)

print("完整场景")
new_scene, is_success = respace.handle_prompt("create a modern living room with exactly 12 key objects.",pth_viz_output=output_path_gen)
if is_success:
    print("New Scene generated successfully.")
    print(f"New Scene roomtype: {new_scene.get('room_type', 'N/A')}")
    print(f"Numver of objects: {len(new_scene.get('objects', []))}")
    new_scene_with_assets = respace.sampling_engine.sample_all_assets(new_scene, is_greedy_sampling=True)
    respace.render_scene_frame(new_scene, filename="frame_full", pth_viz_output=output_path_gen)
    respace.render_scene_360video(new_scene, filename="video-360_full", pth_viz_output=str(output_path_gen))
else:
    print("Failed to generate new scene.")