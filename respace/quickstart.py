import os
os.environ["PYOPENGL_PLATFORM"] = "egl"

from src.respace import ReSpace
from pathlib import Path
import json


# respace = ReSpace()
# # 上下文验证
# try:
#     # with open('D:\\3D-Dataset\\dataset-ssr3dfront\\scenes\\0b0aa9c5-fe43-4926-bfaa-34764214c250-3deb82ae-5906-4d07-9ace-894079cf9104.json','r') as f:
#     with open(
#             '/mnt/d/3D-Dataset/dataset-ssr3dfront/scenes/0b0aa9c5-fe43-4926-bfaa-34764214c250-3deb82ae-5906-4d07-9ace-894079cf9104.json',
#             'r') as f:
#         sample_scene_json_str = f.read()
#     scene = json.loads(sample_scene_json_str)
#     print('json文件加载成功...')
#     n_obj = len(scene.get('objects',[]))
#     print(f'场景内物体数量:{n_obj}')
# except FileNotFoundError:
#     print('Error --- 未找到json文件')
# except json.JSONDecodeError:
#     print('Error --- json格式出错')
#
# # 添加物体
#
# scene_with_assets = respace.resample_all_assets(scene, is_greedy_sampling=True)
# scene_4,_ = respace.handle_prompt('add a small sculpture on the coffee table',scene_with_assets)
# scene_4,_ = respace.handle_prompt('add a large, rustic dining table and four chairs next to the L-shaped floor.',scene_with_assets)
#
#
# respace.render_scene_frame(scene_4,filename='test2_1016_frame',pth_viz_output=Path('./eval/viz/misc/test2_1016'))
# respace.render_scene_360video(scene_4,filename='test2_1016_360view',pth_viz_output=Path('./eval/viz/misc/test2_1016'))

# 完整场景
respace = ReSpace()
print("完整场景")
new_scene, is_success = respace.handle_prompt("create a modern living room with exactly 6 key objects. Place a single, small coffee table exactly 0.5 meters from the sofa.")
if is_success:
    print("New Scene generated successfully.")
    print(f"New Scene roomtype: {new_scene.get('room_type', 'N/A')}")
    print(f"Numver of objects: {len(new_scene.get('objects', []))}")
    # 示例：用户自定义想要的物体偏好（prototypes）
    user_prototypes = [
        "warm wooden nightstand with rounded edges and single drawer",
        "brass floor lamp with arc arm and white fabric shade"
    ]
    # strength 控制偏好影响力（0.0 = 无影响，1.0 = 完全以偏好为准）
    respace.sampling_engine.set_user_preferences(user_prototypes, strength=0.8)

    # 将描述映射到具体资产
    new_scene_with_assets = respace.sampling_engine.sample_all_assets(new_scene, is_greedy_sampling=True)

    # 如果希望恢复默认检索行为，可以清除用户偏好
    # respace.sampling_engine.clear_user_preferences()
    output_path_gen = Path("/mnt/e/Respace_output/test5_1016")
    respace.render_scene_frame(new_scene, filename="frame_full", pth_viz_output=output_path_gen)
    respace.render_scene_360video(new_scene, filename="video-360_full", pth_viz_output=str(output_path_gen))
else:
    print("Failed to generate new scene.")

print(new_scene_with_assets["objects"][0].keys())
print(new_scene_with_assets["objects"][0]["sampled_asset_jid"])
print(new_scene_with_assets["objects"][0]["sampled_asset_desc"])