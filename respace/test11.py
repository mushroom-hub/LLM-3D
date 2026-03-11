import os
os.environ["PYOPENGL_PLATFORM"] = "egl"  # 或 "osmesa"

import pyrender
import trimesh
import numpy as np

# 创建一个立方体 mesh
mesh = pyrender.Mesh.from_trimesh(
    trimesh.primitives.Box(extents=[1, 1, 1])
)

scene = pyrender.Scene()
scene.add(mesh)

# ✅ 添加相机（透视相机）
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
camera_pose = np.array([
    [1.0, 0.0, 0.0, 0.0],   # X
    [0.0, 1.0, 0.0, -1.0],  # Y
    [0.0, 0.0, 1.0, 3.0],   # Z (距离模型3个单位)
    [0.0, 0.0, 0.0, 1.0]
])
scene.add(camera, pose=camera_pose)

# ✅ 添加光源
light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
scene.add(light, pose=camera_pose)

# ✅ 离屏渲染
renderer = pyrender.OffscreenRenderer(256, 256)
color, depth = renderer.render(scene)

print("✅ Render success, color shape:", color.shape)
