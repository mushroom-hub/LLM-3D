import bpy
import os

# 这个可以成功实现单个.obj到.glb的转换

def log(message):
    print(message)
    with open(r"D:\conversion_log.txt", "a") as f:
        f.write(message + "\n")

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    # 清理残留数据（增强版）
    for block in [bpy.data.meshes, bpy.data.materials, bpy.data.images, bpy.data.textures]:
        for item in block:
            block.remove(item)

def convert_obj_to_glb(obj_path, texture_dir, output_path):
    try:
        clear_scene()
        log(f"\n=== 开始转换: {obj_path} ===")

        # 1. 导入OBJ（保留材质）
        bpy.ops.import_scene.obj(
            filepath=obj_path,
            use_split_objects=False,
            use_split_groups=False,
            use_image_search=True  # 自动搜索纹理
        )
        log(f"已导入OBJ: {obj_path}")

        # 2. 强制重新加载纹理（处理mtl可能的问题）
        for img in bpy.data.images:
            if img.filepath:
                log(f"重新加载图像: {img.filepath}")
                img.reload()

        # 3. 确保所有材质使用节点
        for mat in bpy.data.materials:
            if not mat.use_nodes:
                mat.use_nodes = True
                bsdf = mat.node_tree.nodes.get("Principled BSDF")
                if not bsdf:
                    bsdf = mat.node_tree.nodes.new("ShaderNodeBsdfPrincipled")
                    mat.node_tree.links.new(
                        bsdf.outputs["BSDF"],
                        mat.node_tree.nodes["Material Output"].inputs["Surface"]
                    )

        # 4. 手动添加纹理（如果自动加载失败）
        texture_path = os.path.join(texture_dir, "texture.png")
        if not any(img.filepath == texture_path for img in bpy.data.images) and os.path.exists(texture_path):
            img = bpy.data.images.load(texture_path)
            for mat in bpy.data.materials:
                nodes = mat.node_tree.nodes
                tex_node = nodes.new("ShaderNodeTexImage")
                tex_node.image = img
                bsdf = nodes.get("Principled BSDF")
                if bsdf:
                    mat.node_tree.links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
            log(f"手动添加纹理: {texture_path}")

        # 5. 打包所有资源到Blender内部
        bpy.ops.file.pack_all()

        # 6. 导出GLB（兼容旧版Blender）
        bpy.ops.export_scene.gltf(
            filepath=output_path,
            export_format='GLB',
            export_materials='EXPORT',
            export_texcoords=True,
            export_normals=True,
            export_yup=True  # 确保坐标系正确
        )
        log(f"成功导出GLB: {output_path}")

    except Exception as e:
        log(f"错误: {str(e)}")
        raise  # 重新抛出异常以便调试

# === 配置路径 ===
model_dir = r"D:\GradientSpace\respace\3D-FUTURE-model\0a0f0cf2-3a34-4ba2-b24f-34f361c36b3e"
input_obj = os.path.join(model_dir, "normalized_model.obj")
output_glb = os.path.join(model_dir, "output_model.glb")

# 执行转换
convert_obj_to_glb(input_obj, model_dir, output_glb)