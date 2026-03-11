import os
import sys

def fix_mtl_texture_case(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.mtl'):
                mtl_path = os.path.join(dirpath, filename)
                # 读取 mtl 文件内容
                with open(mtl_path, 'r') as f:
                    lines = f.readlines()
                new_lines = []
                for line in lines:
                    if line.strip().startswith(('map_Kd', 'map_Ka', 'map_Ks')):
                        parts = line.strip().split()
                        if len(parts) == 2:
                            tex_name = parts[1].replace('./', '').replace('.\\', '')
                            # 在当前目录下查找实际存在的文件（不区分大小写）
                            for real_file in os.listdir(dirpath):
                                if real_file.lower() == tex_name.lower():
                                    # 替换为实际文件名
                                    line = f"{parts[0]} ./{real_file}\n"
                                    break
                    new_lines.append(line)
                # 写回 mtl 文件
                with open(mtl_path, 'w') as f:
                    f.writelines(new_lines)
                print(f"Fixed: {mtl_path}")

# 用法：指定你的模型根目录
fix_mtl_texture_case('D:/GradientSpace/respace/3D-FUTURE-model')
print("MTL texture case fix completed.")

print("Python version:", sys.version)
print("Script started")
print("Target dir exists:", os.path.exists('D:/GradientSpace/respace/3D-FUTURE-model'))