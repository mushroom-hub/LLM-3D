import bpy
import os
from pathlib import Path
import time
import sys
from contextlib import contextmanager

class SimpleProgressBar:
    def __init__(self, total, desc=""):
        self.total = total
        self.desc = desc
        self.start = time.time()
        
    def update(self, n=1):
        elapsed = time.time() - self.start
        percent = min(100, (n / self.total) * 100)
        print(f"\r{self.desc} [{('#' * int(percent//2)).ljust(50)}] {percent:.1f}% | {n}/{self.total} | 耗时: {elapsed:.1f}s", end="")
        
    def close(self):
        print()

def get_folder_names(directory):
    path = Path(directory)
    folder_names = [f.name for f in path.iterdir() if f.is_dir()]
    return folder_names

def clear_blender_data():
    # Deselect all
    bpy.ops.object.select_all(action='DESELECT')
    # Delete all objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # Clear orphan data with error handling
    for block_type in [bpy.data.meshes, bpy.data.materials, bpy.data.images]:
        for item in list(block_type):  # Create a copy of the list to safely modify it
            try:
                if item.users == 0:
                    block_type.remove(item)
            except Exception as e:
                print(f"Error removing {item}: {str(e)}")

def convert_object(pth_src_file, pth_texture):
    try:
        clear_blender_data()

        # Import OBJ with modern API
        bpy.ops.wm.obj_import(
            filepath=pth_src_file,
            use_split_objects=False,
            use_split_groups=False
        )
        
        # Select all and apply transforms
        bpy.ops.object.select_all(action='SELECT')
        for obj in bpy.context.selected_objects:
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
            
            if obj.type == 'MESH':
                # Mesh cleanup
                bpy.ops.object.mode_set(mode='EDIT')
                bpy.ops.mesh.select_all(action='SELECT')
                bpy.ops.mesh.normals_make_consistent(inside=False)
                bpy.ops.object.mode_set(mode='OBJECT')
                
                # Ensure material exists
                if not obj.data.materials:
                    mat = bpy.data.materials.new(name="DefaultMaterial")
                    obj.data.materials.append(mat)

                # Material processing with error handling
                for mat in obj.data.materials:
                    try:
                        mat.use_nodes = True
                        nodes = mat.node_tree.nodes
                        nodes.clear()
                        
                        # Create nodes
                        node_texture = nodes.new('ShaderNodeTexImage')
                        node_bsdf = nodes.new('ShaderNodeBsdfPrincipled')
                        node_output = nodes.new('ShaderNodeOutputMaterial')
                        
                        # Load texture safely
                        try:
                            if not bpy.data.images.get(os.path.basename(pth_texture)):
                                img = bpy.data.images.load(pth_texture)
                            else:
                                img = bpy.data.images[os.path.basename(pth_texture)]
                            node_texture.image = img
                        except Exception as tex_err:
                            print(f"Texture load error: {str(tex_err)}")
                            continue
                            
                        # Connect nodes
                        mat.node_tree.links.new(
                            node_texture.outputs['Color'],
                            node_bsdf.inputs['Base Color']
                        )
                        mat.node_tree.links.new(
                            node_bsdf.outputs['BSDF'],
                            node_output.inputs['Surface']
                        )
                    except Exception as mat_err:
                        print(f"Material error: {str(mat_err)}")

        # Export GLB instead of OBJ (original code had OBJ export which was probably a typo)
        output_path = os.path.splitext(pth_src_file)[0] + ".glb"
        bpy.ops.export_scene.gltf(
            filepath=output_path,
            export_format='GLB',
            export_materials='EXPORT',
            export_yup=True
        )
        return True

    except Exception as exc:
        print(f"Error processing {pth_src_file}: {str(exc)}")
        return False
    finally:
        # Clean up without resetting entire Blender
        clear_blender_data()

def batch_convert(pth_root, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(pth_root, "converted")
    os.makedirs(output_dir, exist_ok=True)
    
    obj_ids = get_folder_names(pth_root)
    progress = SimpleProgressBar(len(obj_ids), "Converting models")
    
    for i, obj_id in enumerate(obj_ids, 1):
        progress.update(i)
        
        pth_tgt = os.path.join(pth_root, obj_id)
        pth_src_file = os.path.join(pth_tgt, "raw_model.obj")
        pth_texture = os.path.join(pth_tgt, 'texture.png')
        
        if not os.path.exists(pth_src_file):
            print(f"\nOBJ file not found: {pth_src_file}")
            continue
            
        if not os.path.exists(pth_texture):
            print(f"\nTexture file not found: {pth_texture}")
            
        # Convert with retry logic
        success = False
        for attempt in range(2):  # Try twice
            if convert_object(pth_src_file, pth_texture):
                success = True
                break
            time.sleep(1)  # Wait before retry
            
        if not success:
            print(f"\nFailed to convert: {obj_id}")
    
    progress.close()
    print("Batch conversion completed")

# Example usage
if __name__ == "__main__":
    # Windows path example
    pth_root = r"D:\GradientSpace\respace\3D-FUTURE-model"
    
    # Mac/Linux path example
    # pth_root = "/Volumes/apollo11/data/3D-FUTURE-assets/"
    
    batch_convert(pth_root)