import bpy

def render_env(folder,standard_size,tasks,tasks_index,location_list,size_str):
    bpy.ops.mesh.primitive_plane_add()
    bpy.data.objects.remove(bpy.data.objects['Cube'])
    bpy.data.objects['Plane'].scale = [20, 20, 20]

    for i, idx in enumerate(tasks_index):
        file_path = f"D:\\model\\{folder}\\{idx}.glb"
        location = location_list[idx].split('(')[1].rsplit(')')[0].split(',')
        size_list = size_str[idx].split('(')[1].rsplit(')')[0].split(',')
        size = float(size_list[0]) * float(size_list[1]) * float(size_list[2]) / standard_size
        size = size ** (1/3)
        bpy.ops.import_scene.gltf(filepath=file_path)
        obj = bpy.data.objects[f"tripo_node_{tasks[i]['task_id']}"]
        #obj = bpy.data.objects[f"tripo_node_{tasks[i]}"]
        mesh = obj.data
        vertices = mesh.vertices
        min_z = 1000
        min_x = 1000
        min_y = 1000
        max_x = -1000
        max_y = -1000
        for v in vertices:
            if v.co[2] < min_z:
                min_z = v.co[2]
            if v.co[0] < min_x:
                min_x = v.co[0]
            if v.co[1] < min_y:
                min_y = v.co[1]
            if v.co[1] > max_y:
                max_y = v.co[1]
            if v.co[0] > max_x:
                max_x = v.co[0]

        bpy.context.scene.cursor.location = [(max_x + min_x)/2, (max_y + min_y)/2, min_z]
        bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
        obj.location[0] = float(location[0])
        obj.location[1] = float(location[1])
        obj.location[2] = float(location[2])
        obj.scale = (size, size, size)
    export_path = f"D:\\model\\{folder}\\environment.glb"
    bpy.ops.export_scene.gltf(filepath=export_path)

    return None

#render_env("2025-08-14-15-59-33", 1, ['c2ea9100-2935-4b52-b0d6-4802c0bc40b8','22fbd9ba-6b64-499b-a241-c430ae474f0e','92376b82-c824-4a82-8a95-0f25aebd858f','609a78f0-3d91-4751-9da7-ba62d030244f','7aa1b52a-a075-4f08-a8e9-d59f31fc001f','ad11b149-ebbb-46d3-a6f8-13bd85a49dfd','e0860d0e-9282-471d-bbe7-9c94e698e705','915aa332-c40d-4fff-86aa-7d0245bd9889','5e34c44b-0206-43c6-a72c-3b6695470a38','becae13d-740c-4c42-a6ea-fa2adf8b0c87'],[0,1,2,3,4,5,6,7,8,9],['(0,0,0)','(3,1,0)','(-3,-5,0)','(4,-1,0)','(-3,1,0)','(5,6,0)','(-3,6,0)','(-5,-5,0)','(7,-4,0)','(7,7,0)'],['(5,5,5)','(1,1,1)','(1,1,1)','(1,1,1)','(1,1,1)','(1,1,1)','(1,1,1)','(1,1,1)','(1,1,1)','(1,1,1)'])


