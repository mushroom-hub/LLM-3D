import os
import sys
import json
import datetime
import re
from pathlib import Path
from flask import Flask, request, jsonify
import torch
import gc

# 清理GPU内存
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    
# 设置环境变量
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["ZHIPU_API_KEY"] = "1479e00599eb47fdb1f195522e0ee0c4.JhmukvpF06bgxQ7M"

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

print("=== 调试信息 ===")
print(f"当前目录: {current_dir}")
print(f"Python路径: {sys.path}")

try:
    from src.respace import ReSpace
    print(" ReSpace导入成功")
except ImportError as e:
    print(f" ReSpace导入失败: {e}")
    try:
        import src.respace as respace_module
        ReSpace = respace_module.ReSpace
        print(" ReSpace导入成功（备用方式）")
    except ImportError as e2:
        print(f" 备用导入也失败: {e2}")
        ReSpace = None

try:
    from assets_retrieval_test import AssetRetrievalModule
    print("AssetRetrievalModule导入成功")
except ImportError as e:
    print(f" AssetRetrievalModule导入失败: {e}")
    AssetRetrievalModule = None

app = Flask(__name__)

class CommandProcessor:
    """命令处理器 - 使用实际的handle_prompt函数"""
    def __init__(self):
        # 这里会使用实际的respace实例
        self.respace = None
        self.sampling_engine = None
    
    def set_respace_instance(self, respace_instance, sampling_engine_instance):
        """设置respace实例"""
        self.respace = respace_instance
        self.sampling_engine = sampling_engine_instance
    
    def process_commands(self, commands, room_type="bedroom"):
        """使用handle_prompt处理命令并生成场景数据"""
        if not self.respace:
            raise ValueError("Respace实例未设置")
        
        print(f"使用handle_prompt处理命令，房间类型: {room_type}")
        print(f"命令数量: {len(commands)}")
        print(f"命令内容: {commands}")
        
        # 构建命令格式的prompt
        command_prompt = {
            "commands": commands
        }
        
        # 调用handle_prompt处理命令
        # current_scene=None 会触发房间边框生成
        # room_type指定房间类型
        try:
            final_scene, is_success = self.respace.handle_prompt(
                prompt=command_prompt,
                current_scene=None,  # 传入None会生成房间边框
                room_type=room_type,
                do_rendering_with_object_count=False,
                pth_viz_output=None
            )
            

            if is_success:
                print(f"handle_prompt处理成功，生成 {len(final_scene.get('objects', []))} 个物体")
                import json
                scene_json = json.dumps(final_scene, indent=2, ensure_ascii=False)
                print( scene_json)
                return final_scene
            else:
                print("handle_prompt处理失败")
                # 返回空场景
                return self._create_empty_scene(room_type)
                
        except Exception as e:
            print(f"handle_prompt处理出错: {e}")
            import traceback
            traceback.print_exc()
            return self._create_empty_scene(room_type)
    
    def _create_empty_scene(self, room_type):
        """创建空场景（备用）"""
        return {
            "bounds_top": [
                [-1.35, 2.6, 1.45], [0.15, 2.6, 1.45],
                [0.15, 2.6, 2.15], [1.35, 2.6, 2.15],
                [1.35, 2.6, -2.15], [-1.35, 2.6, -2.15]
            ],
            "bounds_bottom": [
                [-1.35, 0.0, 1.45], [0.15, 0.0, 1.45],
                [0.15, 0.0, 2.15], [1.35, 0.0, 2.15],
                [1.35, 0.0, -2.15], [-1.35, 0.0, -2.15]
            ],
            "room_type": room_type,
            "objects": []  # 空对象列表
        }

# 初始化命令处理器
command_processor = CommandProcessor()

# 检查组件是否成功初始化
if ReSpace is None or AssetRetrievalModule is None:
    print(" 关键组件初始化失败，服务无法启动")
    
    @app.route('/generate_3d_with_commands', methods=['POST'])
    def generate_3d_with_commands_fallback():
        return jsonify({
            'status': 'error', 
            'message': 'Respace组件初始化失败，请检查控制台输出'
        })
    
    @app.route('/generate_3d_with_model_ids', methods=['POST'])
    def generate_3d_with_model_ids_fallback():
        return jsonify({
            'status': 'error', 
            'message': 'Respace组件初始化失败，请检查控制台输出'
        })
    
else:
    # 正常初始化respace组件
    try:
        print("正在初始化3D资产检索系统...")
        sampling_engine = AssetRetrievalModule(
            temp=0.2, top_p=0.95, top_k=20, 
            use_chinese_embeddings=True,
            rand_seed=1234, do_print=True
        )
        respace = ReSpace(n_bon_assets=1)
        
        # 设置命令处理器的respace实例
        command_processor.set_respace_instance(respace, sampling_engine)
        
        print(" Respace组件初始化成功")
        
        # 原逻辑：处理没有model_id的commands命令
        @app.route('/generate_3d_with_commands', methods=['POST'])
        def generate_3d_with_commands():
            """使用命令模式生成3D场景"""
            try:
                request_data = request.json
                if not request_data:
                    return jsonify({'status': 'error', 'message': '未提供请求数据'})
                
                commands = request_data.get('commands', [])
                room_type = request_data.get('room_type', 'bedroom')
                
                print(f"收到命令生成请求，房间类型: {room_type}")
                print(f"命令数量: {len(commands)}")
                print(f"命令内容: {commands}")
                
                # 步骤1: 使用handle_prompt生成场景布局（包含位置、尺寸、旋转）
                print("=== 步骤1: 使用handle_prompt生成场景布局 ===")
                layout_scene = command_processor.process_commands(commands, room_type)
                
                print(f"布局场景包含 {len(layout_scene.get('objects', []))} 个物体")
                for i, obj in enumerate(layout_scene.get('objects', [])):
                    print(f"  物体{i+1}: {obj.get('desc')}")
                    print(f"    位置: {obj.get('pos')}")
                    print(f"    尺寸: {obj.get('size')}")
                    print(f"    旋转: {obj.get('rot')}")
                
                # 步骤2: 使用AssetRetrievalModule进行资产检索和替换
                print("=== 步骤2: 进行资产检索 ===")
                sampled_scene = sampling_engine.sample_all_assets(layout_scene, is_greedy_sampling=True)
                
                print(f"资产检索完成，最终场景包含 {len(sampled_scene.get('objects', []))} 个物体")
                for i, obj in enumerate(sampled_scene.get('objects', [])):
                    print(f"  物体{i+1}: {obj.get('desc')} -> {obj.get('sampled_asset_desc')}")
                    print(f"    位置: {obj.get('pos')}")
                    print(f"    尺寸: {obj.get('size')} -> {obj.get('sampled_asset_size')}")
                    print(f"    旋转: {obj.get('rot')}")
                
                # 步骤3: 创建输出目录并渲染
                print("=== 步骤3: 渲染场景 ===")
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = Path(f"./api_output_{timestamp}")
                output_dir.mkdir(exist_ok=True)
                
                # 渲染场景
                respace.render_scene_frame(sampled_scene, filename="frame", pth_viz_output=output_dir)
                respace.render_scene_360video(sampled_scene, filename="video", pth_viz_output=str(output_dir))
                
                # 保存场景数据
                scene_json_path = output_dir / "sampled_scene.json"
                with open(scene_json_path, "w", encoding='utf-8') as fp:
                    json.dump(sampled_scene, fp, indent=4, ensure_ascii=False)
                
                # 保存布局数据（用于调试）
                layout_json_path = output_dir / "layout_scene.json"
                with open(layout_json_path, "w", encoding='utf-8') as fp:
                    json.dump(layout_scene, fp, indent=4, ensure_ascii=False)
                
                print(" 3D场景生成完成")
                
                return jsonify({
                    'status': 'success',
                    'message': '3D场景生成完成',
                    'layout_method': 'handle_prompt',
                    'processed_commands': len(commands),
                    'layout_objects_count': len(layout_scene.get('objects', [])),
                    'final_objects_count': len(sampled_scene.get('objects', [])),
                    'output_dir': str(output_dir),
                    'frame_path': str(output_dir / "frame.png"),
                    'video_path': str(output_dir / "video_360.mp4"),
                    'timestamp': timestamp
                })
                
            except Exception as e:
                print(f" 生成3D场景时出错: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({
                    'status': 'error', 
                    'message': f'生成3D场景时出错: {str(e)}'
                }), 500
        # ================================
        # 1. 新增：处理带model_id的命令API
        # ================================

        #@app.route('/generate_3d_with_model_ids', methods=['POST'])
        # def generate_3d_with_model_ids():
            # """新的API端点：使用带model_id的命令生成3D场景，跳过资产检索"""
            # try:
            #     request_data = request.json
            #     if not request_data:
            #         return jsonify({'status': 'error', 'message': '未提供请求数据'})
                
            #     commands = request_data.get('commands', [])
            #     room_type = request_data.get('room_type', 'bedroom')
                
            #     print(f"收到带model_id的命令生成请求，房间类型: {room_type}")
            #     print(f"命令数量: {len(commands)}")
            #     print(f"命令内容: {commands}")
                
            #     # 步骤1: 提取model_id映射
            #     print("=== 步骤1: 提取model_id映射 ===")
            #     model_id_map = {}
            #     plain_commands = []
                
            #     for cmd in commands:
            #         # 提取物体名称
            #         obj_match = re.search(r'<add>(.*?)</add>', cmd)
            #         if obj_match:
            #             obj_name = obj_match.group(1)
            #             plain_commands.append(f"<add>{obj_name}</add>")
                        
            #             # 提取model_id
            #             model_id_match = re.search(r'<model_id>(.*?)</model_id>', cmd)
            #             if model_id_match:
            #                 model_id = model_id_match.group(1)
            #                 model_id_map[obj_name.lower()] = model_id
            #                 print(f"  物体 '{obj_name}' -> model_id: {model_id}")
            #             else:
            #                 print(f"  警告: 物体 '{obj_name}' 没有model_id")
                
            #     # 步骤2: 跳过资产检索
            #     print("=== 步骤2: 设置跳过资产检索 ===")
            #     # 方法：设置n_bon_assets=0，让ReSpace跳过资产检索
            #     original_n_bon_assets = respace.n_bon_assets
            #     respace.n_bon_assets = 0
            #     print(f"  设置 n_bon_assets: {original_n_bon_assets} -> 0")
                
            #     # 步骤3: 生成场景布局
            #     print("=== 步骤3: 生成场景布局 ===")
                
            #     # 构建命令prompt（不包含model_id）
            #     command_prompt = {
            #         "commands": plain_commands
            #     }
                
            #     # 调用handle_prompt生成布局（会跳过资产检索）
            #     final_scene, is_success = respace.handle_prompt(
            #         prompt=command_prompt,
            #         current_scene=None,
            #         room_type=room_type,
            #         do_rendering_with_object_count=False,
            #         pth_viz_output=None
            #     )
                
            #     # 恢复原始n_bon_assets值
            #     respace.n_bon_assets = original_n_bon_assets
                
            #     if not is_success:
            #         return jsonify({
            #             'status': 'error',
            #             'message': '场景布局生成失败'
            #         })
                
            #     # 步骤4: 添加model_id到场景中的物体
            #     print("=== 步骤4: 添加model_id到场景物体 ===")
                
            #     matched_count = 0
            #     for obj in final_scene.get('objects', []):
            #         obj_desc = obj.get('desc', '').lower()
                    
            #         # 查找匹配的model_id
            #         matched_model_id = None
            #         for key in model_id_map:
            #             # 匹配逻辑: 关键字匹配（简单实现）
            #             if key.lower() in obj_desc or obj_desc in key.lower():
            #                 matched_model_id = model_id_map[key]
            #                 break
                    
            #         if matched_model_id:
            #             obj['model_id'] = matched_model_id
            #             # 设置必要的资产字段以便渲染
            #             obj['sampled_asset_jid'] = matched_model_id
            #             if 'sampled_asset_desc' not in obj:
            #                 obj['sampled_asset_desc'] = obj.get('desc', 'unknown')
            #             if 'sampled_asset_size' not in obj:
            #                 obj['sampled_asset_size'] = obj.get('size', [1, 1, 1])
                        
            #             matched_count += 1
            #             print(f"  物体 '{obj_desc}' -> model_id: {matched_model_id}")
                
            #     print(f"成功为 {matched_count}/{len(model_id_map)} 个物体添加model_id")
                
            #     # 步骤5: 渲染场景
            #     print("=== 步骤5: 渲染场景（带model_id）===")
            #     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            #     output_dir = Path(f"./api_output_modelid_{timestamp}")
            #     output_dir.mkdir(exist_ok=True)
                
            #     # 渲染场景
            #     respace.render_scene_frame(final_scene, filename="frame", pth_viz_output=output_dir)
            #     respace.render_scene_360video(final_scene, filename="video", pth_viz_output=str(output_dir))
                
            #     # 保存场景数据
            #     scene_json_path = output_dir / "scene_with_model_ids.json"
            #     with open(scene_json_path, "w", encoding='utf-8') as fp:
            #         json.dump(final_scene, fp, indent=4, ensure_ascii=False)
                
            #     # 保存model_id映射（用于调试）
            #     mapping_path = output_dir / "model_id_mapping.json"
            #     with open(mapping_path, "w", encoding='utf-8') as fp:
            #         json.dump({
            #             "original_commands": commands,
            #             "model_id_map": model_id_map,
            #             "object_matches": [
            #                 {
            #                     "desc": obj.get('desc'),
            #                     "model_id": obj.get('model_id'),
            #                     "position": obj.get('pos'),
            #                     "rotation": obj.get('rot'),
            #                     "size": obj.get('size'),
            #                     "asset_jid": obj.get('sampled_asset_jid')
            #                 }
            #                 for obj in final_scene.get('objects', [])
            #                 if 'model_id' in obj
            #             ]
            #         }, fp, indent=4, ensure_ascii=False)
                
            #     print(" 3D场景生成完成（带model_id，跳过资产检索）")
                
            #     return jsonify({
            #         'status': 'success',
            #         'message': '3D场景生成完成（带model_id，跳过资产检索）',
            #         'processed_commands': len(commands),
            #         'matched_model_ids': matched_count,
            #         'total_objects': len(final_scene.get('objects', [])),
            #         'output_dir': str(output_dir),
            #         'frame_path': str(output_dir / "frame.png"),
            #         'video_path': str(output_dir / "video_360.mp4"),
            #         'timestamp': timestamp,
            #         'scene_data': {
            #             'room_type': final_scene.get('room_type'),
            #             'object_count': len(final_scene.get('objects', [])),
            #             'objects_with_model_ids': [
            #                 {
            #                     'desc': obj.get('desc'),
            #                     'model_id': obj.get('model_id'),
            #                     'position': obj.get('pos'),
            #                     'rotation': obj.get('rot'),
            #                     'size': obj.get('size'),
            #                     'asset_size': obj.get('sampled_asset_size', obj.get('size'))
            #                 }
            #                 for obj in final_scene.get('objects', [])
            #                 if 'model_id' in obj
            #             ]
            #         }
            #     })
                
            # except Exception as e:
            #     print(f" 生成3D场景（带model_id）时出错: {e}")
            #     import traceback
            #     traceback.print_exc()
            #     return jsonify({
            #         'status': 'error', 
            #         'message': f'生成3D场景时出错: {str(e)}'
            #     }), 500
        
        @app.route('/generate_3d_with_model_ids', methods=['POST'])
        def generate_3d_with_model_ids():
            """使用带model_id的命令生成3D场景，直接替换资产ID"""
            try:
                request_data = request.json
                if not request_data:
                    return jsonify({'status': 'error', 'message': '未提供请求数据'})
                
                commands = request_data.get('commands', [])
                room_type = request_data.get('room_type', 'bedroom')
                
                print(f"收到带model_id的命令生成请求，房间类型: {room_type}")
                print(f"命令数量: {len(commands)}")
                print(f"命令内容: {commands}")
                
                # 步骤1: 提取model_id映射
                print("=== 步骤1: 提取model_id映射 ===")
                model_id_map = {}
                plain_commands = []
                
                # 为每个物体创建映射
                object_mappings = []  # 存储每个物体的映射信息
                
                for cmd in commands:
                    # 提取物体名称
                    obj_match = re.search(r'<add>(.*?)</add>', cmd)
                    if obj_match:
                        obj_name = obj_match.group(1)
                        plain_commands.append(f"<add>{obj_name}</add>")
                        
                        # 提取model_id
                        model_id_match = re.search(r'<model_id>(.*?)</model_id>', cmd)
                        if model_id_match:
                            model_id = model_id_match.group(1)
                            # 存储映射：物体顺序 -> model_id
                            object_mappings.append({
                                'index': len(plain_commands) - 1,  # 物体在列表中的位置
                                'name': obj_name,
                                'model_id': model_id
                            })
                            model_id_map[obj_name.lower()] = model_id
                            print(f"  物体 '{obj_name}' (位置{len(plain_commands)-1}) -> model_id: {model_id}")
                        else:
                            print(f"  警告: 物体 '{obj_name}' 没有model_id，将使用默认资产")
                            object_mappings.append({
                                'index': len(plain_commands) - 1,
                                'name': obj_name,
                                'model_id': None
                            })
                
                # 步骤2: 使用handle_prompt生成完整场景
                print("=== 步骤2: 使用handle_prompt生成完整场景 ===")
                command_prompt = {"commands": plain_commands}
                
                final_scene, is_success = respace.handle_prompt(
                    prompt=command_prompt,
                    current_scene=None,
                    room_type=room_type,
                    do_rendering_with_object_count=False,
                    pth_viz_output=None
                )
                
                if not is_success:
                    return jsonify({
                        'status': 'error',
                        'message': '场景布局生成失败'
                    })
                
                print(f"handle_prompt生成 {len(final_scene.get('objects', []))} 个物体")
                
                # 步骤3: 替换资产ID（核心步骤）
                print("=== 步骤3: 用指定model_id替换资产ID ===")
                
                replaced_count = 0
                scene_objects = final_scene.get('objects', [])
                
                # 方法1：按顺序匹配（最简单可靠）
                for i, mapping in enumerate(object_mappings):
                    if i < len(scene_objects) and mapping['model_id']:
                        obj = scene_objects[i]
                        original_jid = obj.get('sampled_asset_jid', '无')
                        
                        # 替换资产ID
                        obj['sampled_asset_jid'] = mapping['model_id']
                        obj['model_id'] = mapping['model_id']  # 添加model_id字段便于追踪
                        
                        # 更新资产描述和尺寸（如果需要）
                        try:
                            asset = sampling_engine.all_assets_metadata.get(mapping['model_id'])
                            if asset:
                                obj['sampled_asset_desc'] = asset.get("summary", mapping['name'])
                                obj['sampled_asset_size'] = asset.get("size", obj.get('size', [1, 1, 1]))
                            else:
                                # 检查缩放资产
                                scaled_asset = sampling_engine.all_assets_metadata_scaled.get(mapping['model_id'])
                                if scaled_asset:
                                    obj['sampled_asset_desc'] = f"缩放资产: {mapping['name']}"
                                    obj['sampled_asset_size'] = scaled_asset.get("size", obj.get('size', [1, 1, 1]))
                        except Exception as e:
                            print(f"  警告: 获取资产信息失败: {e}")
                            # 保留原有描述和尺寸
                        
                        replaced_count += 1
                        print(f"  物体{i+1}: '{mapping['name']}'")
                        print(f"    原资产ID: {original_jid}")
                        print(f"    新资产ID: {mapping['model_id']}")
                
                # 方法2：名称匹配（备用，如果顺序不匹配时使用）
                if replaced_count < len(object_mappings):
                    print("  尝试按名称匹配...")
                    for mapping in object_mappings:
                        if mapping['model_id']:
                            # 查找名称匹配的物体
                            for obj in scene_objects:
                                obj_desc = obj.get('desc', '').lower()
                                if mapping['name'].lower() in obj_desc:
                                    if obj.get('model_id') is None:  # 避免重复替换
                                        original_jid = obj.get('sampled_asset_jid', '无')
                                        obj['sampled_asset_jid'] = mapping['model_id']
                                        obj['model_id'] = mapping['model_id']
                                        replaced_count += 1
                                        print(f"    名称匹配: '{obj_desc}' -> {mapping['model_id']}")
                                        break
                
                print(f"成功替换 {replaced_count}/{len([m for m in object_mappings if m['model_id']])} 个物体的资产ID")
                
                # 步骤4: 渲染场景
                print("=== 步骤4: 渲染场景（使用指定model_id）===")
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = Path(f"./api_output_modelid_replaced_{timestamp}")
                output_dir.mkdir(exist_ok=True)
                
                # 渲染场景
                respace.render_scene_frame(final_scene, filename="frame", pth_viz_output=output_dir)
                respace.render_scene_360video(final_scene, filename="video", pth_viz_output=str(output_dir))
                
                # 保存场景数据
                scene_json_path = output_dir / "scene_modelid_replaced.json"
                with open(scene_json_path, "w", encoding='utf-8') as fp:
                    json.dump(final_scene, fp, indent=4, ensure_ascii=False)
                
                # 保存替换记录
                replacement_path = output_dir / "asset_replacement_log.json"
                with open(replacement_path, "w", encoding='utf-8') as fp:
                    json.dump({
                        "original_commands": commands,
                        "object_mappings": object_mappings,
                        "replacement_results": [
                            {
                                "object_index": i,
                                "object_desc": obj.get('desc', '')[:50],
                                "original_asset_jid": "替换前已存在" if i < len(scene_objects) else "N/A",
                                "new_model_id": obj.get('model_id', '未替换'),
                                "position": obj.get('pos'),
                                "size": obj.get('size')
                            }
                            for i, obj in enumerate(scene_objects)
                        ],
                        "summary": {
                            "total_objects": len(scene_objects),
                            "provided_model_ids": len([m for m in object_mappings if m['model_id']]),
                            "successfully_replaced": replaced_count
                        }
                    }, fp, indent=4, ensure_ascii=False)
                
                print(" 3D场景生成完成（使用指定model_id替换资产）")
                
                return jsonify({
                    'status': 'success',
                    'message': '3D场景生成完成（使用指定model_id替换资产）',
                    'processed_commands': len(commands),
                    'provided_model_ids': len([m for m in object_mappings if m['model_id']]),
                    'replaced_assets': replaced_count,
                    'total_objects': len(scene_objects),
                    'output_dir': str(output_dir),
                    'frame_path': str(output_dir / "frame.png"),
                    'video_path': str(output_dir / "video_360.mp4"),
                    'timestamp': timestamp,
                    'scene_data': {
                        'room_type': final_scene.get('room_type'),
                        'object_count': len(scene_objects),
                        'replaced_objects': [
                            {
                                'index': i,
                                'desc': obj.get('desc', '')[:50],
                                'model_id': obj.get('model_id'),
                                'asset_jid': obj.get('sampled_asset_jid'),
                                'position': obj.get('pos'),
                                'size': obj.get('size')
                            }
                            for i, obj in enumerate(scene_objects)
                            if obj.get('model_id') is not None
                        ]
                    }
                })
            
            except Exception as e:
                print(f" 生成3D场景（替换model_id）时出错: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({
                    'status': 'error', 
                    'message': f'生成3D场景时出错: {str(e)}'
                }), 500

        @app.route('/health', methods=['GET'])
        def health():
            return jsonify({'status': 'healthy', 'message': 'Respace API服务运行正常'})
        
    except Exception as e:
        print(f" 组件初始化过程中出错: {e}")
        import traceback
        traceback.print_exc()
        
        @app.route('/generate_3d_with_commands', methods=['POST'])
        def generate_3d_with_commands_error():
            return jsonify({
                'status': 'error', 
                'message': f'组件初始化失败: {str(e)}'
            })
        
        @app.route('/generate_3d', methods=['POST'])
        def generate_3d_error():
            return jsonify({
                'status': 'error', 
                'message': f'组件初始化失败: {str(e)}'
            })
        
        @app.route('/health', methods=['GET'])
        def health_error():
            return jsonify({'status': 'unhealthy', 'message': f'初始化失败: {str(e)}'})


@app.route('/')
def index():
    return jsonify({
        'message': 'Respace API服务（支持命令模式）',
        'endpoints': {
            'GET /health': '健康检查',
            'POST /generate_3d_with_commands': '生成3D场景（（使用带model_id的命令，跳过资产检索）'
        },
        'features': {
            'command_processing': '使用handle_prompt处理命令',
            'auto_layout': '自动生成房间边框和物体布局',
            'asset_retrieval': '基于语义的3D资产检索',
            'rendering': '3D场景渲染和360度视频生成'
        }
    })

if __name__ == '__main__':
    print(" Respace API服务启动中...")
    print(" 可用端点:")
    print("   GET  /health                    - 健康检查")
    print("   POST /generate_3d_with_commands- 生成3D场景（使用带model_id的命令，跳过资产检索）")
    print("   GET  /                          - 服务信息")
    
    app.run(host='0.0.0.0', port=5000, debug=False)