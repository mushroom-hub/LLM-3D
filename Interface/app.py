import os
import requests
import datetime
from flask import Flask, render_template, request, jsonify, redirect
from flask import send_file
import uuid
import shutil
import json
from typing import List, Dict, Any

# from MongoDBaccess import ModelSearcher
from EnvGenerator1 import chat_with_model

is_api = input("Use API mode? (y/n): ").strip().lower() == 'y'

this_path = __file__.rsplit('\\',1)[0]
path = f"E:\\project\\Assets Package\\Model"
os.chdir(path)

current_time = datetime.datetime.now()
new_folder = current_time.strftime("%Y-%m-%d-%H-%M-%S")
os.makedirs(new_folder)

os.chdir(this_path)

class AssetSearcher:
    def __init__(self, 
                 model_info_path: str = r"D:\3D-Dataset\info_collection\model_info.json",
                 future_assets_path: str = r"D:\GradientSpace\respace\data\metadata\model_info_3dfuture_assets_prompts.json"):
        self.model_info_path = model_info_path
        self.future_assets_path = future_assets_path
        self.model_info = self._load_model_info()
        self.future_assets_info = self._load_future_assets_info()
    
    def _load_model_info(self) -> List[Dict]:
        try:
            with open(self.model_info_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"成功加载模型信息，共 {len(data)} 个模型")
                return data
        except Exception as e:
            print(f"加载模型信息失败: {e}")
            return []
    
    def _load_future_assets_info(self) -> Dict:
        try:
            with open(self.future_assets_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"成功加载3D-FUTURE资产信息，共 {len(data)} 个模型")
                return data
        except Exception as e:
            print(f"加载3D-FUTURE资产信息失败: {e}")
            return {}
    
    def _safe_get(self, model: Dict, key: str, default: str = '') -> str:
        value = model.get(key, default)
        return str(value) if value is not None else default
    
    def _search_in_model_info(self, category: str, max_styles: int = 4) -> List[Dict]:
        if not self.model_info:
            return []
        
        print(f"在model_info中搜索类别: {category}")
        
        # 将中文类别映射到英文
        category_mapping = {
            "床": "Bed",
            "床架": "Bed Frame", 
            "桌子": "Desk",
            "书桌": "Desk",
            "椅子": "Chair",
            "沙发": "Sofa",
            "电视柜": "TV Stand",
            "咖啡桌": "Coffee Table",
            "衣柜": "Wardrobe",
            "书架": "Bookshelf",
            "台灯": "Lamp",
            "床头柜": "Nightstand",
            "茶几": "Coffee Table",
            "餐桌": "Dining Table",
            "办公桌": "Desk",
            "沙发床": "Sofa Bed",
            "躺椅": "Chair",
            "吧台椅": "Chair",
            "pillow": "Pillow",
            "blanket": "Blanket",
            "alarm": "Clock",
            "rug": "Rug",
            "mirror": "Mirror"
        }
        
    
        target_category = category_mapping.get(category, category)
        print(f"映射后的目标类别: {target_category}")
        
        matched_models = []
        seen_styles = set()
        
        for model in self.model_info:
            model_category = self._safe_get(model, 'category', '').lower()
            model_super_category = self._safe_get(model, 'super-category', '').lower()
            target_category_lower = target_category.lower()
            
            match_found = (
                model_category == target_category_lower or 
                model_super_category == target_category_lower or
                target_category_lower in model_category or
                target_category_lower in model_super_category or
                category.lower() in model_category or  
                category.lower() in model_super_category
            )
            
            if match_found:
                style = self._safe_get(model, 'style', 'Unknown')
                if style not in seen_styles or len(seen_styles) < max_styles:
                    matched_models.append(model)
                    seen_styles.add(style)
                    print(f"在model_info中找到匹配模型: {model.get('model_id', 'Unknown')}, 风格: {style}")
                
                if len(seen_styles) >= max_styles:
                    break
        
        return matched_models
    
    def _search_in_future_assets(self, category: str, max_models: int = 4) -> List[Dict]:
        if not self.future_assets_info:
            return []
        
        print(f"在future_assets中搜索类别: {category}")
        
        category_mapping = {
            "床": ["bed", "bedframe"],
            "床架": ["bed", "bedframe"],
            "桌子": ["desk", "table"],
            "书桌": ["desk", "writing"],
            "椅子": ["chair", "armchair", "seat"],
            "沙发": ["sofa", "couch"],
            "电视柜": ["tv", "television", "stand"],
            "咖啡桌": ["coffee", "table"],
            "衣柜": ["wardrobe", "closet"],
            "书架": ["bookshelf", "bookcase"],
            "台灯": ["lamp", "light"],
            "床头柜": ["nightstand", "bedside"],
            "茶几": ["coffee", "table", "tea"],
            "餐桌": ["dining", "table"],
            "办公桌": ["desk", "office"],
            "沙发床": ["sofa", "bed", "sofabed"],
            "躺椅": ["chair", "armchair", "recliner"],
            "吧台椅": ["chair", "barstool", "stool"],
            "pillow": ["pillow", "cushion"],
            "blanket": ["blanket", "throw"],
            "alarm": ["clock", "alarm"],
            "rug": ["rug", "carpet", "mat"],
            "mirror": ["mirror"]
        }
    
        search_keywords = category_mapping.get(category, [category.lower()])
        if isinstance(search_keywords, str):
            search_keywords = [search_keywords]
        
        print(f"搜索关键词: {search_keywords}")
        
        matched_models = []
        
        for model_id, prompts in self.future_assets_info.items():
            if len(matched_models) >= max_models:
                break
          
            prompts_text = " ".join(prompts).lower()
            
            match_found = any(keyword in prompts_text for keyword in search_keywords)
            
            if match_found:
                category_from_prompt = prompts[0] if prompts else "Unknown"
                
                model_data = {
                    "model_id": model_id,
                    "style": None,  
                    "category": category_from_prompt,
                    "theme": None,
                    "material": None,
                    "super-category": None,
                    "source": "3d_future_assets"  # 标记来源
                }
                
                matched_models.append(model_data)
                print(f"在future_assets中找到匹配模型: {model_id}, 类别: {category_from_prompt}")
        
        return matched_models
    
    def search_models_by_category(self, category: str, max_styles: int = 4) -> List[Dict]:
        all_matched_models = []
        
        models_from_first = self._search_in_model_info(category, max_styles)
        all_matched_models.extend(models_from_first)
        
        if len(all_matched_models) < max_styles:
            remaining_slots = max_styles - len(all_matched_models)
            models_from_second = self._search_in_future_assets(category, remaining_slots)
            all_matched_models.extend(models_from_second)
        
        print(f"总共为类别 '{category}' 找到 {len(all_matched_models)} 个匹配模型")
        return all_matched_models
    
    def get_model_image_path(self, model_id: str) -> str:
        path1 = fr"D:\3D-Dataset\3D-FUTURE-model\{model_id}\image.jpg"
        if os.path.exists(path1):
            return path1
        
        path2 = fr"D:\3D-Dataset\3D-FUTURE-model\{model_id}\image.jpg"  
        return path2
    
    def check_image_exists(self, model_id: str) -> bool:
        image_path = self.get_model_image_path(model_id)
        exists = os.path.exists(image_path)
        print(f"检查图片路径: {image_path}, 存在: {exists}")
        return exists
    

class EnhancedObjectProcessor:
    def __init__(self):
        self.asset_searcher = AssetSearcher()
        self.selected_models = {}
    
    def process_object_list(self, object_names: List[str]) -> List[Dict]:
        enhanced_objects = []
        
        for obj_name in object_names:
            print(f"\n处理物体: {obj_name}")
            
            matched_models = self.asset_searcher.search_models_by_category(obj_name)
            
            if matched_models:
                selected_model = matched_models[0]
                enhanced_objects.append({
                    'name': obj_name,
                    'models': matched_models,
                    'selected_model': selected_model,
                    'description': f'{obj_name} 家具风格: {selected_model.get("style", "Unknown")}',
                    'has_models': True
                })
                print(f"为 '{obj_name}' 找到 {len(matched_models)} 个模型")
            else:
                enhanced_objects.append({
                    'name': obj_name,
                    'models': [],
                    'selected_model': None,
                    'description': f'{obj_name} (无匹配模型)',
                    'has_models': False
                })
                print(f"为 '{obj_name}' 未找到匹配模型")
        
        print(f"\n总共处理了 {len(enhanced_objects)} 个物体")
        return enhanced_objects
    
    def update_selected_model(self, object_name: str, model_id: str, enhanced_objects: List[Dict]) -> bool:
        """更新用户选择的模型"""
        for obj in enhanced_objects:
            if obj['name'] == object_name:
                for model in obj['models']:
                    if model['model_id'] == model_id:
                        obj['selected_model'] = model
                        self.selected_models[object_name] = model
                        print(f"更新选择: {object_name} -> {model_id}")
                        return True
        return False
    
    def get_selected_model_ids(self) -> List[str]:
        """获取所有选择的模型ID"""
        return [model['model_id'] for model in self.selected_models.values()]

app = Flask(__name__,
            static_folder='page/static',
            template_folder='page/templates')


RESPACE_API_URL = "http://172.17.170.86:5000/generate_3d_with_commands"


app.config.update({
    'DOWNLOAD_FOLDER': 'downloads',
    'ALLOWED_EXTENSIONS': {'glb', 'fbx', 'obj', 'stl'},
    'MAX_CONTENT_LENGTH': 16 * 1024 * 1024 * 1024  # 16GB
})

with app.app_context():
    app.config['stuffs'] = []
    app.config['prompts'] = []
    app.config['time'] = new_folder
    app.config['respace_video_path'] = None
    app.config['room_type'] = 'bedroom'
    app.config['selected_objects'] = []
    app.config['enhanced_objects'] = []
    app.config['object_processor'] = EnhancedObjectProcessor()
    

os.makedirs(app.config['DOWNLOAD_FOLDER'], exist_ok=True)

def determine_room_type(keyword):
    """根据用户输入的关键词确定房间类型"""
    keyword_lower = keyword.lower()
    
    if any(word in keyword_lower for word in ['卧室', '床', 'sleep', 'bedroom']):
        return "bedroom"
    elif any(word in keyword_lower for word in ['客厅', 'living room', 'lounge']):
        return "livingroom"
    else:
        return "other"

def create_command_data(selected_objects):
    """将选中的物体转换为命令格式，包含模型ID"""
    commands = []
    object_processor = app.config['object_processor']
    
    for obj_name in selected_objects:
        model_info = object_processor.selected_models.get(obj_name)
        if model_info:
            commands.append(f"<add>{obj_name}</add><model_id>{model_info['model_id']}</model_id>")
            print(f"添加带模型ID的命令: {obj_name} -> {model_info['model_id']}")
        else:

            commands.append(f"<add>{obj_name}</add>")
            print(f"添加普通命令: {obj_name}")
    
    command_data = {
        "commands": commands,
        "room_type": app.config['room_type'],
        "selected_model_ids": object_processor.get_selected_model_ids()
    }
    
    print(f"生成的命令数据: {command_data}")
    return command_data

def call_respace_api_with_commands(command_data):
    """调用respace API生成3D场景（使用命令模式）"""
    try:
        print(f"调用respace API: {RESPACE_API_URL}")
        print(f"命令数据: {command_data}")
        
        response = requests.post(RESPACE_API_URL, json=command_data, timeout=300)
        print(f"API响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"API返回结果: {result}")
            
            if result.get('status') == 'success':
                original_video_path = result.get('video_path')
                output_dir = result.get('output_dir')
                
                print(f"原始视频路径: {original_video_path}")
                print(f"输出目录: {output_dir}")
                
                respace_base_dir = r"D:\GradientSpace\respace"
                print(f"Respace基目录: {respace_base_dir}")
                

                full_video_path = None
                
                if original_video_path:
                    full_path = os.path.join(respace_base_dir, original_video_path)
                    print(f"尝试路径1: {full_path}")
                    if os.path.exists(full_path):
                        full_video_path = full_path
                        print(f" 找到视频文件(方法1): {full_video_path}")

                if not full_video_path and output_dir:
                    full_output_dir = os.path.join(respace_base_dir, output_dir)
                    video_360_path = os.path.join(full_output_dir, "video_360.mp4")
                    print(f"尝试路径2: {video_360_path}")
                    if os.path.exists(video_360_path):
                        full_video_path = video_360_path
                        print(f" 找到视频文件(方法2): {full_video_path}")
                
                if not full_video_path and output_dir:
                    full_output_dir = os.path.join(respace_base_dir, output_dir)
                    video_path = os.path.join(full_output_dir, "video.mp4")
                    print(f"尝试路径3: {video_path}")
                    if os.path.exists(video_path):
                        full_video_path = video_path
                        print(f" 找到视频文件(方法3): {full_video_path}")
                
                if full_video_path:
                    print(f" 最终确定的视频路径: {full_video_path}")
                    
                    video_filename = f"scene_{uuid.uuid4().hex[:8]}.mp4"
                    static_video_dir = os.path.join(app.static_folder, 'videos')
                    static_video_path = os.path.join(static_video_dir, video_filename)
                    
                    os.makedirs(static_video_dir, exist_ok=True)
                    
                    shutil.copy2(full_video_path, static_video_path)
                    print(f" 视频文件已复制到: {static_video_path}")
                    
                    result['video_path'] = f"/static/videos/{video_filename}"
                    result['video_name'] = video_filename
                    print(f" 可访问的视频URL: {result['video_path']}")
                    
                    return result, None
                else:
                    print("所有方法都无法找到视频文件")
                    return None, "无法找到生成的视频文件"
            else:
                return None, f"API返回错误: {result.get('message', '未知错误')}"
        else:
            return None, f"API调用失败: {response.status_code} - {response.text}"
            
    except Exception as e:
        return None, f"API连接失败: {str(e)}"

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        return render_template('page.html')
    else:
        return render_template('page.html')

@app.route('/generate', methods=['GET', 'POST'])
def generate():
    try:
        keyword = request.form.get('keyword', '').strip()
        if not keyword:
            return render_template('page.html', message="请输入环境描写")
        
        app.config['stuffs'] = []
        app.config['prompts'] = []
        
        app.config['room_type'] = determine_room_type(keyword)
        
        global is_api
        object_names = chat_with_model(keyword, is_api).get_object_list_only()
        
        enhanced_objects = app.config['object_processor'].process_object_list(object_names)
        app.config['enhanced_objects'] = enhanced_objects
        
        available_stuffs = []
        asset_searcher = app.config['object_processor'].asset_searcher  
        
        for obj in enhanced_objects:
            matched_models = asset_searcher.search_models_by_category(obj['name'])
            if matched_models:  
                available_stuffs.append({
                    'name': obj['name'],
                    'prompt': obj['description']
                })
        
        for stuff in available_stuffs:
            app.config['stuffs'].append(stuff['name'])
            app.config['prompts'].append(stuff['prompt'])

        context = {
            'available_stuffs': available_stuffs,  
            'stuffs': app.config['stuffs'],        
            'prompts': app.config['prompts'],      
            'room_type': app.config['room_type']
        }
        return render_template('search.html', **context)
    
    except Exception as e:
        print(f"生成页面出错: {e}")
        return render_template('page.html', message=f"生成过程中出现错误: {str(e)}")

@app.route('/search', methods=['GET', 'POST'])
def search():
    print("/search 路由被调用 ")
    print(f"请求方法: {request.method}")
    print(f"表单数据: {request.form}")
    
    selected_indices = request.form.getlist('var')
    print(f"选择的索引: {selected_indices}")
    
    selected_objects = []
    if selected_indices:
        for i in selected_indices:
            idx = int(i) - 1
            if idx < len(app.config['stuffs']):
                selected_objects.append(app.config['stuffs'][idx])
        
        app.config['selected_objects'] = selected_objects
        print(f"选择的物体: {selected_objects}")
    
    return redirect('/preview')

@app.route('/preview', methods=['GET', 'POST'])
def preview():
    try:
        selected_objects = app.config.get('selected_objects', [])
        enhanced_objects = app.config.get('enhanced_objects', [])
        
        print(f"预览页面 - 选择的物体: {selected_objects}")
        print(f"预览页面 - 增强物体数量: {len(enhanced_objects)}")
        
        selected_enhanced_objects = []
        for obj_name in selected_objects:
            for enhanced_obj in enhanced_objects:
                if enhanced_obj['name'] == obj_name:
                    selected_enhanced_objects.append(enhanced_obj)
                    break
        
        print(f"预览页面 - 选择的增强物体数量: {len(selected_enhanced_objects)}")
        
        if request.method == 'POST':
            print("处理预览页面POST请求")
            for obj in selected_enhanced_objects:
                if obj['has_models']:
                    model_key = f"model_{obj['name']}"
                    selected_model_id = request.form.get(model_key)
                    if selected_model_id:
                        app.config['object_processor'].update_selected_model(
                            obj['name'], selected_model_id, enhanced_objects
                        )
            
            return redirect('/rendering')
        
        context = {
            'selected_objects': selected_enhanced_objects,
            'object_processor': app.config['object_processor']
        }
        return render_template('preview.html', **context)
    
    except Exception as e:
        print(f"预览页面出错: {e}")
        return render_template('error.html', message=f"预览页面错误: {str(e)}")

@app.route('/rendering', methods=['GET'])
def rendering_page():
    """渲染页面"""
    return render_template('rendering.html')

@app.route('/start_render', methods=['POST'])
def start_render():
    """开始渲染场景 - 使用命令模式"""
    try:
        command_data = create_command_data(app.config['selected_objects'])
        
        print("生成的命令数据:")
        print(command_data)
        
        result, error = call_respace_api_with_commands(command_data)
        
        if result:
            video_path = result.get('video_path')
            task_id = result.get('task_id')
            
            if video_path:
                app.config['respace_video_path'] = video_path
                return jsonify({
                    "status": "success", 
                    "message": "3D场景生成成功！",
                    "video_path": video_path
                })
            elif task_id:
                app.config['respace_task_id'] = task_id
                return jsonify({
                    "status": "processing",
                    "message": "场景正在生成中...",
                    "task_id": task_id
                })
            else:
                return jsonify({"status": "error", "message": "API返回成功但未包含视频路径"})
        else:
            return jsonify({"status": "error", "message": f"3D场景生成失败: {error}"})
            
    except Exception as e:
        return jsonify({"status": "error", "message": f"生成3D场景时出错: {str(e)}"})

@app.route('/result')
def result():
    """显示渲染结果"""
    video_path = app.config.get('respace_video_path')
    if video_path:
        return render_template('final_result.html', 
                            video_path=video_path,
                            message="3D场景生成成功！",
                            selected_objects=app.config['selected_objects'],
                            room_type=app.config['room_type'])
    else:
        return render_template('error.html', message="视频尚未生成完成")



@app.route('/static/model_images/<model_id>.jpg')
def serve_model_image(model_id):
    """提供模型图片的静态文件路由"""
    try:
        image_path = app.config['object_processor'].asset_searcher.get_model_image_path(model_id)
        if os.path.exists(image_path):
            return send_file(image_path, mimetype='image/jpeg')
        else:
            return "Image not found", 404
    except Exception as e:
        print(f"提供模型图片失败: {e}")
        return "Error", 500

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', message="服务器内部错误"), 500

@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', message="页面未找到"), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=False)