import os
import requests
import datetime
from flask import Flask, render_template, request, jsonify
import uuid
import shutil

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

# 初始化Flask
app = Flask(__name__,
            static_folder='page/static',
            template_folder='page/templates')

# respace API配置
RESPACE_API_URL = "http://172.17.170.86:5000/generate_3d_with_commands"
# RESPACE_VIDEO_URL = 172.17.170.86:5000

# 配置
app.config.update({
    'DOWNLOAD_FOLDER': 'downloads',
    'ALLOWED_EXTENSIONS': {'glb', 'fbx', 'obj', 'stl'},
    'MAX_CONTENT_LENGTH': 16 * 1024 * 1024 * 1024  # 16GB
})

with app.app_context():
    app.config['stuffs'] = []
    app.config['prompts'] = []  # 移除location, size, rotation
    app.config['time'] = new_folder
    app.config['respace_video_path'] = None
    app.config['room_type'] = 'bedroom'
    app.config['selected_objects'] = []  # 存储用户选择的物体名称
    
# 确保下载目录存在
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
    """将选中的物体转换为命令格式"""
    commands = []
    for obj_name in selected_objects:
        commands.append(f"<add>{obj_name}</add>")
    
    command_data = {
        "commands": commands,
        "room_type": app.config['room_type']
    }
    
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
                # 处理视频文件路径
                original_video_path = result.get('video_path')
                output_dir = result.get('output_dir')
                
                print(f"原始视频路径: {original_video_path}")
                print(f"输出目录: {output_dir}")
                
                # Respace服务的基目录
                respace_base_dir = r"D:\GradientSpace\respace"
                print(f"Respace基目录: {respace_base_dir}")
                
                # 构建完整的视频路径
                full_video_path = None
                
                # 方法1: 使用原始视频路径构建完整路径
                if original_video_path:
                    full_path = os.path.join(respace_base_dir, original_video_path)
                    print(f"尝试路径1: {full_path}")
                    if os.path.exists(full_path):
                        full_video_path = full_path
                        print(f" 找到视频文件(方法1): {full_video_path}")
                
                # 方法2: 使用输出目录构建video_360.mp4路径
                if not full_video_path and output_dir:
                    full_output_dir = os.path.join(respace_base_dir, output_dir)
                    video_360_path = os.path.join(full_output_dir, "video_360.mp4")
                    print(f"尝试路径2: {video_360_path}")
                    if os.path.exists(video_360_path):
                        full_video_path = video_360_path
                        print(f" 找到视频文件(方法2): {full_video_path}")
                
                # 方法3: 使用输出目录构建video.mp4路径
                if not full_video_path and output_dir:
                    full_output_dir = os.path.join(respace_base_dir, output_dir)
                    video_path = os.path.join(full_output_dir, "video.mp4")
                    print(f"尝试路径3: {video_path}")
                    if os.path.exists(video_path):
                        full_video_path = video_path
                        print(f" 找到视频文件(方法3): {full_video_path}")
                
                if full_video_path:
                    print(f" 最终确定的视频路径: {full_video_path}")
                    
                    # 生成唯一文件名
                    video_filename = f"scene_{uuid.uuid4().hex[:8]}.mp4"
                    static_video_dir = os.path.join(app.static_folder, 'videos')
                    static_video_path = os.path.join(static_video_dir, video_filename)
                    
                    # 确保静态视频目录存在
                    os.makedirs(static_video_dir, exist_ok=True)
                    
                    # 复制视频文件到静态目录
                    shutil.copy2(full_video_path, static_video_path)
                    print(f" 视频文件已复制到: {static_video_path}")
                    
                    # 更新结果为可访问的URL路径
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


@app.route('/', methods=['GET', 'POST'])  # 添加POST方法支持
def home():
    if request.method == 'POST':
        # 处理POST请求
        return render_template('page.html')  # 或者重定向到其他页面
    else:
        # 处理GET请求
        return render_template('page.html')

@app.route('/generate', methods=['GET', 'POST'])
def generate():
    keyword = request.form.get('keyword', '').strip()
    if not keyword:
        return render_template('page.html', message="请输入环境描写")
    
    app.config['stuffs'] = []
    app.config['prompts'] = []
    
    app.config['room_type'] = determine_room_type(keyword)
    
    global is_api
    # 使用新的方法只获取物体列表
    object_names = chat_with_model(keyword, is_api).get_object_list_only()
    
    # 存储物体名称
    for obj_name in object_names:
        app.config['stuffs'].append(obj_name)
        app.config['prompts'].append(f"一个{obj_name}")  # 简单的描述

    context = {
        'stuffs': app.config['stuffs'],
        'prompts': app.config['prompts'],
        'room_type': app.config['room_type']
    }
    return render_template('search.html', **context)

@app.route('/search', methods=['GET', 'POST'])
def search():
    print("=== /search 路由被调用 ===")
    print(f"请求方法: {request.method}")
    print(f"表单数据: {request.form}")
    
    # 获取用户选择的物体索引
    selected_indices = request.form.getlist('var')
    print(f"选择的索引: {selected_indices}")
    
    if selected_indices:
        selected_objects = []
        for i in selected_indices:
            idx = int(i) - 1
            if idx < len(app.config['stuffs']):
                selected_objects.append(app.config['stuffs'][idx])
        
        app.config['selected_objects'] = selected_objects
        print(f"选择的物体: {selected_objects}")
    
    # 直接跳转到渲染页面
    return render_template('rendering.html')

@app.route('/start_render', methods=['POST'])
def start_render():
    """开始渲染场景 - 使用命令模式"""
    try:
        # 使用命令数据而不是场景数据
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=False)