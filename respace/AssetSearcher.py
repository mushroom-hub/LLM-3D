# deal with D:\3D-Dataset\info_collection\model_info.json
import json
import os
from typing import List, Dict, Any
import logging

class AssetSearcher:
    def __init__(self, model_info_path: str = r"D:\3D-Dataset\info_collection\model_info.json"):
        self.model_info_path = model_info_path
        self.model_info = self._load_model_info()
        self.logger = logging.getLogger(__name__)
    
    def _load_model_info(self) -> List[Dict]:
        """加载模型信息"""
        try:
            with open(self.model_info_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载模型信息失败: {e}")
            return []
    
    def search_models_by_category(self, category: str, max_styles: int = 4) -> List[Dict]:
        """
        根据类别搜索模型，返回最多4种不同风格的模型
        
        Args:
            category: 物体类别（如"床", "桌子"等）
            max_styles: 最大风格数量
            
        Returns:
            模型信息列表
        """
        if not self.model_info:
            return []
        
        # 将中文类别映射到英文（根据您的模型信息中的category字段）
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
            "床头柜": "Nightstand"
        }
        
        # 尝试匹配类别
        target_category = category_mapping.get(category, category)
        
        matched_models = []
        seen_styles = set()
        
        for model in self.model_info:
            # 匹配类别或超类别
            if (model.get('category', '').lower() == target_category.lower() or 
                model.get('super-category', '').lower() == target_category.lower()):
                
                style = model.get('style', 'Unknown')
                if style not in seen_styles or len(seen_styles) < max_styles:
                    matched_models.append(model)
                    seen_styles.add(style)
                
                # 如果已经找到足够的不同风格，停止搜索
                if len(seen_styles) >= max_styles:
                    break
        
        return matched_models
    
    def get_model_image_path(self, model_id: str) -> str:
        """获取模型图片路径"""
        return fr"D:\3D-Dataset\3D-FUTURE-model\{model_id}\image.jpg"
    
    def check_image_exists(self, model_id: str) -> bool:
        """检查模型图片是否存在"""
        image_path = self.get_model_image_path(model_id)
        return os.path.exists(image_path)

# 新建文件：EnhancedObjectProcessor.py
import os
from typing import List, Dict, Any
from AssetSearcher import AssetSearcher

class EnhancedObjectProcessor:
    def __init__(self):
        self.asset_searcher = AssetSearcher()
        self.selected_models = {}  # 存储用户选择的模型 {object_name: model_info}
    
    def process_object_list(self, object_names: List[str]) -> List[Dict]:
        """
        处理物体列表，为每个物体查找匹配的模型
        
        Returns:
            增强的物体信息列表，包含模型信息
        """
        enhanced_objects = []
        
        for obj_name in object_names:
            # 为每个物体搜索匹配的模型
            matched_models = self.asset_searcher.search_models_by_category(obj_name)
            
            if matched_models:
                # 选择第一个匹配的模型作为默认
                selected_model = matched_models[0]
                enhanced_objects.append({
                    'name': obj_name,
                    'models': matched_models,  # 所有匹配的模型
                    'selected_model': selected_model,  # 默认选择的模型
                    'description': f'{obj_name} 家具风格: {selected_model.get("style", "Unknown")}',
                    'has_models': True
                })
            else:
                # 没有找到匹配的模型
                enhanced_objects.append({
                    'name': obj_name,
                    'models': [],
                    'selected_model': None,
                    'description': f'{obj_name} (无匹配模型)',
                    'has_models': False
                })
        
        return enhanced_objects
    
    def update_selected_model(self, object_name: str, model_id: str, enhanced_objects: List[Dict]) -> bool:
        """更新用户选择的模型"""
        for obj in enhanced_objects:
            if obj['name'] == object_name:
                for model in obj['models']:
                    if model['model_id'] == model_id:
                        obj['selected_model'] = model
                        self.selected_models[object_name] = model
                        return True
        return False
    
    def get_selected_model_ids(self) -> List[str]:
        """获取所有选择的模型ID"""
        return [model['model_id'] for model in self.selected_models.values()]