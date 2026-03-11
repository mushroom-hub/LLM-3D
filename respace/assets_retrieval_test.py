 # import os
# os.environ["PYOPENGL_PLATFORM"] = "egl"

# import json
# import numpy as np
# from transformers import AutoTokenizer, SiglipTextModel
# import torch
# import torch.nn as nn
# import uuid
# import copy
# from dotenv import load_dotenv

# from src.respace import ReSpace
# from pathlib import Path
# import datetime 

# os.environ["ZHIPU_API_KEY"] = "1479e00599eb47fdb1f195522e0ee0c4.JhmukvpF06bgxQ7M"

# class Word2Vector:
#     """中英文嵌入向量计算类"""
#     def __init__(self, is_api: bool = False, model: str = "embedding-3") -> None:
#         self.list_item = []
#         self.vectors = []
#         self.embedding_model = model
#         self.is_api = is_api
#         self.embedding_dim = 1024
#         if self.is_api:
#             from zhipuai import ZhipuAI
#             self.client = ZhipuAI(api_key=os.getenv("ZHIPU_API_KEY"))

#     def get_embedding(self, context: list) -> list[float]:
#         if self.is_api:
#             try:
#                 context1 = context[0]
#                 vector1 = self.client.embeddings.create(input=[context1], model=self.embedding_model).data[0].embedding
#                 vector1_norm = vector1 / np.linalg.norm(vector1)
#                 context2 = context[1]
#                 vector2 = self.client.embeddings.create(input=[context2], model=self.embedding_model).data[0].embedding
#                 vector2_norm = vector2 / np.linalg.norm(vector2)
#                 combined_vector = 0.8 * vector1_norm + 0.2 * vector2_norm
                
#                 if len(combined_vector) != self.embedding_dim:
#                     if len(combined_vector) > self.embedding_dim:
#                         combined_vector = combined_vector[:self.embedding_dim]
#                     else:
#                         padding = np.zeros(self.embedding_dim - len(combined_vector))
#                         combined_vector = np.concatenate([combined_vector, padding])
                
#                 return combined_vector.tolist()
#             except Exception as e:
#                 print(f"智谱API调用失败: {e}，使用随机向量")
#                 return np.random.randn(self.embedding_dim).tolist()
#         else:
#             return np.random.randn(self.embedding_dim).tolist()

# class AssetRetrievalModule(nn.Module):
#     def __init__(self, temp, top_p, top_k, 
#                  use_chinese_embeddings=False, rand_seed=None, accelerator=None, 
#                  dvc=None, do_print=False, is_sft_training=False):
#         super().__init__()

#         self.accelerator = accelerator
#         self.dvc = dvc
#         self.use_chinese_embeddings = use_chinese_embeddings
#         self.do_print = do_print
#         self.embedding_dim = 1024

#         try:
#             print("=== 开始设备初始化 ===")
#             self.device = self._get_device()
#             print(f"使用设备: {self.device}")
#         except Exception as e:
#             print(f"设备初始化出错: {e}")
#             import traceback
#             traceback.print_exc()
#             self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#             print(f"使用默认设备: {self.device}")

#         # 加载真实的资产元数据
#         print("加载资产元数据...")
#         metadata_path = "/mnt/d/GradientSpace/respace/data/metadata/model_info_3dfuture_assets.json"
#         metadata_scaled_path = "/mnt/d/GradientSpace/respace/data/metadata/model_info_3dfuture_assets_scaled.json"
        
#         if not os.path.exists(metadata_path):
#             raise FileNotFoundError(f"元数据文件不存在: {metadata_path}")
        
#         self.all_assets_metadata = json.load(open(metadata_path, 'r', encoding='utf-8'))
#         print(f"成功加载 {len(self.all_assets_metadata)} 个资产元数据")
        
#         if os.path.exists(metadata_scaled_path):
#             self.all_assets_metadata_scaled = json.load(open(metadata_scaled_path, 'r', encoding='utf-8'))
#             print(f"成功加载 {len(self.all_assets_metadata_scaled)} 个缩放资产元数据")
#         else:
#             self.all_assets_metadata_scaled = {}
#             print("缩放元数据文件不存在，使用空字典")

#         # 初始化SIGLIP模型
#         print("初始化SIGLIP模型...")
#         self.siglip_model = SiglipTextModel.from_pretrained("google/siglip-so400m-patch14-384").to(self.device)
#         self.siglip_tokenizer = AutoTokenizer.from_pretrained("google/siglip-so400m-patch14-384")
        
#         # 初始化中文嵌入模型
#         if use_chinese_embeddings:
#             print("初始化中文嵌入模型...")
#             self.w2v = Word2Vector(is_api=True)
#             print("中文嵌入模型初始化成功")
        
#         # 创建基于真实元数据的资产目录
#         self._create_asset_catalog_from_metadata()
        
#         # 可学习参数
#         self.temp = torch.tensor(temp, device=self.device, requires_grad=True)

#         # 固定超参数
#         self.top_p = top_p
#         self.top_k = top_k
#         self.is_sft_training = is_sft_training

#     def _get_device(self):
#         """获取设备"""
#         if self.accelerator:
#             return self.accelerator.device
#         elif self.dvc:
#             return self.dvc
#         else:
#             return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
#     def _create_asset_catalog_from_metadata(self):
#         """从真实元数据创建资产目录 - 只关注语义匹配"""
#         device = self.device
        
#         # 收集所有资产的JID
#         all_jids = list(self.all_assets_metadata.keys())
        
#         if not all_jids:
#             raise ValueError("资产元数据为空，请检查文件内容")
        
#         print(f"处理 {len(all_jids)} 个资产...")
        
#         # 从元数据中提取描述信息
#         all_descriptions = []
#         valid_jids = []
#         asset_categories = []
        
#         for jid in all_jids:
#             asset = self.all_assets_metadata.get(jid)
#             if asset:
#                 description = asset.get("summary", "")
#                 category = self._infer_category_from_description(description)
                
#                 if description:  # 确保描述不为空
#                     all_descriptions.append(description)
#                     valid_jids.append(jid)
#                     asset_categories.append(category)
        
#         if not all_descriptions:
#             print("警告: 未找到有效的描述信息")
#             all_descriptions = ["default asset"] * len(all_jids)
#             valid_jids = all_jids
#             asset_categories = ["unknown"] * len(all_jids)
#         else:
#             print(f"找到 {len(valid_jids)} 个有效资产")
        
#         # 生成文本嵌入
#         print("生成文本嵌入...")
#         batch_size = 32
#         all_embeds = []
        
#         for i in range(0, len(all_descriptions), batch_size):
#             batch_descriptions = all_descriptions[i:i+batch_size]
            
#             with torch.no_grad():
#                 batch_embeds = self.get_text_embeddings(batch_descriptions, is_chinese=True)
#                 all_embeds.append(batch_embeds.cpu())
        
#         # 合并所有嵌入
#         all_embeds = torch.cat(all_embeds, dim=0).to(device)
        
#         print(f"生成的嵌入形状: {all_embeds.shape}")

#         # 归一化嵌入
#         self.all_embeds_catalog = torch.nn.functional.normalize(all_embeds, p=2, dim=1)
#         self.all_jids_catalog = valid_jids
#         self.all_descriptions_catalog = all_descriptions
#         self.asset_categories = asset_categories
        
#         # 分析资产类别分布
#         self._analyze_asset_categories()
        
#         print(f"创建资产目录完成:")
#         print(f"  - 嵌入形状: {self.all_embeds_catalog.shape}")
#         print(f"  - 资产数量: {len(self.all_jids_catalog)}")
#         print(f"  - 类别数量: {len(set(asset_categories))}")
        
#         # 验证一些样本
#         print("\n验证资产嵌入样本:")
#         for i in range(min(3, len(self.all_jids_catalog))):
#             self_similarity = torch.matmul(
#                 self.all_embeds_catalog[i:i+1], 
#                 self.all_embeds_catalog[i:i+1].T
#             ).item()
#             print(f"{i+1}. JID: {self.all_jids_catalog[i]}")
#             print(f"   描述: {self.all_descriptions_catalog[i][:80]}...")
#             print(f"   类别: {self.asset_categories[i]}")
#             print(f"   自相似度: {self_similarity:.4f}")
    
#     def _infer_category_from_description(self, description):
#         """从描述推断资产类别"""
#         description_lower = description.lower()
        
#         # 常见家具类别关键词映射
#         category_keywords = {
#             'sofa': ['sofa', 'couch', 'loveseat', 'settee'],
#             'bed': ['bed', 'mattress', 'headboard'],
#             'chair': ['chair', 'armchair', 'dining chair', 'desk chair'],
#             'table': ['table', 'desk', 'coffee table', 'dining table', 'side table'],
#             'cabinet': ['cabinet', 'wardrobe', 'cupboard', 'closet'],
#             'shelf': ['shelf', 'bookcase', 'bookshelf'],
#             'lamp': ['lamp', 'lighting', 'ceiling light', 'floor lamp'],
#             'storage': ['storage', 'chest', 'drawer', 'cabinet'],
#             'rug': ['rug', 'carpet', 'mat'],
#         }
        
#         for category, keywords in category_keywords.items():
#             if any(keyword in description_lower for keyword in keywords):
#                 return category
        
#         return 'other'

#     def _analyze_asset_categories(self):
#         """分析资产类别分布"""
#         from collections import Counter
#         category_counts = Counter(self.asset_categories)
        
#         print("\n=== 资产类别分析 ===")
#         for category, count in category_counts.most_common():
#             print(f"{category}: {count} 个资产")

#     def get_text_embeddings(self, txts, is_chinese=True):
#         """获取文本嵌入，支持中英文"""
#         if is_chinese and self.use_chinese_embeddings:
#             embeddings = []
#             for txt in txts:
#                 if self.do_print:
#                     print(f"获取中文嵌入: {txt}")
#                 embedding = self.w2v.get_embedding([txt, txt])
#                 embeddings.append(embedding)
            
#             embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
            
#             if embeddings_tensor.shape[1] != self.embedding_dim:
#                 print(f"警告: 嵌入维度不匹配，期望{self.embedding_dim}，实际{embeddings_tensor.shape[1]}")
#                 if embeddings_tensor.shape[1] > self.embedding_dim:
#                     embeddings_tensor = embeddings_tensor[:, :self.embedding_dim]
#                 else:
#                     padding = torch.zeros(embeddings_tensor.shape[0], 
#                                         self.embedding_dim - embeddings_tensor.shape[1])
#                     embeddings_tensor = torch.cat([embeddings_tensor, padding], dim=1)
            
#             return embeddings_tensor.to(self.device)
#         else:
#             # 英文SIGLIP嵌入
#             try:
#                 if isinstance(txts, str):
#                     txts = [txts]
                    
#                 inputs = self.siglip_tokenizer(
#                     txts, 
#                     truncation=True, 
#                     padding=True,
#                     max_length=64,
#                     return_tensors="pt", 
#                     return_attention_mask=True
#                 )

#                 inputs = {k: v.to(self.device) for k, v in inputs.items()}

#                 with torch.no_grad():
#                     outputs = self.siglip_model(**inputs)
                
#                 embeds = outputs.pooler_output
#                 if embeds.shape[1] != self.embedding_dim:
#                     print(f"SIGLIP嵌入维度调整: {embeds.shape[1]} -> {self.embedding_dim}")
#                     if embeds.shape[1] > self.embedding_dim:
#                         embeds = embeds[:, :self.embedding_dim]
#                     else:
#                         padding = torch.zeros(embeds.shape[0], 
#                                         self.embedding_dim - embeds.shape[1],
#                                         device=self.device)
#                         embeds = torch.cat([embeds, padding], dim=1)
                
#                 return embeds

#             except Exception as exc:
#                 print(f"计算文本嵌入时出错: {exc}")
#                 return torch.randn(len(txts), self.embedding_dim, dtype=torch.float32).to(self.device)

#     def compute_semantic_similarities(self, embeds):
#         """计算语义相似度 - 只使用语义匹配"""
#         if embeds.shape[1] != self.all_embeds_catalog.shape[1]:
#             print(f"维度不匹配: 查询嵌入{embeds.shape}, 资产嵌入{self.all_embeds_catalog.shape}")
#             if embeds.shape[1] > self.all_embeds_catalog.shape[1]:
#                 embeds = embeds[:, :self.all_embeds_catalog.shape[1]]
#             else:
#                 padding = torch.zeros(embeds.shape[0], 
#                                     self.all_embeds_catalog.shape[1] - embeds.shape[1],
#                                     device=embeds.device)
#                 embeds = torch.cat([embeds, padding], dim=1)
        
#         embeds_norm = torch.nn.functional.normalize(embeds, p=2, dim=1)
#         similarities = torch.matmul(self.all_embeds_catalog, embeds_norm.T)
        
#         if self.do_print:
#             print(f"语义相似度范围: [{similarities.min():.4f}, {similarities.max():.4f}]")
#             # 显示最高相似度
#             max_sim = similarities.max().item()
#             print(f"最高语义相似度: {max_sim:.4f}")
            
#         return similarities

#     def compute_final_probabilities(self, sims_batch):
#         """计算最终概率分布"""
#         all_probs_batch = []

#         for sims in sims_batch.T:
#             scaled_sims = sims / self.temp

#             # top-k 过滤
#             top_k = min(self.top_k, len(scaled_sims))
#             top_k_sims, top_k_indices = torch.topk(scaled_sims, k=top_k)
            
#             # softmax
#             top_k_probs = torch.softmax(top_k_sims, dim=0)

#             # 散射回完整张量
#             all_probs = torch.zeros_like(scaled_sims)
#             all_probs.scatter_(0, top_k_indices, top_k_probs)
#             all_probs = all_probs / (all_probs.sum() + 1e-8)

#             # top-p 过滤
#             sorted_probs, sorted_indices = torch.sort(all_probs, descending=True)
#             cumulative_probs = torch.cumsum(sorted_probs, dim=0)

#             sorted_indices_to_remove = cumulative_probs > self.top_p
#             sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
#             sorted_indices_to_remove[0] = False
#             indices_to_remove = sorted_indices[sorted_indices_to_remove]

#             all_probs[indices_to_remove] = 0
#             all_probs = all_probs / (all_probs.sum() + 1e-8)

#             all_probs_batch.append(all_probs)

#         return torch.stack(all_probs_batch, dim=1).T

#     def forward_batch(self, query_texts, query_sizes=None, is_chinese=True):
#         """批量前向传播 - 只使用语义匹配"""
#         query_embeds = self.get_text_embeddings(query_texts, is_chinese=is_chinese)
        
#         if self.do_print:
#             print(f"\n=== 批量查询 ===")
#             print(f"查询文本: {query_texts}")
#             print(f"查询嵌入形状: {query_embeds.shape}")
        
#         semantic_sims = self.compute_semantic_similarities(query_embeds)

#         # 直接使用语义相似度作为最终相似度（忽略尺寸）
#         weighted_sims = semantic_sims
        
#         if self.do_print:
#             print(f"最终相似度范围: [{weighted_sims.min():.4f}, {weighted_sims.max():.4f}]")
        
#         probs = self.compute_final_probabilities(weighted_sims)

#         return probs

#     def create_sampled_obj(self, obj, probs, is_greedy_sampling):
#         """创建采样对象 - 增强调试信息"""
#         if self.do_print:
#             print(f"\n采样对象描述: {obj.get('desc')}")
#             n_top = min(10, self.top_k)  # 显示更多结果
#             idxs_top = torch.argsort(probs, descending=True)[:n_top]
#             top_probs = torch.sort(probs, descending=True)[0].detach().cpu().numpy()[:n_top]
#             print("顶部概率:", [f"{p:.4f}" for p in top_probs])
            
#             # 计算语义相似度
#             query_embed = self.get_text_embeddings([obj.get('desc')], is_chinese=True)
#             semantic_sims = self.compute_semantic_similarities(query_embed)
            
#             jids = [self.all_jids_catalog[idx.item()] for idx in idxs_top]
#             for i, (idx, jid) in enumerate(zip(idxs_top, jids)):
#                 asset = self.all_assets_metadata.get(jid)
#                 if asset:
#                     desc = asset.get("summary", "无描述")
#                     size = asset.get("size", [0, 0, 0])
#                     category = self.asset_categories[idx]
#                 else:
#                     asset = self.all_assets_metadata_scaled.get(jid)
#                     if asset:
#                         orig_jid = asset.get("jid")
#                         orig_asset = self.all_assets_metadata.get(orig_jid)
#                         desc = orig_asset.get("summary", "无描述") if orig_asset else "未知资产"
#                         size = asset.get("size", [0, 0, 0])
#                         category = "scaled"
#                     else:
#                         desc = "未知资产"
#                         size = [0, 0, 0]
#                         category = "unknown"
                
#                 sem_sim = semantic_sims[idx, 0].item()
                
#                 print(f"{i+1}. 索引: [{idx}] - 概率: {top_probs[i]:.4f}")
#                 print(f"   类别: {category}")
#                 print(f"   JID: {jid}")
#                 print(f"   描述: {desc}")
#                 print(f"   尺寸: {size}")
#                 print(f"   语义相似度: {sem_sim:.4f}")
#                 print()

#         # 获取采样对象的JID
#         if obj.get("jid") is None:
#             if is_greedy_sampling:
#                 _, idx_sampled = torch.max(probs, dim=0)
#             else:
#                 idx_sampled = torch.multinomial(probs, num_samples=1)
#                 if self.do_print:
#                     print("采样索引:", idx_sampled)
#             jid_sampled_obj = self.all_jids_catalog[idx_sampled]
#         else:
#             jid_sampled_obj = obj.get("jid")

#         # 获取资产信息
#         asset = self.all_assets_metadata.get(jid_sampled_obj)
#         if asset is None:
#             asset = self.all_assets_metadata_scaled.get(jid_sampled_obj)
#             if asset:
#                 size_sampled_obj = asset.get("size", [1.0, 1.0, 1.0])
#                 orig_jid = asset.get("jid")
#                 orig_asset = self.all_assets_metadata.get(orig_jid)
#                 desc_sampled_obj = orig_asset.get("summary", "无描述") if orig_asset else "缩放资产"
#             else:
#                 desc_sampled_obj = "未知资产"
#                 size_sampled_obj = [1.0, 1.0, 1.0]
#         else:
#             desc_sampled_obj = asset.get("summary", "无描述")
#             size_sampled_obj = asset.get("size", [1.0, 1.0, 1.0])

#         new_obj = copy.deepcopy(obj)
#         new_obj.update({
#             "sampled_asset_jid": jid_sampled_obj,
#             "sampled_asset_desc": desc_sampled_obj,
#             "sampled_asset_size": size_sampled_obj,
#             "uuid": str(uuid.uuid4())
#         })

#         return new_obj

#     def sample_all_assets(self, scene, batch_size=64, is_greedy_sampling=True, is_chinese=True):
#         """采样所有资产"""
#         if self.do_print: 
#             print(f"采样完整场景... (对象数量: {len(scene.get('objects', []))})")

#         sampled_scene = copy.deepcopy(scene)
#         sampled_scene["objects"] = []

#         descriptions = [obj.get("desc") for obj in scene.get("objects", [])]
#         sizes = [obj.get("size", []) for obj in scene.get("objects", [])]
        
#         for batch_start in range(0, len(descriptions), batch_size):
#             batch_end = min(batch_start + batch_size, len(descriptions))

#             batch_descriptions = descriptions[batch_start:batch_end]
#             batch_sizes = sizes[batch_start:batch_end]

#             # 注意：这里不再传递尺寸信息
#             batch_probs = self.forward_batch(batch_descriptions, is_chinese=is_chinese)

#             for i, obj in enumerate(scene.get("objects", [])[batch_start:batch_end]):
#                 new_obj = self.create_sampled_obj(obj, batch_probs[i], is_greedy_sampling)
#                 sampled_scene["objects"].append(new_obj)

#         return sampled_scene

#     def debug_search(self, query_text, top_k=10):
#         """调试搜索功能 - 增强版本"""
#         print(f"\n=== 调试搜索: '{query_text}' ===")
        
#         # 获取查询嵌入
#         query_embed = self.get_text_embeddings([query_text], is_chinese=True)
#         print(f"查询嵌入形状: {query_embed.shape}")
        
#         # 计算语义相似度
#         semantic_sims = self.compute_semantic_similarities(query_embed)
#         print(f"语义相似度形状: {semantic_sims.shape}")
        
#         # 获取顶部匹配
#         top_k_sims, top_k_indices = torch.topk(semantic_sims.squeeze(), k=min(top_k, len(semantic_sims)))
        
#         print("顶部匹配结果:")
#         for i, (sim, idx) in enumerate(zip(top_k_sims, top_k_indices)):
#             jid = self.all_jids_catalog[idx]
#             desc = self.all_descriptions_catalog[idx]
#             category = self.asset_categories[idx]
#             print(f"{i+1}. 相似度: {sim:.4f}")
#             print(f"   类别: {category}")
#             print(f"   JID: {jid}")
#             print(f"   描述: {desc}")
#             print()

#     @staticmethod
#     def calculate_size_difference(size1, size2):
#         """计算尺寸差异"""
#         return np.linalg.norm(np.array(size1) - np.array(size2))


# # 使用示例
# if __name__ == "__main__":
#     print("开始初始化3D资产检索系统...")
    
#     # 初始化检索模块，只关注语义匹配
#     sampling_engine = AssetRetrievalModule(
#         temp=0.2, top_p=0.95, top_k=20, 
#         use_chinese_embeddings=True,
#         rand_seed=1234, do_print=True
#     )

#     # 测试搜索功能
#     print("\n=== 测试搜索功能 ===")
#     sampling_engine.debug_search("sofa")
#     sampling_engine.debug_search("bed")
#     sampling_engine.debug_search("chair")

#     test_scene_en = {
#         "bounds_top": [[-1.35, 2.6, 1.45], [0.15, 2.6, 1.45], [0.15, 2.6, 2.15], [1.35, 2.6, 2.15], [1.35, 2.6, -2.15], [-1.35, 2.6, -2.15]],
#         "bounds_bottom": [[-1.35, 0.0, 1.45], [0.15, 0.0, 1.45], [0.15, 0.0, 2.15], [1.35, 0.0, 2.15], [1.35, 0.0, -2.15], [-1.35, 0.0, -2.15]],
#         "room_type": "bedroom",
#         "objects": [
#             {
#                 "desc": "壁炉",
#                 "size": [1.5, 0.8, 0.8],
#                 "pos": [1.1, 0.0, 0.95],
#                 "rot": [0, -0.70711, 0, 0.70711]
#             },
#             {
#                 "desc": "书桌", 
#                 "size": [1.2, 0.6, 0.1],
#                 "pos": [1.23, 0.0, -1.42],
#                 "rot": [0, -0.70711, 0, 0.70711]
#             },
#             {
#                 "desc": "床",
#                 "size": [0.8, 0.4, 0.8],
#                 "pos": [1.34, 0.0, 1.42],
#                 "rot": [0, -0.70711, 0, 0.70711]
#             },
#         ]
#     }
    
#     respace = ReSpace()
#     # 采样资产
#     sampled_scene = sampling_engine.sample_all_assets(test_scene_en, is_greedy_sampling=True,is_chinese=True)
    
#     # 创建输出目录
#     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#     output_path_gen = Path(f"./3d_output_{timestamp}")
#     output_path_gen.mkdir(exist_ok=True)
    
#     respace.render_scene_frame(sampled_scene, filename="frame_full", pth_viz_output=output_path_gen)
#     respace.render_scene_360video(sampled_scene, filename="video-360_full", pth_viz_output=str(output_path_gen))
    
#     print("采样结果:")
#     print(json.dumps(sampled_scene, indent=2, ensure_ascii=False))

#     # 保存结果到当前目录
#     pth_tgt = output_path_gen / "sampled_scene_with_real_assets.json"
#     with open(pth_tgt, "w", encoding='utf-8') as fp:
#         json.dump(sampled_scene, fp, indent=4, ensure_ascii=False)
    
#     print(f"结果已保存到: {pth_tgt}")

import os
os.environ["PYOPENGL_PLATFORM"] = "egl"

import json
import numpy as np
from transformers import AutoTokenizer, SiglipTextModel
import torch
import torch.nn as nn
import uuid
import copy
from dotenv import load_dotenv
import pickle
import hashlib

from src.respace import ReSpace
from pathlib import Path
import datetime 

os.environ["ZHIPU_API_KEY"] = "3ccc767f63c34077a9ec9973dd2efa43.3my0UDQt9gnUy1GH"

class EmbeddingCache:
    """嵌入缓存管理器"""
    def __init__(self, cache_file="chinese_embeddings_cache.pkl"):
        self.cache_file = Path(cache_file)
        self.cache = self._load_cache()
        
    def _load_cache(self):
        """加载缓存文件"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    cache = pickle.load(f)
                print(f"加载嵌入缓存，包含 {len(cache)} 个条目")
                return cache
            except Exception as e:
                print(f"加载缓存失败: {e}，创建新缓存")
                return {}
        else:
            print("创建新的嵌入缓存")
            return {}
    
    def _save_cache(self):
        """保存缓存到文件"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
            print(f"保存嵌入缓存，包含 {len(self.cache)} 个条目")
        except Exception as e:
            print(f"保存缓存失败: {e}")
    
    def _get_cache_key(self, text):
        """生成缓存键"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def get_embedding(self, text, embedding_func):
        """获取嵌入，优先使用缓存"""
        cache_key = self._get_cache_key(text)
        
        if cache_key in self.cache:
            # print(f"缓存命中: {text}")
            return self.cache[cache_key]
        else:
            # print(f"缓存未命中，计算嵌入: {text}")
            embedding = embedding_func(text)
            self.cache[cache_key] = embedding
            self._save_cache()  # 每次新增都保存
            return embedding
    
    def batch_get_embeddings(self, texts, embedding_func):
        """批量获取嵌入"""
        results = []
        need_update = False
        
        for text in texts:
            cache_key = self._get_cache_key(text)
            if cache_key in self.cache:
                results.append(self.cache[cache_key])
            else:
                embedding = embedding_func(text)
                self.cache[cache_key] = embedding
                results.append(embedding)
                need_update = True
        
        if need_update:
            self._save_cache()
        
        return results

class Word2Vector:
    """中英文嵌入向量计算类"""
    def __init__(self, is_api: bool = False, model: str = "embedding-3") -> None:
        self.list_item = []
        self.vectors = []
        self.embedding_model = model
        self.is_api = is_api
        self.embedding_dim = 1024
        self.cache = EmbeddingCache("chinese_w2v_cache.pkl")  # 添加缓存
        
        if self.is_api:
            from zhipuai import ZhipuAI
            self.client = ZhipuAI(api_key=os.getenv("ZHIPU_API_KEY"))

    def get_embedding(self, context: list) -> list[float]:
        # 使用上下文文本作为缓存键
        cache_text = "|".join(context)
        
        # 检查缓存
        cached_result = self.cache.get_embedding(cache_text, self._compute_embedding)
        return cached_result
    
    def _compute_embedding(self, cache_text):
        """实际计算嵌入的方法"""
        context = cache_text.split("|")
        
        if self.is_api:
            try:
                context1 = context[0]
                vector1 = self.client.embeddings.create(input=[context1], model=self.embedding_model).data[0].embedding
                vector1_norm = vector1 / np.linalg.norm(vector1)
                context2 = context[1]
                vector2 = self.client.embeddings.create(input=[context2], model=self.embedding_model).data[0].embedding
                vector2_norm = vector2 / np.linalg.norm(vector2)
                combined_vector = 0.8 * vector1_norm + 0.2 * vector2_norm
                
                if len(combined_vector) != self.embedding_dim:
                    if len(combined_vector) > self.embedding_dim:
                        combined_vector = combined_vector[:self.embedding_dim]
                    else:
                        padding = np.zeros(self.embedding_dim - len(combined_vector))
                        combined_vector = np.concatenate([combined_vector, padding])
                
                return combined_vector.tolist()
            except Exception as e:
                print(f"智谱API调用失败: {e}，使用随机向量")
                return np.random.randn(self.embedding_dim).tolist()
        else:
            return np.random.randn(self.embedding_dim).tolist()

class AssetRetrievalModule(nn.Module):
    def __init__(self, temp, top_p, top_k, 
                 use_chinese_embeddings=False, rand_seed=None, accelerator=None, 
                 dvc=None, do_print=False, is_sft_training=False):
        super().__init__()

        self.accelerator = accelerator
        self.dvc = dvc
        self.use_chinese_embeddings = use_chinese_embeddings
        self.do_print = do_print
        self.embedding_dim = 1024
        self.embedding_cache = EmbeddingCache("semantic_embeddings_cache.pkl")  # 语义嵌入缓存

        try:
            print("=== 开始设备初始化 ===")
            self.device = self._get_device()
            print(f"使用设备: {self.device}")
        except Exception as e:
            print(f"设备初始化出错: {e}")
            import traceback
            traceback.print_exc()
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            print(f"使用默认设备: {self.device}")

        # 加载真实的资产元数据
        print("加载资产元数据...")
        metadata_path = "/mnt/d/GradientSpace/respace/data/metadata/model_info_3dfuture_assets.json"
        metadata_scaled_path = "/mnt/d/GradientSpace/respace/data/metadata/model_info_3dfuture_assets_scaled.json"
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"元数据文件不存在: {metadata_path}")
        
        self.all_assets_metadata = json.load(open(metadata_path, 'r', encoding='utf-8'))
        print(f"成功加载 {len(self.all_assets_metadata)} 个资产元数据")
        
        if os.path.exists(metadata_scaled_path):
            self.all_assets_metadata_scaled = json.load(open(metadata_scaled_path, 'r', encoding='utf-8'))
            print(f"成功加载 {len(self.all_assets_metadata_scaled)} 个缩放资产元数据")
        else:
            self.all_assets_metadata_scaled = {}
            print("缩放元数据文件不存在，使用空字典")

        # 初始化SIGLIP模型
        print("初始化SIGLIP模型...")
        self.siglip_model = SiglipTextModel.from_pretrained("google/siglip-so400m-patch14-384").to(self.device)
        self.siglip_tokenizer = AutoTokenizer.from_pretrained("google/siglip-so400m-patch14-384")
        
        # 初始化中文嵌入模型
        if use_chinese_embeddings:
            print("初始化中文嵌入模型...")
            self.w2v = Word2Vector(is_api=True)
            print("中文嵌入模型初始化成功")
        
        # 创建基于真实元数据的资产目录
        self._create_asset_catalog_from_metadata()
        
        # 可学习参数
        self.temp = torch.tensor(temp, device=self.device, requires_grad=True)

        # 固定超参数
        self.top_p = top_p
        self.top_k = top_k
        self.is_sft_training = is_sft_training

    def _get_device(self):
        """获取设备"""
        if self.accelerator:
            return self.accelerator.device
        elif self.dvc:
            return self.dvc
        else:
            return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
    def _create_asset_catalog_from_metadata(self):
        """从真实元数据创建资产目录 - 只关注语义匹配"""
        device = self.device
        
        # 收集所有资产的JID
        all_jids = list(self.all_assets_metadata.keys())
        
        if not all_jids:
            raise ValueError("资产元数据为空，请检查文件内容")
        
        print(f"处理 {len(all_jids)} 个资产...")
        
        # 从元数据中提取描述信息
        all_descriptions = []
        valid_jids = []
        asset_categories = []
        
        for jid in all_jids:
            asset = self.all_assets_metadata.get(jid)
            if asset:
                description = asset.get("summary", "")
                category = self._infer_category_from_description(description)
                
                if description:  # 确保描述不为空
                    all_descriptions.append(description)
                    valid_jids.append(jid)
                    asset_categories.append(category)
        
        if not all_descriptions:
            print("警告: 未找到有效的描述信息")
            all_descriptions = ["default asset"] * len(all_jids)
            valid_jids = all_jids
            asset_categories = ["unknown"] * len(all_jids)
        else:
            print(f"找到 {len(valid_jids)} 个有效资产")
        
        # 生成文本嵌入
        print("生成文本嵌入...")
        batch_size = 32
        all_embeds = []
        
        for i in range(0, len(all_descriptions), batch_size):
            batch_descriptions = all_descriptions[i:i+batch_size]
            
            with torch.no_grad():
                batch_embeds = self.get_text_embeddings(batch_descriptions, is_chinese=False, use_cache=True)
                all_embeds.append(batch_embeds.cpu())
        
        # 合并所有嵌入
        all_embeds = torch.cat(all_embeds, dim=0).to(device)
        
        print(f"生成的嵌入形状: {all_embeds.shape}")

        # 归一化嵌入
        self.all_embeds_catalog = torch.nn.functional.normalize(all_embeds, p=2, dim=1)
        self.all_jids_catalog = valid_jids
        self.all_descriptions_catalog = all_descriptions
        self.asset_categories = asset_categories
        
        # 分析资产类别分布
        self._analyze_asset_categories()
        
        print(f"创建资产目录完成:")
        print(f"  - 嵌入形状: {self.all_embeds_catalog.shape}")
        print(f"  - 资产数量: {len(self.all_jids_catalog)}")
        print(f"  - 类别数量: {len(set(asset_categories))}")
        
        # 验证一些样本
        print("\n验证资产嵌入样本:")
        for i in range(min(3, len(self.all_jids_catalog))):
            self_similarity = torch.matmul(
                self.all_embeds_catalog[i:i+1], 
                self.all_embeds_catalog[i:i+1].T
            ).item()
            print(f"{i+1}. JID: {self.all_jids_catalog[i]}")
            print(f"   描述: {self.all_descriptions_catalog[i][:80]}...")
            print(f"   类别: {self.asset_categories[i]}")
            print(f"   自相似度: {self_similarity:.4f}")
    
    def _infer_category_from_description(self, description):
        """从描述推断资产类别"""
        description_lower = description.lower()
        
        # 常见家具类别关键词映射
        category_keywords = {
            'sofa': ['sofa', 'couch', 'loveseat', 'settee'],
            'bed': ['bed', 'mattress', 'headboard'],
            'chair': ['chair', 'armchair', 'dining chair', 'desk chair'],
            'table': ['table', 'desk', 'coffee table', 'dining table', 'side table'],
            'cabinet': ['cabinet', 'wardrobe', 'cupboard', 'closet'],
            'shelf': ['shelf', 'bookcase', 'bookshelf'],
            'lamp': ['lamp', 'lighting', 'ceiling light', 'floor lamp'],
            'storage': ['storage', 'chest', 'drawer', 'cabinet'],
            'rug': ['rug', 'carpet', 'mat'],
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                return category
        
        return 'other'

    def _analyze_asset_categories(self):
        """分析资产类别分布"""
        from collections import Counter
        category_counts = Counter(self.asset_categories)
        
        print("\n=== 资产类别分析 ===")
        for category, count in category_counts.most_common():
            print(f"{category}: {count} 个资产")

    def get_text_embeddings(self, txts, is_chinese=True, use_cache=True):
        """获取文本嵌入，支持中英文和缓存"""
        if is_chinese and self.use_chinese_embeddings:
            embeddings = []
            for txt in txts:
                if self.do_print:
                    print(f"获取中文嵌入: {txt}")
                
                if use_cache:
                    # 使用缓存获取嵌入
                    embedding = self.embedding_cache.get_embedding(
                        txt, 
                        lambda x: self._compute_chinese_embedding([x])
                    )
                else:
                    embedding = self._compute_chinese_embedding([txt])
                
                embeddings.append(embedding)
            
            embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
            
            if embeddings_tensor.shape[1] != self.embedding_dim:
                print(f"警告: 嵌入维度不匹配，期望{self.embedding_dim}，实际{embeddings_tensor.shape[1]}")
                if embeddings_tensor.shape[1] > self.embedding_dim:
                    embeddings_tensor = embeddings_tensor[:, :self.embedding_dim]
                else:
                    padding = torch.zeros(embeddings_tensor.shape[0], 
                                        self.embedding_dim - embeddings_tensor.shape[1])
                    embeddings_tensor = torch.cat([embeddings_tensor, padding], dim=1)
            
            return embeddings_tensor.to(self.device)
        else:
            # 英文SIGLIP嵌入
            try:
                if isinstance(txts, str):
                    txts = [txts]
                    
                inputs = self.siglip_tokenizer(
                    txts, 
                    truncation=True, 
                    padding=True,
                    max_length=64,
                    return_tensors="pt", 
                    return_attention_mask=True
                )

                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.siglip_model(**inputs)
                
                embeds = outputs.pooler_output
                if embeds.shape[1] != self.embedding_dim:
                    print(f"SIGLIP嵌入维度调整: {embeds.shape[1]} -> {self.embedding_dim}")
                    if embeds.shape[1] > self.embedding_dim:
                        embeds = embeds[:, :self.embedding_dim]
                    else:
                        padding = torch.zeros(embeds.shape[0], 
                                        self.embedding_dim - embeds.shape[1],
                                        device=self.device)
                        embeds = torch.cat([embeds, padding], dim=1)
                
                return embeds

            except Exception as exc:
                print(f"计算文本嵌入时出错: {exc}")
                return torch.randn(len(txts), self.embedding_dim, dtype=torch.float32).to(self.device)
    
    def _compute_chinese_embedding(self, txts):
        """计算中文嵌入的内部方法"""
        embedding = self.w2v.get_embedding([txts[0], txts[0]])
        return embedding

    def compute_semantic_similarities(self, embeds):
        """计算语义相似度 - 只使用语义匹配"""
        if embeds.shape[1] != self.all_embeds_catalog.shape[1]:
            print(f"维度不匹配: 查询嵌入{embeds.shape}, 资产嵌入{self.all_embeds_catalog.shape}")
            if embeds.shape[1] > self.all_embeds_catalog.shape[1]:
                embeds = embeds[:, :self.all_embeds_catalog.shape[1]]
            else:
                padding = torch.zeros(embeds.shape[0], 
                                    self.all_embeds_catalog.shape[1] - embeds.shape[1],
                                    device=embeds.device)
                embeds = torch.cat([embeds, padding], dim=1)
        
        embeds_norm = torch.nn.functional.normalize(embeds, p=2, dim=1)
        similarities = torch.matmul(self.all_embeds_catalog, embeds_norm.T)
        
        if self.do_print:
            print(f"语义相似度范围: [{similarities.min():.4f}, {similarities.max():.4f}]")
            # 显示最高相似度
            max_sim = similarities.max().item()
            print(f"最高语义相似度: {max_sim:.4f}")
            
        return similarities

    def compute_final_probabilities(self, sims_batch):
        """计算最终概率分布"""
        all_probs_batch = []

        for sims in sims_batch.T:
            scaled_sims = sims / self.temp

            # top-k 过滤
            top_k = min(self.top_k, len(scaled_sims))
            top_k_sims, top_k_indices = torch.topk(scaled_sims, k=top_k)
            
            # softmax
            top_k_probs = torch.softmax(top_k_sims, dim=0)

            # 散射回完整张量
            all_probs = torch.zeros_like(scaled_sims)
            all_probs.scatter_(0, top_k_indices, top_k_probs)
            all_probs = all_probs / (all_probs.sum() + 1e-8)

            # top-p 过滤
            sorted_probs, sorted_indices = torch.sort(all_probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=0)

            sorted_indices_to_remove = cumulative_probs > self.top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            indices_to_remove = sorted_indices[sorted_indices_to_remove]

            all_probs[indices_to_remove] = 0
            all_probs = all_probs / (all_probs.sum() + 1e-8)

            all_probs_batch.append(all_probs)

        return torch.stack(all_probs_batch, dim=1).T

    def forward_batch(self, query_texts, query_sizes=None, is_chinese=False):
        """批量前向传播 - 只使用语义匹配"""
        query_embeds = self.get_text_embeddings(query_texts, is_chinese=is_chinese, use_cache=True)
        
        if self.do_print:
            print(f"\n=== 批量查询 ===")
            print(f"查询文本: {query_texts}")
            print(f"查询嵌入形状: {query_embeds.shape}")
        
        semantic_sims = self.compute_semantic_similarities(query_embeds)

        # 直接使用语义相似度作为最终相似度（忽略尺寸）
        weighted_sims = semantic_sims
        
        if self.do_print:
            print(f"最终相似度范围: [{weighted_sims.min():.4f}, {weighted_sims.max():.4f}]")
        
        probs = self.compute_final_probabilities(weighted_sims)

        return probs

    def create_sampled_obj(self, obj, probs, is_greedy_sampling):
        """创建采样对象 - 增强调试信息"""
        if self.do_print:
            print(f"\n采样对象描述: {obj.get('desc')}")
            n_top = min(10, self.top_k)  # 显示更多结果
            idxs_top = torch.argsort(probs, descending=True)[:n_top]
            top_probs = torch.sort(probs, descending=True)[0].detach().cpu().numpy()[:n_top]
            print("顶部概率:", [f"{p:.4f}" for p in top_probs])
            
            # 计算语义相似度
            query_embed = self.get_text_embeddings([obj.get('desc')], is_chinese=False, use_cache=True)
            semantic_sims = self.compute_semantic_similarities(query_embed)
            
            jids = [self.all_jids_catalog[idx.item()] for idx in idxs_top]
            for i, (idx, jid) in enumerate(zip(idxs_top, jids)):
                asset = self.all_assets_metadata.get(jid)
                if asset:
                    desc = asset.get("summary", "无描述")
                    size = asset.get("size", [0, 0, 0])
                    category = self.asset_categories[idx]
                else:
                    asset = self.all_assets_metadata_scaled.get(jid)
                    if asset:
                        orig_jid = asset.get("jid")
                        orig_asset = self.all_assets_metadata.get(orig_jid)
                        desc = orig_asset.get("summary", "无描述") if orig_asset else "未知资产"
                        size = asset.get("size", [0, 0, 0])
                        category = "scaled"
                    else:
                        desc = "未知资产"
                        size = [0, 0, 0]
                        category = "unknown"
                
                sem_sim = semantic_sims[idx, 0].item()
                
                print(f"{i+1}. 索引: [{idx}] - 概率: {top_probs[i]:.4f}")
                print(f"   类别: {category}")
                print(f"   JID: {jid}")
                print(f"   描述: {desc}")
                print(f"   尺寸: {size}")
                print(f"   语义相似度: {sem_sim:.4f}")
                print()

        # 获取采样对象的JID
        if obj.get("jid") is None:
            if is_greedy_sampling:
                _, idx_sampled = torch.max(probs, dim=0)
            else:
                idx_sampled = torch.multinomial(probs, num_samples=1)
                if self.do_print:
                    print("采样索引:", idx_sampled)
            jid_sampled_obj = self.all_jids_catalog[idx_sampled]
        else:
            jid_sampled_obj = obj.get("jid")

        # 获取资产信息
        asset = self.all_assets_metadata.get(jid_sampled_obj)
        if asset is None:
            asset = self.all_assets_metadata_scaled.get(jid_sampled_obj)
            if asset:
                size_sampled_obj = asset.get("size", [1.0, 1.0, 1.0])
                orig_jid = asset.get("jid")
                orig_asset = self.all_assets_metadata.get(orig_jid)
                desc_sampled_obj = orig_asset.get("summary", "无描述") if orig_asset else "缩放资产"
            else:
                desc_sampled_obj = "未知资产"
                size_sampled_obj = [1.0, 1.0, 1.0]
        else:
            desc_sampled_obj = asset.get("summary", "无描述")
            size_sampled_obj = asset.get("size", [1.0, 1.0, 1.0])

        new_obj = copy.deepcopy(obj)
        new_obj.update({
            "sampled_asset_jid": jid_sampled_obj,
            "sampled_asset_desc": desc_sampled_obj,
            "sampled_asset_size": size_sampled_obj,
            "uuid": str(uuid.uuid4())
        })

        return new_obj

    def sample_all_assets(self, scene, batch_size=64, is_greedy_sampling=True, is_chinese=False):
        """采样所有资产"""
        if self.do_print: 
            print(f"采样完整场景... (对象数量: {len(scene.get('objects', []))})")

        sampled_scene = copy.deepcopy(scene)
        sampled_scene["objects"] = []

        descriptions = [obj.get("desc") for obj in scene.get("objects", [])]
        sizes = [obj.get("size", []) for obj in scene.get("objects", [])]
        
        for batch_start in range(0, len(descriptions), batch_size):
            batch_end = min(batch_start + batch_size, len(descriptions))

            batch_descriptions = descriptions[batch_start:batch_end]
            batch_sizes = sizes[batch_start:batch_end]

            # 注意：这里不再传递尺寸信息
            batch_probs = self.forward_batch(batch_descriptions, is_chinese=is_chinese)

            for i, obj in enumerate(scene.get("objects", [])[batch_start:batch_end]):
                new_obj = self.create_sampled_obj(obj, batch_probs[i], is_greedy_sampling)
                sampled_scene["objects"].append(new_obj)

        return sampled_scene

    def debug_search(self, query_text, top_k=10):
        """调试搜索功能 - 增强版本"""
        print(f"\n=== 调试搜索: '{query_text}' ===")
        
        # 获取查询嵌入
        query_embed = self.get_text_embeddings([query_text], is_chinese=False, use_cache=True)
        print(f"查询嵌入形状: {query_embed.shape}")
        
        # 计算语义相似度
        semantic_sims = self.compute_semantic_similarities(query_embed)
        print(f"语义相似度形状: {semantic_sims.shape}")
        
        # 获取顶部匹配
        top_k_sims, top_k_indices = torch.topk(semantic_sims.squeeze(), k=min(top_k, len(semantic_sims)))
        
        print("顶部匹配结果:")
        for i, (sim, idx) in enumerate(zip(top_k_sims, top_k_indices)):
            jid = self.all_jids_catalog[idx]
            desc = self.all_descriptions_catalog[idx]
            category = self.asset_categories[idx]
            print(f"{i+1}. 相似度: {sim:.4f}")
            print(f"   类别: {category}")
            print(f"   JID: {jid}")
            print(f"   描述: {desc}")
            print()

    @staticmethod
    def calculate_size_difference(size1, size2):
        """计算尺寸差异"""
        return np.linalg.norm(np.array(size1) - np.array(size2))


# 使用示例
if __name__ == "__main__":
    print("开始初始化3D资产检索系统...")
    
    # 初始化检索模块，只关注语义匹配
    sampling_engine = AssetRetrievalModule(
        temp=0.2, top_p=0.95, top_k=20, 
        use_chinese_embeddings=False,
        rand_seed=1234, do_print=True
    )

    # # 测试搜索功能
    # print("\n=== 测试搜索功能 ===")
    # sampling_engine.debug_search("sofa")
    # sampling_engine.debug_search("bed")
    # sampling_engine.debug_search("chair")

    # test_scene_en = {
    #     "bounds_top": [[-1.35, 2.6, 1.45], [0.15, 2.6, 1.45], [0.15, 2.6, 2.15], [1.35, 2.6, 2.15], [1.35, 2.6, -2.15], [-1.35, 2.6, -2.15]],
    #     "bounds_bottom": [[-1.35, 0.0, 1.45], [0.15, 0.0, 1.45], [0.15, 0.0, 2.15], [1.35, 0.0, 2.15], [1.35, 0.0, -2.15], [-1.35, 0.0, -2.15]],
    #     "room_type": "bedroom",
    #     "objects": [
    #         {
    #             "desc": "床头柜",
    #             "size": [1.5, 0.8, 0.8],
    #             "pos": [1.1, 0.0, 0.95],
    #             "rot": [0, -0.70711, 0, 0.70711]
    #         },
    #         {
    #             "desc": "书桌", 
    #             "size": [1.2, 0.6, 0.1],
    #             "pos": [1.23, 0.0, -1.42],
    #             "rot": [0, -0.70711, 0, 0.70711]
    #         },
    #         {
    #             "desc": "床",
    #             "size": [0.8, 0.4, 0.8],
    #             "pos": [1.34, 0.0, 1.42],
    #             "rot": [0, -0.70711, 0, 0.70711]
    #         },
    #     ]
    # }
    
    # respace = ReSpace()
    # # 采样资产
    # sampled_scene = sampling_engine.sample_all_assets(test_scene_en, is_greedy_sampling=True,is_chinese=True)
    
    # # 创建输出目录
    # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # output_path_gen = Path(f"./3d_output_{timestamp}")
    # output_path_gen.mkdir(exist_ok=True)
    
    # respace.render_scene_frame(sampled_scene, filename="frame_full", pth_viz_output=output_path_gen)
    # respace.render_scene_360video(sampled_scene, filename="video-360_full", pth_viz_output=str(output_path_gen))
    
    # print("采样结果:")
    # print(json.dumps(sampled_scene, indent=2, ensure_ascii=False))

    # # 保存结果到当前目录
    # pth_tgt = output_path_gen / "sampled_scene_with_real_assets.json"
    # with open(pth_tgt, "w", encoding='utf-8') as fp:
    #     json.dump(sampled_scene, fp, indent=4, ensure_ascii=False)
    
    # print(f"结果已保存到: {pth_tgt}")