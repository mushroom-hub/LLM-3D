from pymongo import MongoClient
import pprint
from typing import List, Optional, Dict, Any

def check_db_connection():
    """检查MongoDB连接"""
    try:
        client = MongoClient(
            'localhost', 27017,
            serverSelectionTimeoutMS=1000
        )
        client.server_info()
        print("✅ 数据库连接正常")
        print("数据库列表:", client.list_database_names())
        return True
    except Exception as e:
        print(f"❌ 数据库连接失败: {e}")
        return False

class ModelSearcher:
    def __init__(self):
        """初始化MongoDB连接"""
        self.client = MongoClient('localhost', 27017)
        self.db = self.client["models"]
        self.collection = self.db["models_collection"]
        print("当前集合:", self.db.list_collection_names())

    def search_by_prompt(self, prompt: str) -> Optional[List[Dict[str, Any]]]:
        """根据用户输入搜索模型"""
        try:
            query = {
                "$or": [
                    {"name": {"$regex": prompt, "$options": "i"}},
                    {"category": {"$regex": prompt, "$options": "i"}},
                    {"tags": {"$elemMatch": {"$regex": prompt, "$options": "i"}}}
                ]
            }
            results = list(self.collection.find(query))
            print(f"✅ 找到 {len(results)} 个匹配模型")
            for model in results:
                print("\n" + "="*50)
                print(f"模型ID: {model.get('id')}")
                print(f"名称: {model.get('name')}")
                print(f"类别: {model.get('category')}")
                print(f"标签: {', '.join(model.get('tags', []))}")
                print(f"文件URL: {model.get('file_url')}")
                pprint.pprint(model.get("metadata", {}))
            return results
        except Exception as e:
            print(f"❌ 搜索失败: {type(e).__name__}: {str(e)}")
            return None

    def close(self):
        """关闭数据库连接"""
        self.client.close()



def main(user_prompt: str = None):
    """主函数（支持直接调用或外部传参）"""
    # 如果没有传入user_prompt，则从控制台获取输入
    if user_prompt is None:
        user_prompt = input("🔍 请输入搜索关键词: ").strip()
        if not user_prompt:
            print("❌ 输入不能为空")
            return

    # 1. 连接数据库并搜索
    if not check_db_connection():
        return

    searcher = ModelSearcher()

    searcher.search_by_prompt(user_prompt or user_prompt)

    searcher.close()

if __name__ == "__main__":
    # 直接运行脚本时从控制台获取输入
    main()