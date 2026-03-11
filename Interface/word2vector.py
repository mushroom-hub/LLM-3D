import numpy as np
import os
import json

os.environ["ZHIPU_API_KEY"] = "6731a3fc70e34d41ba6bb66fb93a9953.kRyMUbdwahOaXPER"
os.environ["ZHIPU_BASE_URL"] = "https://open.bigmodel.cn/api/paas/v4/embeddings"

def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    calculate cosine similarity between two vectors
    """
    dot_prod = np.dot(vec1, vec2)
    magnitude = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if not magnitude:
        return 0
    return dot_prod / magnitude

class Word2Vector:
    """
    Base class of embeddings
    """
    def __init__(self, is_api: bool, model: str = "embedding-3") -> None:
        self.list_item = []
        self.vectors = []
        self.embedding_model = model
        self.is_api = is_api
        if self.is_api:
            from zhipuai import ZhipuAI
            self.client = ZhipuAI(api_key=os.getenv("ZHIPU_API_KEY"))

    def get_embedding(self, context: list) -> list[float]:
        if self.is_api:
            context1 = context[0]
            vector1 = self.client.embeddings.create(input=[context1], model=self.embedding_model).data[0].embedding
            vector1_norm = vector1 / np.linalg.norm(vector1)
            context2 = context[1]
            vector2 = self.client.embeddings.create(input=[context2], model=self.embedding_model).data[0].embedding
            vector2_norm = vector2 / np.linalg.norm(vector2)
            return 0.8 * vector1_norm + 0.2 * vector2_norm
        else:
            return []

    def add_vector(self, list_item: list[list]):
        for item in list_item:
            vector = self.get_embedding(item)
            self.vectors.append(vector)

    def add_list_item(self, list_item: list[list]):
        for item in list_item:
            self.list_item.append(item)
        self.add_vector(list_item)

    def item2vector(self, list_item: list[list]) -> list[list]:
        """
        list_item:
        [["itemA","introduceA"],["itemB","introduceB"]]
        vectors:
        [["itemA","introduceA","vectorA"],["itemB","introduceB","vectorB"]]
        """
        vectors = list_item
        for item in vectors:
            vector = self.get_embedding(item)
            item.append(vector)
        return vectors

    def match(self, item: list, num: int):
        item_vector = self.get_embedding(item)
        if len(self.vectors):
            result = np.array([cosine_similarity(item_vector, vector)
                             for vector in self.vectors])
            result_list = np.array(self.list_item)[result.argsort()[-num:][::-1]].tolist()
            return result_list
        else:
            return []

w2v = Word2Vector(is_api=True)

with open("data/bedroom-data.json", "r", encoding='utf-8') as file:
    content = json.load(file)

item_list = []
for items in content['items']:
    item_list.append([items['name'],items['description']])

w2v.add_list_item(item_list)
match_items = w2v.match(["木制床头柜","温润的实木打造，纹理细腻，色泽柔和，散发出自然质朴的气息"],5)
print(match_items)
