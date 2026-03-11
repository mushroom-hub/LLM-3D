# from typing import List, Dict, TypedDict
# from enum import Enum
# import os
# import warnings
# import langchain_core.utils
# from langchain.agents import create_agent
# from langchain_core.messages import HumanMessage
# from langchain_openai import ChatOpenAI
# from langgraph.graph import StateGraph, START, END
# from langchain_core.tools import tool
# from langchain_tavily import TavilySearch
# from langchain_ollama import ChatOllama

# warnings.filterwarnings("ignore")

# # 配置Tavily的API
# os.environ["TAVILY_API_KEY"] = "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"

# # 配置在线大模型的API
# os.environ["OPEN_API_KEY"] = "sk-cf5ccda9343c41dc999d8252a287512e"
# os.environ["OPEN_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# # 定义搜索工具
# search = TavilySearch(
#     max_results=5,
#     include_answer=True,
#     include_raw_content=False,
#     include_images=False,
# )

# @tool
# def search_web(query: str) -> str:
#     """联网搜索"""
#     result = search.invoke({"query": query})
#     return result["answer"]

# # 创建工具实例
# search_tool_web = [search_web]

# # 定义状态类型
# class AgentState(TypedDict):
#     input: List[Dict[str, str]] #历史消息
#     terminated: List[int] #Supervisor返回的各种状态，如extender，splicer和END

# # 定义节点类型
# class AgentType(Enum):
#     judger = 0
#     extender = 1
#     naturalEnvBuilder = 2
#     humanEnvBuilder = 3
#     splicer = 4
#     supervisor = 5
#     roomBuilder = 6
#     builder = 7

# # 定义Agent的基础类
# class GenerateAgent:
#     def __init__(self, name: str, role: str, system_prompt: str):
#         self.name = name
#         self.role = role
#         self.system_prompt = system_prompt
#         self.memory: List[Dict] = []

#     def update_memory(self, message: Dict):
#         self.memory.append(message)

#     def get_context(self, messages: List[Dict], AgentType) -> str:
#         memory_str = "\n".join([f"{m['speaker']}: {m['content']}" for m in self.memory[-5:]]) #中期记忆
#         if AgentType == AgentType.splicer:
#             temp_message = "\n".join([f"{m['content']}" for m in messages[-2:]]) #中期记忆
#             memory_str = ""
#         elif AgentType == AgentType.supervisor  or AgentType == AgentType.builder:
#             temp_message = "\n".join([f"{messages[0]['content']}", f"{messages[-1]['content']}"]) #长期记忆
#             memory_str = "" #无中期记忆
#         elif AgentType == AgentType.extender:
#             #长期记忆
#             temp_message = messages[0]["content"] if len(messages) == 1 else "\n".join([f"{messages[0]['content']}", f"{messages[-1]['content']}"])
#         else:
#             #短期记忆
#             temp_message = messages[-1]["content"]
#         return f"""
# {self.system_prompt}
# 历史对话：
# {temp_message}
# {memory_str}
# 现在轮到你创作。请根据历史对话和你的角色，继续进行工作。
# """

# class chat_with_model:
#     def __init__(self, question, is_api: bool = False):
#         self.env = question
#         self.output = ""
#         # 创建判断室外或室内场景的agent
#         self.judger = GenerateAgent(
#             "Judger",
#             "judger",
#             "你是一位经验丰富的3D场景构建者，正在参与一场3D场景环境构建的工作。"
#             "你的职责是根据接受到的场景环境提示词，判断该场景为室外场景，还是室内场景，如果是室内场景则输出A，如果是室外场景则输出B"
#         )
#         self.is_api = is_api
#         if is_api == 1:
#             self.model = ChatOpenAI(
#                 model="qwen-plus",
#                 base_url=os.environ["OPEN_URL"],
#                 api_key=langchain_core.utils.convert_to_secret_str(os.environ["OPEN_API_KEY"]),
#                 temperature=0.7
#             )
#         else:
#             self.model = ChatOllama(base_url="http://localhost:11434", model="gemma3:4b", reasoning=False)

#         # 创建室内场景搭建者
#         self.roomBuilder = GenerateAgent(
#             "RoomBuilder",
#             "roomBuilder",
#             "你是一位经验丰富的室内场景设计师，正在参与一场3D场景环境构建的工作。"
#             "你的职责是根据接受到的场景环境提示词，给出你认为符合该室内场景描述的家具和生活用品，如1.物体(物体描述)(物体大小如长宽高)。"
#         )

#         # 创建场景搭建者
#         self.builder = GenerateAgent(
#             "Builder",
#             "builder",
#             "你是一位经验丰富的室内场景装修师，正在参与一场3D场景环境构建的工作。"
#             "你的职责是根据接受到的家具及生活用品的内容，给出其相应的位置和大小，中心点为(0,0,0),不要带单位, 以表格的形式呈现出来，如|物体|(x坐标,y坐标,z坐标)|(长,宽,高)|物体描述|"
#             "且在该场景中越重要，越必须存在的物体排在前面，如在卧室中床排在书架前面, 并给出依据"
#         )

#         # 创建物体列表生成者（新增）
#         self.objectLister = GenerateAgent(
#             "ObjectLister",
#             "objectLister",
#             "你是一位经验丰富的室内场景设计师，正在参与一场3D场景环境构建的工作。"
#             "你的职责是根据接受到的场景环境提示词，只列出你认为符合该室内场景描述的家具和生活用品的名称。"
#             "请直接返回物体名称列表，用逗号分隔，不要包含位置、尺寸等其他信息。"
#             "例如：床,床头柜,书桌,椅子,衣柜,台灯"
#         )

#         # 创建输入提示扩展者
#         self.extender = GenerateAgent(
#             "Extender",
#             "extender",
#             "你是一位经验丰富的续写者，正在参与一场3D场景环境构建的工作。"
#             "你的职责是根据接受到的场景环境提示词，联网搜索相关的环境描述，书籍里的环境描写或类似景点的旅游介绍，并进行适当的补充扩写和合理的想象联想，要带有物体的相对位置，如a在b的脚下。"
#             "不少于300字"
#         )

#         # 创建自然景观构建者
#         self.naturalEnvBuilder = GenerateAgent(
#             "NaturalEnvBuilder",
#             "builder",
#             "你是一位经验丰富的自然景观搭建者，正在参与一场3D场景环境构建的工作。"
#             "你的职责是根据接受到的场景环境内容，给出你认为符合该场景的自然景观、动植物和大自然里的物体,如1.物体(物体的描述)(物体大小如长宽高)。"
#         )

#         # 创建人文景观构建者
#         self.humanEnvBuilder = GenerateAgent(
#             "HumanEnvBuilder",
#             "builder",
#             "你是一位经验丰富的人文景观搭建者，正在参与一场3D场景环境构建的工作。"
#             "你的职责是根据接受到的场景环境内容，给出你认为符合该场景的人文景观、各式各样的建筑物和其他相关的人造物，如1.物体(物体描述)(物体大小如长宽高)。"
#         )

#         # 创建景观拼接者
#         self.splicer = GenerateAgent(
#             "Splicer",
#             "splicer",
#             "你是一位经验丰富的景观拼接者，正在参与一场3D场景环境构建的工作。"
#             "你的职责是根据接受到的自然和人文景观内容，进行景观的拼接，给出景观的相应位置和大小，中心点为(0,0,0),不要带单位, 以表格的形式呈现出来，如|物体|(x坐标,y坐标,z坐标)|(长,宽,高)|物体描述|。"
#             "说明拼接的理由"
#         )

#         # 创建监督者
#         self.supervisor = GenerateAgent(
#             "Supervisor",
#             "supervisor",
#             "你是一位经验丰富的监督者，正在参与一场3D场景环境构建的工作。"
#             "你的工作是判断每一个物体是否都符合提示词的要求,如左上角的物体应该为|物体|(x坐标,y坐标,z坐标)|(长,宽,高)|物体描述|"
#             "如果符合输出Yes，否则如果是输出的内容不符则输出extender，并说明如何调整。"
#         )

#     def get_object_list_only(self):
#         """只返回物体名称列表，不包含位置信息"""
#         print(f"\n=== 开始生成物体列表 ===")
#         print(f"场景描述: {self.env}")

#         # 直接使用objectLister生成物体列表
#         context = self.objectLister.get_context([{
#             "speaker": "System",
#             "content": f"请为以下场景生成合适的家具和物品列表: {self.env}"
#         }], AgentType.builder)

#         response = self.model.invoke([HumanMessage(content=context)])
#         object_list_str = response.content.strip()

#         print(f"ObjectLister 原始响应: {object_list_str}")

#         # 解析响应，提取物体名称
#         objects = self._parse_object_list(object_list_str)

#         print(f"解析后的物体列表: {objects}")
#         print(f"共生成 {len(objects)} 个物体")

#         return objects

#     def _parse_object_list(self, response_text: str) -> List[str]:
#         """解析模型响应，提取物体名称列表"""
#         # 去除可能的标点符号和数字编号
#         import re

#         # 去除数字编号和括号
#         cleaned_text = re.sub(r'\d+\.\s*', '', response_text)
#         cleaned_text = re.sub(r'[\(\)（）]', '', cleaned_text)

#         # 按逗号、顿号、换行符分割
#         objects = re.split(r'[，,、\n]', cleaned_text)

#         # 清理每个物体名称
#         cleaned_objects = []
#         for obj in objects:
#             obj = obj.strip()
#             # 去除可能的大小描述等额外信息
#             obj = re.sub(r'大小.*|尺寸.*|长.*|宽.*|高.*', '', obj).strip()
#             # 只保留主要物体名称
#             if obj and len(obj) > 0:
#                 # 如果包含描述，只取第一个词作为物体类型
#                 if ' ' in obj or '：' in obj or ':' in obj:
#                     obj = obj.split()[0] if ' ' in obj else obj.split('：')[0].split(':')[0]
#                 cleaned_objects.append(obj)

#         # 去重
#         unique_objects = list(dict.fromkeys(cleaned_objects))

#         return unique_objects

#     # 原有的节点函数保持不变
#     def judger_node(self, state: AgentState) -> Dict:
#         """判断者节点处理函数"""
#         context = self.judger.get_context(state["input"],AgentType.judger)
#         response = self.model.invoke([HumanMessage(content=context)])

#         print(f"\n{self.judger.name}: {response.content}")

#         if "A" in str(response.content):
#             decide = "roomBuilder"
#             state["terminated"].append(1)
#         elif "B" in str(response.content):
#             decide = "extender"
#             state["terminated"].append(2)
#         else:
#             raise ValueError("Judger无法判断场景类型，请确保回答中包含A或B。")
#         print(f"Decide way with terminated state: {state['terminated']}")

#         return {"state": state, "next": decide}

#     def decide_way(self, state: AgentState):
#         print(f"Decide way with terminated state: {state['terminated']}")
#         if state["terminated"][-1] == 1:
#             return "roomBuilder"
#         elif state["terminated"][-1] == 2:
#             return "extender"
#         else:
#             raise ValueError("Judger无法判断场景类型，请确保回答中包含A或B。")

#     def roombuilder_node(self, state: AgentState) -> Dict:
#         """室内设计师节点处理函数"""
#         context = self.roomBuilder.get_context(state["input"],AgentType.roomBuilder)
#         response = self.model.invoke([HumanMessage(content=context)])

#         message = {
#             "speaker": self.roomBuilder.name,
#             "content": response.content
#         }

#         state["input"].append(message)
#         self.roomBuilder.update_memory(message)
#         print(f"\n{self.roomBuilder.name}: {response.content}")

#         return {"state": state, "next": "Builder"}

#     def builder_node(self, state: AgentState) -> Dict:
#         """室内装修师节点处理函数"""
#         context = self.builder.get_context(state["input"],AgentType.builder)
#         response = self.model.invoke([HumanMessage(content=context)])

#         message = {
#             "speaker": self.builder.name,
#             "content": response.content
#         }

#         state["input"].append(message)
#         self.builder.update_memory(message)
#         print(f"\n{self.builder.name}: {response.content}")

#         self.output = "|" + state["input"][-1]["content"].rsplit("|", 1)[0].split("|", 1)[1]

#         return {"state": state, "next": END}

#     def extender_node(self, state: AgentState) -> Dict:
#         """扩展者节点处理函数"""
#         context = self.extender.get_context(state["input"],AgentType.extender)
#         if self.is_api:
#             agent = create_agent(tools=search_tool_web, model=self.model)
#             inputs = {"messages": [{"role": "user", "content": context}]}
#             content = ""
#             for chunk in agent.stream(inputs, stream_mode="updates"):
#                 print(chunk[list(chunk.keys())[0]]['messages'][0].content)
#                 content = chunk[list(chunk.keys())[0]]['messages'][0].content
#         else:
#             content = self.model.invoke([HumanMessage(content=context)]).content

#         message = {
#             "speaker": self.extender.name,
#             "content": content
#         }

#         state["input"].append(message)
#         self.extender.update_memory(message)
#         print(f"\n{self.extender.name}: {content}")

#         return {"state": state, "next": "naturalEnvBuilder, humanEnvBuilder"}

#     def naturalistically_node(self, state: AgentState) -> Dict:
#         """自然景观构建节点处理函数"""
#         context = self.naturalEnvBuilder.get_context(state["input"],AgentType.naturalEnvBuilder)
#         response = self.model.invoke([HumanMessage(content=context)])

#         message = {
#             "speaker": self.naturalEnvBuilder.name,
#             "content": response.content
#         }

#         state["input"].append(message)
#         self.naturalEnvBuilder.update_memory(message)
#         print(f"\n{self.naturalEnvBuilder.name}: {response.content}")

#         return {"state": state, "next": "splicer"}

#     def humanlistically_node(self, state: AgentState) -> Dict:
#         """人文景观构建节点处理函数"""
#         context = self.humanEnvBuilder.get_context(state["input"],AgentType.humanEnvBuilder)
#         response = self.model.invoke([HumanMessage(content=context)])

#         message = {
#             "speaker": self.humanEnvBuilder.name,
#             "content": response.content
#         }

#         state["input"].append(message)
#         self.humanEnvBuilder.update_memory(message)
#         print(f"\n{self.humanEnvBuilder.name}: {response.content}")

#         return {"state": state, "next": "splicer"}

#     def splicer_node(self, state: AgentState) -> Dict:
#         """拼接节点处理函数"""
#         context = self.splicer.get_context(state["input"],AgentType.splicer)
#         response = self.model.invoke([HumanMessage(content=context)])

#         message = {
#             "speaker": self.splicer.name,
#             "content": response.content
#         }

#         state["input"].append(message)
#         self.splicer.update_memory(message)
#         print(f"\n{self.splicer.name}: {response.content}")

#         return {"state": state, "next": self.supervisor}

#     def supervisor_node(self, state: AgentState) -> Dict:
#         """拼接节点处理函数"""
#         context = self.supervisor.get_context(state["input"],AgentType.supervisor)
#         response = self.model.invoke([HumanMessage(content=context)])

#         message = {
#             "speaker": self.supervisor.name,
#             "content": response.content
#         }
#         print(f"\n{self.supervisor.name}: {response.content}")

#         decide = ""
#         if "Yes" in str(response.content):
#             state["terminated"].append(1)
#         elif "extender" in str(response.content):
#             decide = "extender"
#             state["terminated"].append(2)
#         else:
#             raise ValueError("Supervisor无法判断下一步路径，请确保回答中包含Yes, extender或splicer。")

#         if state["terminated"][-1] == 1:
#             self.output= "|" + state["input"][-1]["content"].rsplit("|",1)[0].split("|",1)[1]
#             return {"state": state, "next": END}
#         else:
#             state["input"].append(message)
#             return {"state": state,"next": decide}

#     def decide_path(self, state: AgentState):
#         if state["terminated"][-1] == 1:
#             return "END"
#         elif state["terminated"][-1] == 2:
#             return "extender"
#         else:
#             raise ValueError("Supervisor无法判断下一步路径，请确保回答中包含Yes, extender或splicer。")

#     # 构建图
#     def build_graph(self):
#         """构建3D环境搭建流程图"""
#         self.workflow = StateGraph(AgentState)

#         # 添加节点
#         self.workflow.add_node("judger", self.judger_node)
#         self.workflow.add_node("roomBuilder", self.roombuilder_node)
#         self.workflow.add_node("builder", self.builder_node)
#         self.workflow.add_node("extender", self.extender_node)
#         self.workflow.add_node("naturalEnvBuilder", self.naturalistically_node)
#         self.workflow.add_node("humanEnvBuilder",  self.humanlistically_node)
#         self.workflow.add_node("splicer", self.splicer_node)
#         self.workflow.add_node("supervisor", self.supervisor_node)

#         # 设置顺序边
#         # judger -> roomBuilder -> builder -> END 室内场景
#         #        -> extender -> naturalEnvBuilder -> splicer -> supervisor -> decide_path -> splicer 室外场景
#         #                    -> humanEnvBuilder   ->                                      -> END
#         #                                                                                 -> extender
#         self.workflow.add_edge(START, "judger")
#         self.workflow.add_conditional_edges("judger",
#                                     self.decide_way,
#                                     {"roomBuilder": "roomBuilder", "extender": "extender"})
#         self.workflow.add_edge("roomBuilder", "builder")
#         self.workflow.add_edge("builder", END)
#         self.workflow.add_edge("extender", "naturalEnvBuilder")
#         self.workflow.add_edge("extender", "humanEnvBuilder")
#         self.workflow.add_edge("naturalEnvBuilder", "splicer")
#         self.workflow.add_edge("humanEnvBuilder", "splicer")
#         self.workflow.add_edge("splicer", "supervisor")
#         self.workflow.add_conditional_edges("supervisor",
#                                     self.decide_path,
#                                     {"extender": "extender", "END": END})

#         self.graph = self.workflow.compile()

#     def stuff_generate(self):
#         print("\n=== 开始构建图 ===")
#         self.build_graph()
#         print("图构建完成")

#         # 初始化状态
#         initial_state = AgentState(
#             input = [{
#                "speaker": "System",
#                "content": f"让我们开始关于场景的构建,{self.env}。"
#             }],
#             terminated =  []
#         )
#         print("\n=== 初始状态已设置 ===")
#         print("系统: 让我们开始关于场景的构建。\n")

#         # 运行图
#         print("=== 开始构建场景 ===")
#         for _ in self.graph.stream(initial_state):
#             print("-" * 50)

#         return self.output.replace("**", "").split("|\n")[2:]  # 去除markdown格式的**加粗**，转换为列表

from typing import List, Dict, TypedDict
from enum import Enum
import os
import warnings
import langchain_core.utils
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langchain_ollama import ChatOllama

warnings.filterwarnings("ignore")

# 配置Tavily的API
os.environ["TAVILY_API_KEY"] = "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"

# 配置在线大模型的API
os.environ["OPEN_API_KEY"] = "sk-cf5ccda9343c41dc999d8252a287512e"
os.environ["OPEN_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 定义搜索工具
search = TavilySearch(
    max_results=5,
    include_answer=True,
    include_raw_content=False,
    include_images=False,
)


@tool
def search_web(query: str) -> str:
    """Web search"""
    result = search.invoke({"query": query})
    return result["answer"]


# 创建工具实例
search_tool_web = [search_web]


# 定义状态类型
class AgentState(TypedDict):
    input: List[Dict[str, str]]  # History messages
    terminated: List[int]  # Supervisor return status, such as extender, splicer and END


# 定义节点类型
class AgentType(Enum):
    judger = 0
    extender = 1
    naturalEnvBuilder = 2
    humanEnvBuilder = 3
    splicer = 4
    supervisor = 5
    roomBuilder = 6
    builder = 7


# 定义Agent的基础类
class GenerateAgent:
    def __init__(self, name: str, role: str, system_prompt: str):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.memory: List[Dict] = []

    def update_memory(self, message: Dict):
        self.memory.append(message)

    def get_context(self, messages: List[Dict], AgentType) -> str:
        memory_str = "\n".join([f"{m['speaker']}: {m['content']}" for m in self.memory[-5:]])  # Medium-term memory
        if AgentType == AgentType.splicer:
            temp_message = "\n".join([f"{m['content']}" for m in messages[-2:]])  # Medium-term memory
            memory_str = ""
        elif AgentType == AgentType.supervisor or AgentType == AgentType.builder:
            temp_message = "\n".join([f"{messages[0]['content']}", f"{messages[-1]['content']}"])  # Long-term memory
            memory_str = ""  # No medium-term memory
        elif AgentType == AgentType.extender:
            # Long-term memory
            temp_message = messages[0]["content"] if len(messages) == 1 else "\n".join(
                [f"{messages[0]['content']}", f"{messages[-1]['content']}"])
        else:
            # Short-term memory
            temp_message = messages[-1]["content"]
        return f"""
{self.system_prompt}
History dialogue:
{temp_message}
{memory_str}
Now it's your turn to create. Please continue working based on the history dialogue and your role.
"""


class chat_with_model:
    def __init__(self, question, is_api: bool = False):
        self.workflow = None
        self.env = question
        self.output = ""
        # Create agent for judging indoor or outdoor scenes
        self.judger = GenerateAgent(
            "Judger",
            "judger",
            "You are an experienced 3D scene builder participating in a 3D scene environment construction project. "
            "Your responsibility is to determine whether the received scene environment prompt is an indoor scene or an outdoor scene. "
            "If it's an indoor scene, output A; if it's an outdoor scene, output B. "
            "Please respond in English only."
        )
        self.is_api = is_api
        if is_api == 1:
            self.model = ChatOpenAI(
                model="qwen-plus",
                base_url=os.environ["OPEN_URL"],
                api_key=langchain_core.utils.convert_to_secret_str(os.environ["OPEN_API_KEY"]),
                temperature=0.7
            )
        else:
            self.model = ChatOllama(base_url="http://localhost:11434", model="gemma3:4b", reasoning=False)

        # Create indoor scene builder
        self.roomBuilder = GenerateAgent(
            "RoomBuilder",
            "roomBuilder",
            "You are an experienced indoor scene designer participating in a 3D scene environment construction project. "
            "Your responsibility is to provide furniture and household items that match the indoor scene description, "
            "such as 1. Object (object description) (object size like length, width, height). "
            "Please respond in English only."
        )

        # Create scene builder
        self.builder = GenerateAgent(
            "Builder",
            "builder",
            "You are an experienced indoor scene decorator participating in a 3D scene environment construction project. "
            "Your responsibility is to provide the corresponding positions and sizes for furniture and household items, "
            "with the center point at (0,0,0), without units, presented in table format, such as |Object|(x,y,z)|(length,width,height)|Object description|. "
            "More important and essential objects in the scene should be listed first, such as bed before bookshelf in a bedroom, and provide reasoning. "
            "Please respond in English only."
        )

        # Create object list generator (new)
        self.objectLister = GenerateAgent(
            "ObjectLister",
            "objectLister",
            "You are an experienced indoor scene designer participating in a 3D scene environment construction project. "
            "Your responsibility is to list only the names of furniture and household items that you think match the indoor scene description. "
            "Please directly return a list of object names, separated by commas, without including other information like position or size. "
            "Example: bed, nightstand, desk, chair, wardrobe, lamp "
            "Please respond in English only."
        )

        # Create input prompt extender
        self.extender = GenerateAgent(
            "Extender",
            "extender",
            "You are an experienced content extender participating in a 3D scene environment construction project. "
            "Your responsibility is to search the web for relevant environment descriptions, environmental depictions from books, or introductions to similar attractions, "
            "and appropriately supplement and expand with reasonable imagination, including relative positions of objects, such as a is at the foot of b. "
            "At least 300 words. "
            "Please respond in English only."
        )

        # Create natural landscape builder
        self.naturalEnvBuilder = GenerateAgent(
            "NaturalEnvBuilder",
            "builder",
            "You are an experienced natural landscape builder participating in a 3D scene environment construction project. "
            "Your responsibility is to provide natural landscapes, plants, animals, and natural objects that match the scene, "
            "such as 1. Object (object description) (object size like length, width, height). "
            "Please respond in English only."
        )

        # Create human landscape builder
        self.humanEnvBuilder = GenerateAgent(
            "HumanEnvBuilder",
            "builder",
            "You are an experienced human landscape builder participating in a 3D scene environment construction project. "
            "Your responsibility is to provide human landscapes, various buildings, and other related man-made objects that match the scene, "
            "such as 1. Object (object description) (object size like length, width, height). "
            "Please respond in English only."
        )

        # Create landscape splicer
        self.splicer = GenerateAgent(
            "Splicer",
            "splicer",
            "You are an experienced landscape splicer participating in a 3D scene environment construction project. "
            "Your responsibility is to splice the landscapes based on the received natural and human landscape content, "
            "providing the corresponding positions and sizes of the landscapes, with the center point at (0,0,0), without units, "
            "presented in table format, such as |Object|(x,y,z)|(length,width,height)|Object description|. "
            "Explain the splicing reasoning. "
            "Please respond in English only."
        )

        # Create supervisor
        self.supervisor = GenerateAgent(
            "Supervisor",
            "supervisor",
            "You are an experienced supervisor participating in a 3D scene environment construction project. "
            "Your job is to judge whether each object meets the requirements of the prompt, such as whether the object in the upper left corner should be |Object|(x,y,z)|(length,width,height)|Object description|. "
            "If it meets the requirements, output Yes; otherwise, if the output content does not meet the requirements, output extender and explain how to adjust. "
            "Please respond in English only."
        )

    '''
    def get_object_list_only(self):
        """Return only object name list, without position information"""
        print(f"\n=== Starting object list generation ===")
        print(f"Scene description: {self.env}")
        
        # Directly use objectLister to generate object list
        context = self.objectLister.get_context([{
            "speaker": "System",
            "content": f"Please generate appropriate furniture and item list for the following scene: {self.env}"
        }], AgentType.builder)
        
        response = self.model.invoke([HumanMessage(content=context)])
        object_list_str = response.content.strip()
        
        print(f"ObjectLister raw response: {object_list_str}")
        
        # Parse response, extract object names
        objects = self._parse_object_list(object_list_str)
        
        print(f"Parsed object list: {objects}")
        print(f"Generated {len(objects)} objects")
        
        return objects
    
    def _parse_object_list(self, response_text: str) -> List[str]:
        """Parse model response, extract object name list"""
        # Remove possible punctuation and number labels
        import re
        
        # Remove number labels and brackets
        cleaned_text = re.sub(r'\d+\.\s*', '', response_text)
        cleaned_text = re.sub(r'[\(\)（）]', '', cleaned_text)
        
        # Split by comma, Chinese comma, newline
        objects = re.split(r'[，,、\n]', cleaned_text)
        
        # Clean each object name
        cleaned_objects = []
        for obj in objects:
            obj = obj.strip()
            # Remove possible size descriptions and other extra information
            obj = re.sub(r'size.*|dimension.*|length.*|width.*|height.*', '', obj).strip()
            # Keep only main object name
            if obj and len(obj) > 0:
                # If contains description, only take the first word as object type
                if ' ' in obj or '：' in obj or ':' in obj:
                    obj = obj.split()[0] if ' ' in obj else obj.split('：')[0].split(':')[0]
                cleaned_objects.append(obj)
        
        # Remove duplicates
        unique_objects = list(dict.fromkeys(cleaned_objects))
        
        return unique_objects
        '''

    def get_object_list_only(self):
        """Return only object name list, without position information"""
        print(f"\n=== Starting object list generation ===")
        print(f"Scene description: {self.env}")

        # Directly use objectLister to generate object list
        context = self.objectLister.get_context([{
            "speaker": "System",
            "content": f"Please generate appropriate furniture and item list for the following scene: {self.env}"
        }], AgentType.builder)

        response = self.model.invoke([HumanMessage(content=context)])
        object_list_str = response.content.strip()

        print(f"ObjectLister raw response: {object_list_str}")

        # 直接按逗号分割，不做过多处理
        import re
        objects = [obj.strip() for obj in re.split(r'[，,]', object_list_str)]
        objects = [obj for obj in objects if obj and len(obj) > 0]

        # 去重
        unique_objects = list(dict.fromkeys(objects))

        print(f"Final object list: {unique_objects}")
        print(f"Generated {len(unique_objects)} objects")

        return unique_objects

    # CHANGED HEREEEEEEEEEEEEEE

    # Keep original node functions but update print messages to English
    def judger_node(self, state: AgentState) -> Dict:
        """Judger node processing function"""
        context = self.judger.get_context(state["input"], AgentType.judger)
        response = self.model.invoke([HumanMessage(content=context)])

        print(f"\n{self.judger.name}: {response.content}")

        if "A" in str(response.content):
            decide = "roomBuilder"
            state["terminated"].append(1)
        elif "B" in str(response.content):
            decide = "extender"
            state["terminated"].append(2)
        else:
            raise ValueError("Judger cannot determine scene type, please ensure the response contains A or B.")
        print(f"Decide way with terminated state: {state['terminated']}")

        return {"state": state, "next": decide}

    def decide_way(self, state: AgentState):
        print(f"Decide way with terminated state: {state['terminated']}")
        if state["terminated"][-1] == 1:
            return "roomBuilder"
        elif state["terminated"][-1] == 2:
            return "extender"
        else:
            raise ValueError("Judger cannot determine scene type, please ensure the response contains A or B.")

    def roombuilder_node(self, state: AgentState) -> Dict:
        """Indoor designer node processing function"""
        context = self.roomBuilder.get_context(state["input"], AgentType.roomBuilder)
        response = self.model.invoke([HumanMessage(content=context)])

        message = {
            "speaker": self.roomBuilder.name,
            "content": response.content
        }

        state["input"].append(message)
        self.roomBuilder.update_memory(message)
        print(f"\n{self.roomBuilder.name}: {response.content}")

        return {"state": state, "next": "Builder"}

    def builder_node(self, state: AgentState) -> Dict:
        """Indoor decorator node processing function"""
        context = self.builder.get_context(state["input"], AgentType.builder)
        response = self.model.invoke([HumanMessage(content=context)])

        message = {
            "speaker": self.builder.name,
            "content": response.content
        }

        state["input"].append(message)
        self.builder.update_memory(message)
        print(f"\n{self.builder.name}: {response.content}")

        self.output = "|" + state["input"][-1]["content"].rsplit("|", 1)[0].split("|", 1)[1]

        return {"state": state, "next": END}

    def extender_node(self, state: AgentState) -> Dict:
        """Extender node processing function"""
        context = self.extender.get_context(state["input"], AgentType.extender)
        if self.is_api:
            agent = create_agent(tools=search_tool_web, model=self.model)
            inputs = {"messages": [{"role": "user", "content": context}]}
            content = ""
            for chunk in agent.stream(inputs, stream_mode="updates"):
                print(chunk[list(chunk.keys())[0]]['messages'][0].content)
                content = chunk[list(chunk.keys())[0]]['messages'][0].content
        else:
            content = self.model.invoke([HumanMessage(content=context)]).content

        message = {
            "speaker": self.extender.name,
            "content": content
        }

        state["input"].append(message)
        self.extender.update_memory(message)
        print(f"\n{self.extender.name}: {content}")

        return {"state": state, "next": "naturalEnvBuilder, humanEnvBuilder"}

    def naturalistically_node(self, state: AgentState) -> Dict:
        """Natural landscape building node processing function"""
        context = self.naturalEnvBuilder.get_context(state["input"], AgentType.naturalEnvBuilder)
        response = self.model.invoke([HumanMessage(content=context)])

        message = {
            "speaker": self.naturalEnvBuilder.name,
            "content": response.content
        }

        state["input"].append(message)
        self.naturalEnvBuilder.update_memory(message)
        print(f"\n{self.naturalEnvBuilder.name}: {response.content}")

        return {"state": state, "next": "splicer"}

    def humanlistically_node(self, state: AgentState) -> Dict:
        """Human landscape building node processing function"""
        context = self.humanEnvBuilder.get_context(state["input"], AgentType.humanEnvBuilder)
        response = self.model.invoke([HumanMessage(content=context)])

        message = {
            "speaker": self.humanEnvBuilder.name,
            "content": response.content
        }

        state["input"].append(message)
        self.humanEnvBuilder.update_memory(message)
        print(f"\n{self.humanEnvBuilder.name}: {response.content}")

        return {"state": state, "next": "splicer"}

    def splicer_node(self, state: AgentState) -> Dict:
        """Splicer node processing function"""
        context = self.splicer.get_context(state["input"], AgentType.splicer)
        response = self.model.invoke([HumanMessage(content=context)])

        message = {
            "speaker": self.splicer.name,
            "content": response.content
        }

        state["input"].append(message)
        self.splicer.update_memory(message)
        print(f"\n{self.splicer.name}: {response.content}")

        return {"state": state, "next": "supervisor"}

    def supervisor_node(self, state: AgentState) -> Dict:
        """Supervisor node processing function"""
        context = self.supervisor.get_context(state["input"], AgentType.supervisor)
        response = self.model.invoke([HumanMessage(content=context)])

        message = {
            "speaker": self.supervisor.name,
            "content": response.content
        }
        print(f"\n{self.supervisor.name}: {response.content}")

        decide = ""
        if "Yes" in str(response.content):
            state["terminated"].append(1)
        elif "extender" in str(response.content):
            decide = "extender"
            state["terminated"].append(2)
        else:
            raise ValueError(
                "Supervisor cannot determine next path, please ensure the response contains Yes, extender or splicer.")

        if state["terminated"][-1] == 1:
            self.output = "|" + state["input"][-1]["content"].rsplit("|", 1)[0].split("|", 1)[1]
            return {"state": state, "next": END}
        else:
            state["input"].append(message)
            return {"state": state, "next": decide}

    def decide_path(self, state: AgentState):
        if state["terminated"][-1] == 1:
            return "END"
        elif state["terminated"][-1] == 2:
            return "extender"
        else:
            raise ValueError(
                "Supervisor cannot determine next path, please ensure the response contains Yes, extender or splicer.")

    # Build graph
    def build_graph(self):
        """Build 3D environment construction flow graph"""
        self.workflow = StateGraph(AgentState)

        # Add nodes
        self.workflow.add_node("judger", self.judger_node)
        self.workflow.add_node("roomBuilder", self.roombuilder_node)
        self.workflow.add_node("builder", self.builder_node)
        self.workflow.add_node("extender", self.extender_node)
        self.workflow.add_node("naturalEnvBuilder", self.naturalistically_node)
        self.workflow.add_node("humanEnvBuilder", self.humanlistically_node)
        self.workflow.add_node("splicer", self.splicer_node)
        self.workflow.add_node("supervisor", self.supervisor_node)

        # Set sequential edges
        # judger -> roomBuilder -> builder -> END indoor scene
        #        -> extender -> naturalEnvBuilder -> splicer -> supervisor -> decide_path -> splicer outdoor scene
        #                    -> humanEnvBuilder   ->                                      -> END
        #                                                                                 -> extender
        self.workflow.add_edge(START, "judger")
        self.workflow.add_conditional_edges("judger",
                                            self.decide_way,
                                            {"roomBuilder": "roomBuilder", "extender": "extender"})
        self.workflow.add_edge("roomBuilder", "builder")
        self.workflow.add_edge("builder", END)
        self.workflow.add_edge("extender", "naturalEnvBuilder")
        self.workflow.add_edge("extender", "humanEnvBuilder")
        self.workflow.add_edge("naturalEnvBuilder", "splicer")
        self.workflow.add_edge("humanEnvBuilder", "splicer")
        self.workflow.add_edge("splicer", "supervisor")
        self.workflow.add_conditional_edges("supervisor",
                                            self.decide_path,
                                            {"extender": "extender", "END": END})

        self.graph = self.workflow.compile()

    def stuff_generate(self):
        print("\n=== Starting graph construction ===")
        self.build_graph()
        print("Graph construction completed")

        # Initialize state
        initial_state = AgentState(
            input=[{
                "speaker": "System",
                "content": f"Let's start building the scene, {self.env}."
            }],
            terminated=[]
        )
        print("\n=== Initial state set ===")
        print("System: Let's start building the scene.\n")

        # Run graph
        print("=== Starting scene construction ===")
        for _ in self.graph.stream(initial_state):
            print("-" * 50)

        return self.output.replace("**", "").split("|\n")[2:]  # Remove markdown format **bold**, convert to list
