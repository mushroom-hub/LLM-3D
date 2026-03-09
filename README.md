
# LLM-3D
## 介绍
LLM-3D建立在Respace的基础上，致力于用简单的描述，快速、准确地搭建3D场景模型，大幅减少人工搭建3D场景所需时间成本。拥有海量3D模型资产库，提供个性化服务，用户可以选择生成符合自己喜好的3D场景风格。


## 核心特性

· 快速生成：基于简单的描述文本，分钟级生成3D场景  

· 精准搭建：自动解析用户意图，智能布置3D模型  

· 海量资产：集成3D-Future资产库，提供丰富的3D模型资源  

· 风格定制：支持个性化风格选择，生成符合用户喜好的3D场景  


## 安装教程
### 环境要求

· Windows系统10/11（需安装WSL）  

· Anaconda/Miniconda  

· Python 3.11及以上  

### 安装步骤
1. 进入到项目目录  
`cd Interface`

2. 用conda创建python为3.11的虚拟环境  
`conda create -n llm3d python=3.11`  
`conda activate llm3d`

3. 安装依赖  
`pip install -r requirements.txt`

4. 启动后端服务  
在wsl中，确保虚拟环境已激活，然后运行：  
`python respace_api.py`

5. 启动前端界面  
另开一个终端（同样在Interface目录下，确保虚拟环境已激活）：  
`python app.py`

6. 选择解析方式  
选择是否使用在线物体解析大模型（Object-Parser LLM）API（Y/N）  
   
7. 访问应用  
在浏览器中打开命令行显示的本地链接（例如：http://127.0.0.1:XXXX），即可开始使用LLM-3D  
 
<img width="440" height="84" alt="image" src="https://github.com/user-attachments/assets/8d35c27c-9e73-422a-8e68-768f90aa2aac" />

## 项目结构

### Interface

Interfce主要包含了前端代码、物体解析大模型（Object-Parser LLM）代码、RAG检索和word2vec词向量相关代码：

```
Interface/
└── code/
    ├── page/                 # 前端页面代码
    ├── EnvGenerator.py       # 物体解析大模型（Object-Parser LLM）代码
    │                         # 用于生成3D场景所需物体列表
    ├── RAG/                  # 增强检索（Retrieval-Augmented Generation）代码
    └── word2vec/             # 词向量相关代码
```
    
### Respace

Respace外接3D-Future资产库，集成了三大核心功能：

· 3D资产检索：从资产库中匹配最合适的3D模型

· 自适应场景布置：智能规划物体的位置和布局

· 自动化验证与调优：自动检查并优化场景合理性

主要文件：

respace_api.py：Respace服务的启动代码


### 参与贡献

1. Fork 本仓库
2. 新建 Feat_xxx 分支
3. 提交代码
4. 新建 Pull Request
