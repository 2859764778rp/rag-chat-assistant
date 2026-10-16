# LangChain RAG 智能问答助手（基于原 chat-bot 重构）

将原 `chat-bot`（手写 RAG + OpenAI SDK 直连）项目全面重构为 **LangChain 组件化**实现，
以满足「基于 LangChain 的 RAG 智能问答应用开发」课程作业要求。
完整覆盖作业要求的六大核心模块：**LLM、VectorStore、Retriever、Chain、Agent、Memory**。

## 架构

```
用户请求 ─▶ Flask(/chat) ─▶ rag_engine.interview_stream
                                 │
              ┌──────────────────┼──────────────────────┐
              ▼                  ▼                       ▼
        RAG 检索链(Chain)   知识库命中判断          无命中回退 Agent
       (单一 knowledge 库)   RAG_MIN_SCORE           web_search 等工具
              │                  │
              ▼                  ▼
     VectorStore + Retriever   Memory(SQLite)
     (numpy 余弦 / as_retriever) RunnableWithMessageHistory
```

> 项目采用**单一知识库检索链**（无意图路由），金融/医疗/法律等领域文档统一入库到
> `knowledge` 向量库；当检索相似度低于 `RAG_MIN_SCORE`（视为知识库未命中）时，
> 回退到 Agent（工具调用，可联网搜索）。

### LangChain 模块对应关系（作业验收映射）

| 作业要求 | 本项目实现 | 文件 |
|---|---|---|
| **LLM**（文本生成/推理） | `ChatOpenAI`（DeepSeek 兼容调用） | `llm.py` |
| **VectorStore**（向量库） | 继承 `langchain_core.vectorstores.VectorStore` 的自定义向量库（numpy 余弦，等价"Chroma 或其他向量库"） | `vectorstore.py` |
| **Embedding**（向量化） | 本地向量化：TF-IDF（可无缝切换 `HuggingFaceEmbeddings` 语义模型） | `embeddings.py` |
| **Retriever**（检索器） | `store.as_retriever(search_kwargs={"k": RAG_TOP_K})` | `vectorstore.py` |
| **Chain**（链式调用） | LCEL：`RunnableParallel(retriever\|format, question) \| prompt \| LLM \| parser` | `rag_engine.py` |
| **Agent**（自主决策） | `llm.bind_tools(TOOLS)` 工具调用循环 | `tools.py` |
| **Memory**（上下文记忆） | `BaseChatMessageHistory`(SQLite) + `RunnableWithMessageHistory` | `database.py` |
| **数据切分** | `RecursiveCharacterTextSplitter` | `ingest.py` |
| **PDF 解析** | `unstructured.partition_pdf`（优先）/ `pypdf`（回退） | `ingest.py` |

## 文件说明

| 文件 | 职责 |
|---|---|
| `config.py` | 全局配置（DeepSeek Key、RAG 参数、路径） |
| `llm.py` | LangChain LLM：`ChatOpenAI`（DeepSeek）、流式输出工具 |
| `embeddings.py` | LangChain `Embeddings` 接口：本地向量化（TF-IDF / 可选语义模型） |
| `vectorstore.py` | LangChain `VectorStore` 接口：add_texts / similarity_search / as_retriever |
| `database.py` | SQLite：LangChain `BaseChatMessageHistory`（Memory 对话历史） |
| `tools.py` | LangChain `@tool` 工具 + Agent 工具调用循环（web_search） |
| `ingest.py` | 数据处理：PDF/TXT/CSV 解析 → 文本切分 → 入库 |
| `rag_engine.py` | 主流程：LCEL 检索链 + Memory + Agent 兜底 |
| `app.py` | Flask 对外接口：`/chat`（流式）、`/vectorstore/update`、`/vectorstore/stats` |
| `init_kb.py` | 将 `documents/` 下的领域知识文档入库到统一知识库 |
| `gradio_interface.py` | 本地调试 UI（可选） |

## 运行步骤

```bash
# 1. 安装依赖（已在本项目 chat-bot.venv 装好）
#    python -m pip install -r requirements.txt

# 2. 初始化向量知识库
python init_kb.py

# （可选）批量导入自有 PDF/TXT/CSV 数据到知识库
#    python ingest.py documents\某文档.txt

# 3. 启动服务
python app.py            # http://127.0.0.1:5000

# 4. 命令行端到端测试
python rag_engine.py

# 5. （可选）Gradio 本地调试 UI
python gradio_interface.py
```

## 配置（.env）

```
DEEPSEEK_API_KEY=sk-xxxx
# 可选：
# CHAT_MODEL=deepseek-chat
# STORE_NAME=knowledge        # 统一知识库名称
# RAG_TOP_K=3
# RAG_MIN_SCORE=0.15          # 相似度低于该值视为知识库未命中，回退 Agent
# CHUNK_SIZE=500
# CHUNK_OVERLAP=50
# EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
```

## 测试问答（当前数据集：金融/医疗/法律）

- 金融："什么是 CPI？它对债券价格有什么影响？"、"存款保险的保障上限是多少？"
- 医疗："流感应该如何预防？"、"高血压的诊断标准是多少？"
- 法律："诉讼时效是多久？"、"签订劳动合同需要注意哪些条款？"
- 多轮追问：先问金融问题，再追问"那对我投资有什么建议？"（验证 Memory）
- 无命中回退：问"今天天气怎么样"（应走 Agent 而非知识库）

## 关于向量化与向量库的说明

- **Embedding**：默认使用本地 TF-IDF 向量化（纯 numpy，无需 torch/GPU/联网），
  已实现 LangChain `Embeddings` 接口。若安装 `sentence-transformers` + `langchain-huggingface`，
  会自动切换为 `HuggingFaceEmbeddings` 语义模型，无需改代码。
- **VectorStore**：当前为自定义 numpy 向量库（实现 LangChain `VectorStore` 接口，对应作业
  "Chroma 或其他向量库"）。若需切换 Chroma，安装 `langchain-chroma` 后把
  `vectorstore.py` 的 `LocalVectorStore` 替换为 `Chroma.from_documents(...)` 即可，
  上游 Retriever / Chain 无需改动。

## 安全提示

对外接口默认未启用鉴权，仅供本地/内网演示。生产部署需增加 API Key/Token 鉴权、
限流与 HTTPS，并将 Flask 换成生产级 WSGI（gunicorn/uwsgi）。
