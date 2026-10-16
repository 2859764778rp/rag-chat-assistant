from __future__ import annotations

from typing import Iterator

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory

from config import STORE_NAME, RAG_TOP_K, RAG_MIN_SCORE, logger
from database import SQLiteChatHistory, connect_to_database, get_session_history
from llm import get_llm
from tools import run_tool_calling_agent
from vectorstore import get_store

INITIAL_GREETING = (
    "您好，我是基于知识库的智能问答助手 🤖。我可以回答金融、医疗、法律等领域的知识问题，"
    "也能帮您解答其他问题。请问有什么可以帮到您？"
)

_BASE_SYSTEM = (
    "你是一个专业的智能问答助手。请严格依据给定的知识库上下文回答用户问题，"
    "答案要准确、简洁、友好，使用中文。若知识库中找不到答案，请如实说明，"
    "不要编造信息。"
)


def _format_docs(docs) -> str:
    return "\n\n".join(f"[知识{i+1}] {d.page_content}" for i, d in enumerate(docs))


def _build_rag_chain():
    store = get_store(STORE_NAME)
    retriever = store.as_retriever(search_kwargs={"k": RAG_TOP_K})
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", _BASE_SYSTEM + "\n\n以下是知识库检索到的相关内容，请优先据此回答：\n{context}"),
        ("placeholder", "{history}"),
        ("human", "{question}"),
    ])

    context_chain = (RunnableLambda(lambda x: x["question"]) | retriever | _format_docs)

    rag_chain = (
        RunnableParallel({"context": context_chain, "question": RunnablePassthrough()})
        | prompt
        | llm
        | StrOutputParser()
    )

    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )


_rag_chain = None


def _get_rag_chain():
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = _build_rag_chain()
    return _rag_chain


def _chat_via_rag(query: str, session_id: str) -> Iterator[str]:
    chain = _get_rag_chain()
    config = {"configurable": {"session_id": session_id}}
    for chunk in chain.stream({"question": query}, config=config):
        if chunk:
            yield chunk


def _has_knowledge(query: str) -> bool:
    res = get_store(STORE_NAME)._similarity_search_with_score(query, k=1)
    return bool(res) and res[0][1] >= RAG_MIN_SCORE


def _handle_other(query: str, history_messages: list) -> str:
    return run_tool_calling_agent(query, history_messages)


def interview_stream(phone_number: str, query: str) -> Iterator[str]:
    connect_to_database()
    history = SQLiteChatHistory(phone_number)

    if not history.messages:
        from langchain_core.messages import AIMessage, HumanMessage
        history.add_message(HumanMessage(content=query))
        history.add_message(AIMessage(content=INITIAL_GREETING))
        yield INITIAL_GREETING
        return

    if _has_knowledge(query):
        for delta in _chat_via_rag(query, phone_number):
            yield delta
    else:
        yield _handle_other(query, history.messages)


if __name__ == "__main__":
    phone = "19836701050"
    print("A:", "".join(interview_stream(phone, "你好")))
    print("B(金融):", end=" ")
    for d in interview_stream(phone, "什么是 CPI？它对债券价格有什么影响？"):
        print(d, end="", flush=True)
    print()
    print("C(医疗):", end=" ")
    for d in interview_stream(phone, "流感应该如何预防？"):
        print(d, end="", flush=True)
    print()
