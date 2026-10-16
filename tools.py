from __future__ import annotations

import ast
import operator
import requests

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from config import logger


@tool(description="当需要最新的、实时的或本地知识库没有的信息时，联网搜索。")
def web_search(query: str) -> str:
   
    try:
        resp = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            timeout=8,
        )
        data = resp.json()
    except Exception as e:
        logger.warning("web_search 失败: %s", e)
        return f"[网络搜索暂不可用] 无法检索“{query}”。"

    parts = []
    if data.get("AbstractText"):
        parts.append(data["AbstractText"])
    for topic in (data.get("RelatedTopics") or [])[:3]:
        if isinstance(topic, dict) and topic.get("Text"):
            parts.append(topic["Text"])
    if not parts:
        return f"未检索到关于“{query}”的明确结果。"
    return "\n".join(f"- {p}" for p in parts[:3])


_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_math(node):
    if isinstance(node, ast.Expression):
        return _eval_math(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval_math(node.left), _eval_math(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNOPS:
        return _UNOPS[type(node.op)](_eval_math(node.operand))
    raise ValueError("仅支持四则运算、幂与取模")


@tool(description="计算四则运算等数学表达式（支持 + - * / // % ** 与括号）。")
def calculator(expression: str) -> str:
    """
     计算四则运算等数学表达式（支持 + - * / // % ** 与括号）。

    Args:
        expression: 数学表达式，例如 "(12.5 + 6) * 2"
    """
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_math(tree)
        return f"{expression} = {result}"
    except Exception as e:
        return f"无法计算“{expression}”：{e}"


TOOLS = [web_search, calculator]
TOOL_MAP = {t.name: t for t in TOOLS}


def run_tool_calling_agent(query: str, history: list, llm=None, max_rounds: int = 4) -> str:
    from llm import get_llm
    llm = llm or get_llm()
    llm_with_tools = llm.bind_tools(TOOLS)

    messages = [SystemMessage(content=_AGENT_SYSTEM)]
    messages.extend(history[-10:])
    messages.append(HumanMessage(content=query))

    for _ in range(max_rounds):
        resp = llm_with_tools.invoke(messages)
        messages.append(resp)

        if not getattr(resp, "tool_calls", None):
            return resp.content or ""

        for tc in resp.tool_calls:
            name = tc.get("name")
            args = tc.get("args") or {}
            tool = TOOL_MAP.get(name)
            logger.info("agent tool call: %s(%s)", name, args)
            if tool is None:
                result = f"未知工具: {name}"
            else:
                try:
                    result = tool.invoke(args)
                except Exception as e:
                    result = f"工具执行失败: {e}"
                    logger.exception("tool %s failed", name)
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    return messages[-1].content or ""


_AGENT_SYSTEM = (
    "你是一个智能助手，可以调用工具获取实时信息。"
    "请根据问题自主决定是否调用工具，回答准确、简洁、友好，使用中文。"
)


if __name__ == "__main__":
    print("tool list:", [t.name for t in TOOLS])