import gradio as gr

from rag_engine import interview_stream


def respond(message, history, session):
    if session is None:
        session = "gradio_local_user"

    message = (message or "").strip()
    if not message:
        yield history, session, ""
        return

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""},
    ]
    acc = ""
    for delta in interview_stream(session, message):
        acc += delta
        history[-1]["content"] = acc
        yield history, session, ""


def reset():
    return [], "gradio_local_user", ""


with gr.Blocks(title="智能助手（LangChain RAG）") as demo:
    session = gr.State(value=None)

    gr.Markdown()

    chatbot = gr.Chatbot(
        label="对话",
        height=460,
        avatar_images=(None, "🤖"),
        placeholder="请向我提问（例如：什么是 CPI？流感如何预防？）",
    )

    with gr.Row():
        msg = gr.Textbox(
            placeholder="输入消息，按回车发送 ...",
            label="你的消息",
            lines=2,
            scale=8,
            autofocus=True,
        )
        send_btn = gr.Button("发送", variant="primary", scale=1)

    clear_btn = gr.Button("🗑️ 清空对话", variant="secondary")

    gr.Examples(
        examples=["什么是 CPI？它对债券价格有什么影响？",
                  "流感应该如何预防？",
                  "签订劳动合同需要注意哪些条款？",
                  "什么是 RAG？"],
        inputs=msg,
        label="试试这些",
    )

    msg.submit(respond, [msg, chatbot, session], [chatbot, session, msg])
    send_btn.click(respond, [msg, chatbot, session], [chatbot, session, msg])
    clear_btn.click(reset, outputs=[chatbot, session, msg])


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
