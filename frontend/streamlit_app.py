from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

API_BASE_URL = os.getenv("RAG_API_BASE_URL", "http://127.0.0.1:8001").rstrip("/")
SUGGESTED_QUESTIONS = [
    "低压蓄电池没电怎么办？",
    "车辆充电时有哪些注意事项？",
    "如何使用自动驻车功能？",
    "车辆长期停放需要注意什么？",
]


def configure_page() -> None:
    st.set_page_config(
        page_title="智驾手册助手",
        page_icon="🚘",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .stApp { background: #f7f8fa; }
        [data-testid="stSidebar"] { background: #111827; }
        [data-testid="stSidebar"] * { color: #f9fafb; }
        .hero {
            padding: 1.5rem 1.7rem;
            border-radius: 18px;
            background: linear-gradient(120deg, #0f172a 0%, #1d4ed8 100%);
            color: white;
            margin-bottom: 1.25rem;
            box-shadow: 0 12px 28px rgba(15, 23, 42, .16);
        }
        .hero h1 { margin: 0 0 .35rem; font-size: 2rem; }
        .hero p { margin: 0; color: #dbeafe; }
        .source-card {
            background: white;
            border: 1px solid #e5e7eb;
            border-left: 4px solid #2563eb;
            border-radius: 10px;
            padding: .8rem 1rem;
            margin: .45rem 0;
        }
        .source-title { color: #111827; font-weight: 650; }
        .source-meta { color: #64748b; font-size: .88rem; margin-top: .2rem; }
        .tech-pill {
            display: inline-block;
            padding: .2rem .55rem;
            margin: .15rem .1rem;
            border: 1px solid #475569;
            border-radius: 999px;
            font-size: .78rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header() -> None:
    st.markdown(
        """
        <div class="hero">
          <h1>🚘 智驾手册助手</h1>
          <p>基于《问界 M5 纯电版用户手册》回答用车问题，每条答案均附说明书依据。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar() -> None:
    with st.sidebar:
        st.title("车辆与知识库")
        st.caption("当前知识库")
        st.markdown("**AITO 问界 M5 · 纯电版**")
        st.success("知识库已就绪", icon="✅")
        st.divider()
        st.caption("检索链路")
        st.markdown(
            """
            <span class="tech-pill">Dense</span>
            <span class="tech-pill">BM25 Sparse</span>
            <span class="tech-pill">RRF Hybrid</span>
            <span class="tech-pill">Reranker</span>
            <span class="tech-pill">Qwen Plus</span>
            """,
            unsafe_allow_html=True,
        )
        st.divider()
        st.caption("回答范围")
        st.write("仅依据已收录的官方用户手册，不确定时会明确拒答。")
        if st.button("清空对话", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


def source_html(title: str, section: str, page_label: str) -> str:
    return f"""
    <div class="source-card">
      <div class="source-title">{title}</div>
      <div class="source-meta">{section} · {page_label}</div>
    </div>
    """


def ask_api(question: str) -> dict[str, Any]:
    request = Request(
        f"{API_BASE_URL}/ask",
        data=json.dumps({"question": question}, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        try:
            detail = json.loads(error.read().decode("utf-8")).get("detail")
        except (json.JSONDecodeError, UnicodeDecodeError):
            detail = None
        raise RuntimeError(detail or f"API 请求失败（HTTP {error.code}）") from error
    except URLError as error:
        raise RuntimeError(
            f"无法连接问答 API：{API_BASE_URL}，请确认 FastAPI 服务已经启动"
        ) from error


def response_to_message(response: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "request_id": response["request_id"],
        "content": response["answer"],
        "refused": response["refused"],
        "citations": response["citations"],
        "evidence": response["evidence"],
        "elapsed_seconds": response["elapsed_seconds"],
        "usage": response["usage"],
    }


def render_assistant_message(message: dict[str, Any]) -> None:
    if message.get("refused"):
        st.warning(message["content"], icon="⚠️")
    else:
        st.markdown(message["content"])

    citations = message.get("citations", [])
    if citations:
        st.caption("说明书依据")
        for citation in citations:
            if citation["page_start"] == citation["page_end"]:
                page_label = f"第 {citation['page_start']} 页"
            else:
                page_label = f"第 {citation['page_start']}–{citation['page_end']} 页"
            st.markdown(
                source_html(
                    citation["document_title"],
                    citation["section"],
                    page_label,
                ),
                unsafe_allow_html=True,
            )

    evidence = message.get("evidence", [])
    if evidence:
        with st.expander("查看检索原文与重排分数"):
            for index, item in enumerate(evidence, start=1):
                page = (
                    str(item["page_start"])
                    if item["page_start"] == item["page_end"]
                    else f"{item['page_start']}–{item['page_end']}"
                )
                st.markdown(
                    f"**候选 {index}｜{item['section']}｜第 {page} 页**  "
                    f"`rerank={item['reranker_score']:.4f}`"
                )
                st.text(item["text"])
                if index < len(evidence):
                    st.divider()

    if message.get("elapsed_seconds") is not None:
        usage = message.get("usage", {})
        usage_label = (
            f" · Token {usage['total_tokens']}" if usage.get("total_tokens") else ""
        )
        request_label = (
            f" · 请求 ID `{message['request_id']}`" if message.get("request_id") else ""
        )
        st.caption(
            f"本次回答耗时 {message['elapsed_seconds']:.2f} 秒"
            f"{usage_label}{request_label}"
        )


def render_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_assistant_message(message)
            else:
                st.markdown(message["content"])


def suggestion_input() -> str | None:
    if st.session_state.messages:
        return None
    st.caption("你可以这样问")
    columns = st.columns(2)
    for index, question in enumerate(SUGGESTED_QUESTIONS):
        if columns[index % 2].button(question, key=f"suggestion-{index}", use_container_width=True):
            return question
    return None


def answer_question(question: str) -> None:
    user_message = {"role": "user", "content": question}
    st.session_state.messages.append(user_message)
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.status("正在检索说明书…", expanded=True) as status:
                st.write("向问答 API 提交问题")
                st.write("后端执行混合召回、重排与答案生成")
                response = ask_api(question)
                st.write("后端已完成引用校验")
                status.update(label="回答生成完成", state="complete", expanded=False)
            message = response_to_message(response)
            st.session_state.messages.append(message)
            render_assistant_message(message)
        except Exception as error:  # noqa: BLE001 - UI boundary must surface provider failures
            st.error("回答生成失败，请检查模型配置或稍后重试。", icon="🚨")
            with st.expander("查看错误详情"):
                st.code(str(error))


def main() -> None:
    configure_page()
    sidebar()
    page_header()
    if "messages" not in st.session_state:
        st.session_state.messages = []

    render_history()
    suggested = suggestion_input()
    typed = st.chat_input("请输入关于车辆功能、充电、保养或故障处理的问题…")
    question = typed or suggested
    if question:
        answer_question(question)


if __name__ == "__main__":
    main()
