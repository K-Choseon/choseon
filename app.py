from __future__ import annotations

import io
import os
import tempfile
from typing import List, Dict, Any

import streamlit as st
from dotenv import load_dotenv

from rag.parser import pdf_parser
from rag.embed import embed_texts
from rag.index import build_faiss_ip_index, save_index
from rag.store import (
    list_manuals,
    register_manual,
    update_meta_counts,
    save_chunks,
    save_embeddings,
    manual_paths,
)
from rag.chat import answer as rag_answer


load_dotenv()
st.set_page_config(page_title="선박 매뉴얼 RAG 챗봇", layout="wide")


def _has_api_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


if not _has_api_key():
    st.warning("OPENAI_API_KEY가 설정되지 않았습니다. .env에 키를 넣어주세요.")


# --- Session State ---
if "conversations" not in st.session_state:
    st.session_state.conversations = {}  # chat_id -> list[ {role, content} ]
if "active_chat" not in st.session_state:
    st.session_state.active_chat = None
if "show_upload" not in st.session_state:
    st.session_state.show_upload = False


def _new_chat() -> str:
    chat_id = f"chat-{len(st.session_state.conversations) + 1}"
    st.session_state.conversations[chat_id] = []
    st.session_state.active_chat = chat_id
    return chat_id


def _sidebar():
    st.sidebar.title("대화 히스토리")
    chats = list(st.session_state.conversations.keys())
    if st.sidebar.button("+ 새 대화", use_container_width=True):
        _new_chat()

    for cid in chats:
        if st.sidebar.button(cid, use_container_width=True):
            st.session_state.active_chat = cid

    st.sidebar.markdown("---")
    if st.sidebar.button("퀴즈", use_container_width=True):
        try:
            st.switch_page("pages/01_퀴즈.py")
        except Exception:
            st.sidebar.info("좌측 Pages에서 '퀴즈'를 선택하세요.")


def _upload_dialog_body():
    st.subheader("소스 업로드")

    if not _has_api_key():
        st.info("OPENAI_API_KEY 설정 후 이용해 주세요.")
        return

    # Existing manuals
    manuals = list_manuals()
    if manuals:
        st.markdown("#### 업로드된 매뉴얼")
        for m in manuals:
            st.caption(f"- {m['title']} (id: {m['id']})")
    else:
        st.caption("아직 업로드된 매뉴얼이 없습니다.")

    st.markdown("---")
    file = st.file_uploader(
        "PDF 매뉴얼 업로드", type=["pdf"], accept_multiple_files=False
    )

    if file is not None:
        title = st.text_input("매뉴얼 제목", value=os.path.splitext(file.name)[0])
        proceed = st.button("업로드 및 인덱싱 시작", type="primary")
        if proceed:
            with st.status("인덱싱 중...", expanded=True) as status:
                try:
                    # Save uploaded PDF to temp
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".pdf"
                    ) as tmp:
                        tmp.write(file.read())
                        tmp_path = tmp.name
                    st.write("1/4 PDF 저장 완료")

                    # Register manual -> copy to data folder
                    meta = register_manual(title, tmp_path)
                    mid = meta["id"]
                    st.write(f"2/4 매뉴얼 등록 완료(id: {mid})")

                    # Parse -> chunks
                    chunks = pdf_parser(manual_paths(mid)["pdf"])
                    save_chunks(mid, chunks)
                    st.write(f"3/4 파싱 완료, 청크 수: {len(chunks)}")

                    # Embed -> index
                    docs = [
                        f"제목: {c.get('header','')}, 내용: {c.get('content','')}"
                        for c in chunks
                    ]
                    emb = embed_texts(docs)
                    save_embeddings(mid, emb)
                    idx = build_faiss_ip_index(emb)
                    save_index(idx, manual_paths(mid)["index"])

                    # Update meta
                    try:
                        import fitz

                        with fitz.open(manual_paths(mid)["pdf"]) as d:
                            update_meta_counts(mid, d.page_count, len(chunks))
                    except Exception:
                        update_meta_counts(mid, None, len(chunks))

                    status.update(label="완료", state="complete")
                    st.success("업로드/인덱싱이 완료되었습니다.")
                except Exception as e:
                    status.update(label="실패", state="error")
                    st.error(f"오류: {e}")
                finally:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass


def _topbar_upload_button():
    col1, col2, col3 = st.columns([1, 6, 1])
    with col3:
        if st.button("소스 업로드", disabled=not _has_api_key()):
            st.session_state.show_upload = True
        if not _has_api_key():
            st.caption("API 키가 없으면 업로드/인덱싱을 사용할 수 없습니다.")

    # Modal/dialog (Streamlit 1.32+: st.dialog)
    if st.session_state.show_upload:
        try:

            @st.dialog("소스 업로드", width="large")
            def _dlg():
                _upload_dialog_body()
                if st.button("닫기"):
                    st.session_state.show_upload = False

            _dlg()
        except Exception:
            with st.expander("소스 업로드", expanded=True):
                _upload_dialog_body()
                if st.button("닫기"):
                    st.session_state.show_upload = False


def _chat_body():
    _topbar_upload_button()

    if st.session_state.active_chat is None:
        _new_chat()

    st.markdown("## AI와 대화")

    # Render history
    for msg in st.session_state.conversations[st.session_state.active_chat]:
        if msg["role"] == "user":
            st.chat_message("user").markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])
                cites = msg.get("citations") or []
                if cites:
                    st.caption(
                        "출처: "
                        + ", ".join([f"{c['title']} (p.{c['page']})" for c in cites])
                    )

    if not _has_api_key():
        st.info("OPENAI_API_KEY 설정 후 채팅을 이용할 수 있습니다.")
        return

    # Guard: no manuals
    if not list_manuals():
        st.info(
            "업로드된 매뉴얼이 없습니다. 우상단 '소스 업로드'에서 PDF를 등록해 주세요."
        )
        return

    # Input
    prompt = st.chat_input("메뉴얼에 대해 물어보세요…")
    if prompt:
        st.session_state.conversations[st.session_state.active_chat].append(
            {"role": "user", "content": prompt}
        )
        with st.chat_message("assistant"):
            with st.spinner("검색 중…"):
                res = rag_answer(prompt, top_k=5)
                answer_text = res.get("answer", "")
                citations = res.get("citations", [])
                st.markdown(answer_text)
                if citations:
                    st.caption(
                        "출처: "
                        + ", ".join(
                            [f"{c['title']} (p.{c['page']})" for c in citations]
                        )
                    )
        st.session_state.conversations[st.session_state.active_chat].append(
            {"role": "assistant", "content": answer_text, "citations": citations}
        )


# --- Layout ---
_sidebar()
_chat_body()
