from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

from rag.store import list_manuals
from rag.quiz import generate_quiz, grade

load_dotenv()
st.set_page_config(page_title="퀴즈", layout="wide")

import os

HAS_KEY = bool(os.getenv("OPENAI_API_KEY"))
if not HAS_KEY:
    st.warning("OPENAI_API_KEY가 설정되지 않았습니다. .env에 키를 넣어주세요.")

if "quiz" not in st.session_state:
    st.session_state.quiz = []
if "quiz_idx" not in st.session_state:
    st.session_state.quiz_idx = 0
if "answers" not in st.session_state:
    st.session_state.answers = []
if "quiz_manual" not in st.session_state:
    st.session_state.quiz_manual = None


# Sidebar: manual select and upload shortcut
st.sidebar.title("퀴즈 설정")
manuals = list_manuals()
manual_opts = {m["title"]: m["id"] for m in manuals} if manuals else None
if manual_opts:
    sel_title = st.sidebar.selectbox("매뉴얼 선택", [*manual_opts.keys()])
else:
    sel_title = "(없음)"

st.sidebar.markdown("---")
if st.sidebar.button("소스 업로드", use_container_width=True):
    try:
        st.switch_page("app.py")
    except Exception:
        st.sidebar.info("메인에서 '소스 업로드'를 이용하세요.")


col_left, col_main = st.columns([1, 3])
with col_main:
    st.markdown("## 객관식 퀴즈")

    if not manuals:
        st.info("먼저 메인 페이지에서 PDF 매뉴얼을 업로드하세요.")
    else:
        manual_id = manual_opts.get(sel_title) if manual_opts else None
        if manual_id and st.session_state.quiz_manual != manual_id:
            st.session_state.quiz_manual = manual_id
            st.session_state.quiz = []
            st.session_state.quiz_idx = 0
            st.session_state.answers = []

        if not st.session_state.quiz:
            if st.button("퀴즈 생성", type="primary", disabled=not HAS_KEY):
                with st.spinner("문항 생성 중…"):
                    st.session_state.quiz = generate_quiz(manual_id, num_questions=5)
                    st.session_state.quiz_idx = 0
                    st.session_state.answers = [-1] * len(st.session_state.quiz)

        if st.session_state.quiz:
            idx = st.session_state.quiz_idx
            q = st.session_state.quiz[idx]
            st.write(f"문제 {idx+1}/{len(st.session_state.quiz)}")
            st.markdown(f"**{q['question']}**")
            choice = st.radio(
                "정답을 선택하세요",
                options=list(range(len(q["options"]))),
                format_func=lambda i: q["options"][i],
                index=(
                    st.session_state.answers[idx]
                    if st.session_state.answers[idx] >= 0
                    else 0
                ),
                key=f"q_{idx}_choice",
            )

            # Save choice
            st.session_state.answers[idx] = choice

            cols = st.columns([1, 1, 6])
            with cols[0]:
                if st.button("이전", disabled=idx == 0):
                    st.session_state.quiz_idx = max(0, idx - 1)
            with cols[1]:
                if st.button(
                    "다음",
                    type="primary",
                    disabled=idx == len(st.session_state.quiz) - 1,
                ):
                    st.session_state.quiz_idx = min(
                        len(st.session_state.quiz) - 1, idx + 1
                    )

            st.markdown("---")
            if st.button("결과 보기", type="secondary", disabled=not HAS_KEY):
                res = grade(st.session_state.quiz, st.session_state.answers)
                st.session_state.quiz_result = res

        if "quiz_result" in st.session_state:
            res = st.session_state.quiz_result
            st.success(f"점수: {res['score']} / {res['total']}")
            for i, d in enumerate(res["details"], 1):
                st.write(
                    f"{i}. {'✅' if d['correct'] else '❌'} 정답: {d['answer']+1}, 선택: {d['user']+1} | 출처: {d['citation'].get('title','')} (p.{d['citation'].get('page','?')})"
                )
