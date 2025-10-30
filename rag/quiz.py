from __future__ import annotations

from typing import List, Dict, Any
import random

from openai import OpenAI

from .store import load_chunks


def _sample_context(chunks: List[Dict[str, Any]], max_chars: int = 3000) -> str:
    random.seed(42)
    random.shuffle(chunks)
    acc = []
    total = 0
    for ch in chunks:
        piece = f"제목: {ch.get('header','')}\n내용: {ch.get('content','')}\n"
        if total + len(piece) > max_chars:
            break
        acc.append(piece)
        total += len(piece)
    return "\n".join(acc)


def generate_quiz(manual_id: str, num_questions: int = 5) -> List[Dict[str, Any]]:
    chunks = load_chunks(manual_id)
    ctx = _sample_context(chunks)

    prompt = (
        "아래 선박 매뉴얼 내용을 바탕으로 객관식 퀴즈를 한국어로 만들어주세요.\n"
        "출력 형식은 JSON 배열이며 각 원소는 다음 키를 가집니다: \n"
        "{question: str, options: [str,str,str,str], answer_index: int, citation: {title: str, page: int}}\n"
        f"[자료]\n{ctx}\n"
        f"문항 수: {num_questions}\n"
    )

    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "당신은 정확한 시험 문제 출제자입니다."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    text = resp.choices[0].message.content or "[]"

    # Try to parse JSON; if fails, fallback to simple generation
    import json
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data[:num_questions]
    except Exception:
        pass

    # Fallback simple questions
    qs: List[Dict[str, Any]] = []
    for i in range(num_questions):
        ch = chunks[i % max(1, len(chunks))]
        qs.append({
            "question": f"다음 중 '{ch.get('header','')}'과 가장 관련이 깊은 것은?",
            "options": ["운전 절차", "정지 절차", "세정 방법", "점검 주기"],
            "answer_index": 0,
            "citation": {"title": ch.get("header",""), "page": ch.get("start_page", 0)},
        })
    return qs


def grade(quiz: List[Dict[str, Any]], user_choices: List[int]) -> Dict[str, Any]:
    correct = 0
    details = []
    for i, q in enumerate(quiz):
        ai = q.get("answer_index", 0)
        uc = user_choices[i] if i < len(user_choices) else -1
        ok = int(ai == uc)
        correct += ok
        details.append({
            "question": q.get("question", ""),
            "user": uc,
            "answer": ai,
            "correct": bool(ok),
            "citation": q.get("citation", {}),
        })
    return {"score": correct, "total": len(quiz), "details": details}
