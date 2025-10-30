from __future__ import annotations

from typing import List, Dict, Any
import heapq

from openai import OpenAI

from .embed import embed_query
from .index import load_index, search as faiss_search
from .store import list_manuals, load_chunks


def _gather_candidates(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    query_vec = embed_query(query)

    candidates: List[Dict[str, Any]] = []
    manuals = list_manuals()

    for m in manuals:
        mid = m["id"]
        try:
            from .store import manual_paths
            idx = load_index(manual_paths(mid)["index"])
            scores, indices = faiss_search(idx, query_vec, top_k=top_k)
            chunks = load_chunks(mid)
            for s, i in zip(scores, indices):
                if i < 0 or i >= len(chunks):
                    continue
                c = chunks[i]
                candidates.append({
                    "manual_id": mid,
                    "score": float(s),
                    "chunk": c,
                })
        except Exception:
            continue

    # take global top_k by score (IP == cosine similarity)
    top = heapq.nlargest(top_k, candidates, key=lambda x: x["score"]) if candidates else []
    return top


def _build_context(cands: List[Dict[str, Any]], max_chars: int = 1600) -> str:
    parts: List[str] = []
    for i, item in enumerate(cands, 1):
        ch = item["chunk"]
        header = ch.get("header", "")
        start_page = ch.get("start_page", "?")
        content = (ch.get("content", "") or "")
        snippet = content[: max(200, min(len(content), max_chars // len(cands) // 2))]
        parts.append(
            f"--- 관련 문서 #{i} (제목: {header}, 페이지: {start_page}) ---\n{snippet}\n"
        )
    return "\n".join(parts)


def _build_prompt(context: str, query: str) -> str:
    return (
        "당신은 선박 기기 매뉴얼 전문가입니다. 아래 근거 문서만 활용해 간결/정확한 한국어 답변을 작성하세요. 모호하면 추가 질문을 요청하세요.\n\n"
        "[근거]\n" + context + "\n\n[질문]\n" + query + "\n\n"
        "요구: 핵심 Bullet→절차→주의/한계. 마지막에 참고 출처(제목·페이지)."
    )


def answer(query: str, top_k: int = 5) -> Dict[str, Any]:
    # Guard: no manuals
    manuals = list_manuals()
    if not manuals:
        return {
            "answer": "업로드된 매뉴얼이 없습니다. 상단 '소스 업로드'로 PDF를 등록·인덱싱한 뒤 다시 질문해 주세요.",
            "citations": [],
        }

    cands = _gather_candidates(query, top_k=top_k)
    if not cands:
        return {
            "answer": "관련 문서를 찾지 못했습니다. 매뉴얼 업로드/인덱싱 상태를 확인하거나 질문을 더 구체화해 주세요.",
            "citations": [],
        }

    context = _build_context(cands)
    prompt = _build_prompt(context, query)

    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "당신은 정확한 기술 매뉴얼 어시스턴트입니다."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    text = resp.choices[0].message.content

    citations = [
        {
            "title": it["chunk"].get("header", ""),
            "page": it["chunk"].get("start_page", "?"),
            "score": it["score"],
        }
        for it in cands
    ]

    return {"answer": text, "citations": citations}
