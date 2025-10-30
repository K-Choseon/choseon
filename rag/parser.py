import statistics
import re
from typing import List, Dict, Any

import fitz  # PyMuPDF
import pandas as pd


def _extract_page_elements(page: fitz.Page) -> List[Dict[str, Any]]:
    elements: List[Dict[str, Any]] = []

    # Tables
    try:
        tables = page.find_tables()
    except Exception:
        tables = []
    table_bboxes = [fitz.Rect(t.bbox) for t in tables]

    # Text blocks
    try:
        page_dict = page.get_text("dict", flags=fitz.TEXTFLAGS_SEARCH)
        blocks = page_dict.get("blocks", [])
    except Exception:
        blocks = []

    # Base font size estimation
    font_sizes = [
        span["size"]
        for block in blocks
        if block.get("type") == 0
        for line in block.get("lines", [])
        for span in line.get("spans", [])
    ]
    base_font_size = statistics.mode(font_sizes) if font_sizes else 10.0

    # Add text blocks excluding table areas
    for block in blocks:
        if block.get("type") != 0:
            continue
        bbox = fitz.Rect(block.get("bbox"))
        if any(bbox.intersects(tb) for tb in table_bboxes):
            continue
        elements.append({"type": "text", "bbox": bbox, "data": block, "base_font": base_font_size})

    # Add table markdown
    for i, t in enumerate(tables):
        try:
            df = t.to_pandas()
            if not df.empty:
                elements.append({
                    "type": "table",
                    "bbox": table_bboxes[i],
                    "data": df.to_markdown(index=False)
                })
        except Exception:
            continue

    # Images
    try:
        for img_info in page.get_images(full=True):
            bbox = page.get_image_bbox(img_info)
            elements.append({"type": "image", "bbox": bbox, "data": "[IMAGE]"})
    except Exception:
        pass

    # Sort by visual order (top to bottom)
    elements.sort(key=lambda x: x["bbox"].y0)
    return elements


def pdf_parser(pdf_path: str) -> List[Dict[str, Any]]:
    doc = fitz.open(pdf_path)

    final_chunks: List[Dict[str, Any]] = []
    current_chunk: Dict[str, Any] | None = None
    last_main_header_text: str = ""

    try:
        for page_num, page in enumerate(doc, start=1):
            elements = _extract_page_elements(page)

            for elem in elements:
                etype = elem["type"]

                # Tables/Images: attach to current or previous section
                if etype in ("table", "image"):
                    if current_chunk is not None:
                        current_chunk.setdefault("content", []).append(elem["data"])
                        if etype == "image":
                            current_chunk["has_image"] = True
                    elif not final_chunks:
                        current_chunk = {
                            "header": "Initial Content",
                            "content": [elem["data"]],
                            "start_page": page_num,
                            "type": "section",
                            "has_image": etype == "image",
                        }
                    else:
                        final_chunks[-1].setdefault("content", []).insert(0, elem["data"])
                        if etype == "image":
                            final_chunks[-1]["has_image"] = True
                    continue

                # Text blocks
                block = elem["data"]
                base_font = elem["base_font"]
                raw_text = " ".join([
                    span.get("text", "")
                    for line in block.get("lines", [])
                    for span in line.get("spans", [])
                ]).strip()
                text = raw_text.replace("\n", " ")
                if not text:
                    continue

                # Heuristic: header vs body
                content_type = "paragraph"
                try:
                    first_span = block["lines"][0]["spans"][0]
                    if first_span["size"] > base_font * 1.5:
                        content_type = "header_level_1"
                    elif first_span["size"] > base_font * 1.2:
                        content_type = "header_level_2"
                except Exception:
                    pass

                # Pattern hints
                if re.match(r"^\d+\.\s*", text):
                    content_type = "header_level_1"
                elif re.match(r"^\d+-\d+\.\s*", text):
                    content_type = "header_level_2"

                if content_type.startswith("header"):
                    # finalize previous chunk
                    if current_chunk is not None:
                        merged = "".join(current_chunk.get("content", [])).strip()
                        current_chunk["content"] = merged
                        if merged:
                            final_chunks.append(current_chunk)

                    header_text = text
                    if content_type == "header_level_1":
                        last_main_header_text = text
                    elif last_main_header_text:
                        header_text = f"{last_main_header_text} - {text}"

                    current_chunk = {
                        "header": header_text,
                        "content": [],
                        "start_page": page_num,
                        "type": "section",
                        "has_image": False,
                    }
                else:
                    if current_chunk is not None:
                        current_chunk.setdefault("content", []).append(text)

        # flush last
        if current_chunk is not None:
            merged = "".join(current_chunk.get("content", [])).strip()
            current_chunk["content"] = merged
            if merged:
                final_chunks.append(current_chunk)

    finally:
        doc.close()

    # assign stable ids
    for i, ch in enumerate(final_chunks):
        ch.setdefault("id", f"chunk-{i+1}")
        ch.setdefault("has_image", False)

    return final_chunks
