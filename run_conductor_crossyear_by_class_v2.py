#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_conductor_crossyear_by_class_v3.py

[목표]
- run_reports_parallel_conductor.py 결과(YYYY_final_corpus.jsonl)를 읽어서
- 중분류(NODE_CLSS_02)별로 2021~2025 변화(연도별 요약 + 21→22→23→24→25 전이 분석 + 총평)를
  "파싱 없이" Markdown으로 안정적으로 생성한다.

[핵심 변경]
- JSON 파싱 완전 제거(파싱 실패 원천 차단)
- LLM 출력 형식을 Markdown 템플릿으로 강제
- 연도 누락이 있어도 전이(Transition) 섹션을 반드시 모두 생성하도록 규칙을 프롬프트에 포함

[필수]
- reports_dir에 다음 파일들이 있어야 함:
  - 2021_final_corpus.jsonl
  - 2022_final_corpus.jsonl
  - ... years로 지정한 연도들

[실행 예시]
python run_conductor_crossyear_by_class_v3.py \
  --reports_dir "/Users/idonghyeon/Desktop/dathon_result/reports_parallel" \
  --out_dir "/Users/idonghyeon/Desktop/dathon_result/reports_parallel" \
  --years "2021,2022,2023,2024,2025" \
  --conductor_model "gpt-5-mini" \
  --max_tokens 2200 \
  --sleep 0.3 \
  --preview
"""

import os
import re
import json
import time
import argparse
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI


# =========================
# utils
# =========================
def safe_filename(s: str) -> str:
    s = str(s)
    return "".join(c if c.isalnum() or c in " _-." else "_" for c in s).strip()


def fill_prompt(tpl: str, **kwargs) -> str:
    # simple replace to avoid brace conflicts
    for k, v in kwargs.items():
        tpl = tpl.replace("{" + k + "}", str(v))
    return tpl


def strip_code_fences(text: str) -> str:
    if not text:
        return ""
    lines = []
    for ln in text.splitlines():
        if ln.strip().startswith("```"):
            continue
        lines.append(ln)
    return "\n".join(lines).strip()


def call_llm(client: OpenAI, model: str, prompt: str, max_output_tokens: int = 2200) -> str:
    resp = client.responses.create(
        model=model,
        input=prompt,
        max_output_tokens=max_output_tokens,
    )
    if getattr(resp, "output_text", None):
        return (resp.output_text or "").strip()

    parts = []
    for item in getattr(resp, "output", []) or []:
        for c in getattr(item, "content", []) or []:
            if getattr(c, "type", "") in ("output_text", "text"):
                parts.append(getattr(c, "text", "") or "")
    return "\n".join([p for p in parts if p]).strip()


def ensure_str_list(x: Any) -> List[str]:
    if not x:
        return []
    if isinstance(x, list):
        return [str(i) for i in x if str(i).strip()]
    s = str(x).strip()
    return [s] if s else []


def save_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text or "")


# =========================
# corpus loader
# =========================
def load_year_corpus_jsonl(reports_dir: str, year: int) -> List[dict]:
    path = os.path.join(reports_dir, f"{year}_final_corpus.jsonl")
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = (ln or "").strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except Exception:
                continue
    return out


def flatten_keywords_from_topics(topics: Any, limit: int = 10) -> List[str]:
    kw = []
    if not isinstance(topics, list):
        return kw
    for t in topics:
        if not isinstance(t, dict):
            continue
        for k in ensure_str_list(t.get("keywords", []) or []):
            k = str(k).strip()
            if k and k not in kw:
                kw.append(k)
            if len(kw) >= limit:
                return kw
    return kw


def build_year_block_for_prompt_short(r: dict) -> str:
    """
    전이 비교에 필요한 신호(stable/emerging/topics)를 최대한 짧게 제공.
    """
    y = int(r.get("Year", 0) or 0)
    status = (r.get("status", "") or "").strip()
    one_line = (r.get("one_line_kr", "") or "").strip()
    final_note = (r.get("final_note_kr", "") or "").strip()

    topics = r.get("topics", []) or []
    comp = r.get("compare", {}) or {}

    stable = ensure_str_list(comp.get("stable_top10", []) or [])[:10]
    emerging = ensure_str_list(comp.get("emerging_top10", []) or [])[:10]

    topic_lines = []
    if isinstance(topics, list):
        for t in topics[:5]:
            if not isinstance(t, dict):
                continue
            kr = (t.get("kr_name", "") or "").strip()
            en = (t.get("en_name", "") or "").strip()
            kws = ensure_str_list(t.get("keywords", []) or [])[:8]
            if kr or en:
                topic_lines.append(f"{kr}/{en}: " + ", ".join(kws))

    flat_kws = flatten_keywords_from_topics(topics, limit=10)

    block = []
    block.append(f"YEAR={y} | status={status}")
    if one_line:
        block.append(f"- one_line_kr: {one_line}")
    if final_note:
        block.append(f"- final_note_kr: {final_note}")
    if topic_lines:
        block.append("- topics:")
        for x in topic_lines:
            block.append(f"  * {x}")
    if flat_kws:
        block.append(f"- topic_keywords_hint: {', '.join(flat_kws)}")
    if stable:
        block.append(f"- stable_top10: {', '.join(stable)}")
    if emerging:
        block.append(f"- emerging_top10: {', '.join(emerging)}")
    return "\n".join(block)


# =========================
# Conductor prompt (cross-year, Markdown only)
# =========================
PROMPT_CROSSYEAR_TRANSITION_MD = """너는 "연도 전이 기반 트렌드 변화"를 요약하는 Conductor다.
대상 중분류: {class_name}
대상 기간: {start_year}~{end_year}

아래 INPUT_BLOCKS는 {class_name}의 연도별 분석 요약이다.
반드시 INPUT_BLOCKS에 등장한 키워드/토픽명만 사용하라(새 키워드 생성 금지).

중요 규칙:
- 출력은 JSON 금지. 오직 Markdown만 출력.
- 아래 섹션 구조/제목/순서를 정확히 지켜라(제목 텍스트 유지).
- "연도별 요약"에는 {start_year}~{end_year} 모든 연도를 반드시 포함하라.
  - 특정 연도의 INPUT_BLOCK이 없으면, 해당 연도는 "데이터 부족"으로 표기하라.
- "전이(Transition) 분석"은 {start_year}→{start_year+1}, ... , {end_year-1}→{end_year} 전이를 모두 반드시 작성하라.
  - 결측 연도가 있으면, 그 전이에서 "데이터 부족/결측"을 원인으로 명시하라.
- 각 전이에는 반드시 아래 4줄을 포함:
  - 변화 요약(1~2문장)
  - 증가/부상 키워드(3~6개, 콤마로)
  - 감소/안정 키워드(3~6개, 콤마로)
  - 근거(입력의 topics/stable_top10/emerging_top10 중 어떤 신호인지 1문장)
- 마지막:
  - 총평(흐름) 4~6문장
  - 핵심 포인트 4~6개 bullet

형식(그대로):
# {class_name} (2021~2025 변화)

## 1) 연도별 요약
- 2021: ...
- 2022: ...
- 2023: ...
- 2024: ...
- 2025: ...

## 2) 전이(Transition) 분석
### 2021→2022
- 변화 요약: ...
- 증가/부상 키워드: k1, k2, ...
- 감소/안정 키워드: k1, k2, ...
- 근거: ...

### 2022→2023
...

### 2023→2024
...

### 2024→2025
...

## 3) 총평(흐름)
...

## 4) 핵심 포인트
- ...
- ...
- ...

INPUT_BLOCKS:
{blocks}
"""


# =========================
# fallback generator (LLM empty / failure)
# =========================
def _extract_year_from_block(block: str) -> int:
    m = re.search(r"YEAR\s*=\s*(\d{4})", block)
    return int(m.group(1)) if m else 0


def build_fallback_md(class_name: str, years: List[int], blocks: List[str]) -> str:
    """
    LLM 출력이 비었을 때 최소한의 보고서 형태를 보장.
    - 입력 블록에서 stable/emerging 힌트만 추출해서 채움
    """
    year_to_block: Dict[int, str] = {}
    for b in blocks:
        y = _extract_year_from_block(b)
        if y:
            year_to_block[y] = b

    def pick_list(b: str, key: str, k: int = 6) -> List[str]:
        # e.g., "- stable_top10: a, b, c"
        m = re.search(rf"-\s*{re.escape(key)}\s*:\s*(.*)", b)
        if not m:
            return []
        items = [x.strip() for x in m.group(1).split(",") if x.strip()]
        return items[:k]

    lines = []
    lines.append(f"# {class_name} ({min(years)}~{max(years)} 변화)\n")
    lines.append("## 1) 연도별 요약")
    for y in years:
        b = year_to_block.get(y, "")
        if not b:
            lines.append(f"- {y}: 데이터 부족")
            continue
        ol = ""
        m1 = re.search(r"-\s*one_line_kr\s*:\s*(.*)", b)
        if m1:
            ol = m1.group(1).strip()
        if not ol:
            m2 = re.search(r"-\s*final_note_kr\s*:\s*(.*)", b)
            if m2:
                ol = m2.group(1).strip()
        lines.append(f"- {y}: {ol if ol else '입력 요약(간략)'}")
    lines.append("")

    lines.append("## 2) 전이(Transition) 분석")
    for i in range(len(years) - 1):
        y1, y2 = years[i], years[i + 1]
        b1, b2 = year_to_block.get(y1, ""), year_to_block.get(y2, "")
        em2 = pick_list(b2, "emerging_top10", 6) if b2 else []
        st2 = pick_list(b2, "stable_top10", 6) if b2 else []
        st1 = pick_list(b1, "stable_top10", 6) if b1 else []

        lines.append(f"### {y1}→{y2}")
        if (not b1) or (not b2):
            lines.append("- 변화 요약: 데이터 결측으로 전이 비교가 제한됨.")
            lines.append(f"- 증가/부상 키워드: {', '.join(em2) if em2 else '데이터 부족'}")
            lines.append(f"- 감소/안정 키워드: {', '.join(st2) if st2 else '데이터 부족'}")
            lines.append("- 근거: 해당 연도의 입력 블록이 부족하여 stable/emerging 신호를 제한적으로 사용함.")
        else:
            lines.append("- 변화 요약: 전년도 대비 emerging 신호와 stable 구성 변화 관측(간략).")
            lines.append(f"- 증가/부상 키워드: {', '.join(em2) if em2 else 'emerging 신호 미약'}")
            lines.append(f"- 감소/안정 키워드: {', '.join(st2 if st2 else st1) if (st2 or st1) else 'stable 신호 미약'}")
            lines.append("- 근거: 입력 블록의 stable_top10 / emerging_top10 기반(LLM 미사용 fallback).")
        lines.append("")

    lines.append("## 3) 총평(흐름)")
    lines.append("LLM 출력 실패로 인해 규칙 기반 요약만 제공됨. 연도별 stable/emerging 신호를 바탕으로 전이 변화를 점검 필요.\n")
    lines.append("## 4) 핵심 포인트")
    lines.append("- 데이터 결측 여부를 먼저 확인")
    lines.append("- emerging 신호가 실제 증가를 반영하는지 원본/병합 지표로 재검증")
    lines.append("- 연도별 stable 구성 변화로 관심사의 이동 여부 확인")
    lines.append("- 후보 풀 희소성(키워드 다양성) 지표를 함께 해석")
    lines.append("")
    return "\n".join(lines)


# =========================
# main runner
# =========================
def run_crossyear(
    reports_dir: str,
    out_dir: str,
    years: List[int],
    sleep_sec: float,
    max_tokens: int,
    conductor_model: str,
    preview: bool,
):
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY 환경변수가 없습니다. export OPENAI_API_KEY=...")

    client = OpenAI()

    all_records: List[dict] = []
    for y in years:
        recs = load_year_corpus_jsonl(reports_dir, y)
        if preview:
            print(f"[INFO] load {y}: {len(recs)} records")
        all_records.extend(recs)

    if not all_records:
        print("[WARN] no corpus records loaded.")
        return

    by_class: Dict[str, List[dict]] = {}
    for r in all_records:
        cls = r.get("NODE_CLSS_02", None)
        if not cls:
            continue
        by_class.setdefault(str(cls), []).append(r)

    out_base = os.path.join(out_dir, "crossyear_by_class")
    raw_dir = os.path.join(out_base, "_raw")
    os.makedirs(out_base, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    start_y = min(years)
    end_y = max(years)

    for cls, items in sorted(by_class.items(), key=lambda x: x[0]):
        items_sorted = sorted(items, key=lambda x: int(x.get("Year", 0) or 0))

        blocks = [build_year_block_for_prompt_short(r) for r in items_sorted]
        prompt = fill_prompt(
            PROMPT_CROSSYEAR_TRANSITION_MD,
            class_name=cls,
            start_year=str(start_y),
            end_year=str(end_y),
            blocks="\n\n---\n\n".join(blocks),
        )

        if preview:
            print(f"\n[INFO] conductor start: class={cls} blocks={len(blocks)}")

        text = ""
        try:
            text = call_llm(client, conductor_model, prompt, max_output_tokens=max_tokens)
        except Exception as e:
            print(f"[ERR] LLM call failed: class={cls} err={e}")
            text = ""

        text = strip_code_fences(text)

        # 최소 안전장치: 빈 출력이면 fallback MD 생성
        if not (text or "").strip():
            text = build_fallback_md(cls, years, blocks)

        cls_safe = safe_filename(cls)
        md_path = os.path.join(out_base, f"{cls_safe}_transition_{start_y}-{end_y}.md")
        raw_path = os.path.join(raw_dir, f"{cls_safe}_transition_{start_y}-{end_y}.txt")

        save_text(md_path, text.strip() + "\n")
        save_text(raw_path, text or "")

        print(f"[OK] {md_path}")
        time.sleep(sleep_sec)

    print("\nDONE (cross-year transition markdown by class)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--reports_dir", required=True, help="YYYY_final_corpus.jsonl이 있는 폴더")
    p.add_argument("--out_dir", required=True, help="출력 폴더")
    p.add_argument("--years", default="2021,2022,2023,2024,2025", help="쉼표 구분")
    p.add_argument("--sleep", type=float, default=0.3)
    p.add_argument("--max_tokens", type=int, default=2200)
    p.add_argument("--conductor_model", default=os.getenv("OPENAI_CONDUCTOR_MODEL", "gpt-5-mini"))
    p.add_argument("--preview", action="store_true")
    args = p.parse_args()

    years = [int(x.strip()) for x in args.years.split(",") if x.strip()]

    run_crossyear(
        reports_dir=args.reports_dir,
        out_dir=args.out_dir,
        years=years,
        sleep_sec=args.sleep,
        max_tokens=args.max_tokens,
        conductor_model=args.conductor_model,
        preview=args.preview,
    )
