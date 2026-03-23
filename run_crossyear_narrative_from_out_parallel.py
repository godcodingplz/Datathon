#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_crossyear_narrative_from_out_parallel_v3.py

v3 변경점:
- (연도×중분류) 입력 블록에 "ref_keywords"를 추가 (LLM이 참고한 키워드 명시)
- 최종 md에도 "연도별 참고 키워드" 섹션을 추가

실행 예시:
python run_crossyear_narrative_from_out_parallel_v3.py \
  --out_parallel_dir "/Users/idonghyeon/Desktop/dathon_result/out_parallel" \
  --out_dir "/Users/idonghyeon/Desktop/dathon_result/crossyear_narratives" \
  --years "2021,2022,2023,2024,2025" \
  --model "gpt-4o-mini" \
  --max_tokens 1800 \
  --sleep 0.2 \
  --debug
"""

import os
import re
import json
import glob
import time
import argparse
from typing import Any, Dict, List, Tuple, Optional

from dotenv import load_dotenv
from openai import OpenAI


# =========================
# utils
# =========================
def safe_filename(s: str) -> str:
    s = str(s)
    return "".join(c if c.isalnum() or c in " _-." else "_" for c in s).strip()


def ensure_str_list(x: Any) -> List[str]:
    if not x:
        return []
    if isinstance(x, list):
        return [str(i).strip() for i in x if str(i).strip()]
    s = str(x).strip()
    return [s] if s else []


def strip_code_fences(text: str) -> str:
    if not text:
        return ""
    out = []
    for ln in text.splitlines():
        if ln.strip().startswith("```"):
            continue
        out.append(ln)
    return "\n".join(out).strip()


def call_llm(client: OpenAI, model: str, prompt: str, max_output_tokens: int) -> str:
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


def is_return_final_path(path: str) -> bool:
    return "return_final" in path.replace("\\", "/").lower()


def load_json_file(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                return json.load(f)
        except Exception:
            return None


def iter_json_records_from_jsonl(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = (ln or "").strip()
                if not ln:
                    continue
                try:
                    obj = json.loads(ln)
                    if isinstance(obj, dict):
                        yield obj
                except Exception:
                    continue
    except Exception:
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                for ln in f:
                    ln = (ln or "").strip()
                    if not ln:
                        continue
                    try:
                        obj = json.loads(ln)
                        if isinstance(obj, dict):
                            yield obj
                    except Exception:
                        continue
        except Exception:
            return


def find_all_json_like_files(out_parallel_dir: str) -> List[str]:
    pat_json = os.path.join(out_parallel_dir, "**", "*.json")
    pat_jsonl = os.path.join(out_parallel_dir, "**", "*.jsonl")
    files = glob.glob(pat_json, recursive=True) + glob.glob(pat_jsonl, recursive=True)
    files = [p for p in files if os.path.isfile(p) and not os.path.basename(p).startswith("._")]
    return sorted(files)


def pick_year_class(payload: dict, path: str) -> Tuple[Optional[int], Optional[str]]:
    y = payload.get("Year") or payload.get("year")
    cls = payload.get("NODE_CLSS_02") or payload.get("class_name") or payload.get("class")

    if y is not None and cls:
        try:
            return int(str(y).strip()), str(cls).strip()
        except Exception:
            pass

    base = os.path.basename(path)
    m = re.search(r"(20\d{2})", base)
    year = int(m.group(1)) if m else None
    return year, None


# =========================
# 키워드 추출(참고키워드 나열용)
# =========================
def extract_ref_keywords(payload: dict, limit: int = 30) -> List[str]:
    """
    우선순위:
    1) topics[*].keywords (토픽 생성 근거 키워드)
    2) compare.stable_top10 + compare.emerging_top10
    3) (있으면) candidate_keywords 같은 필드(확장)
    """
    kws: List[str] = []

    topics = payload.get("topics", []) or []
    if isinstance(topics, list):
        for t in topics:
            if not isinstance(t, dict):
                continue
            for k in ensure_str_list(t.get("keywords", []) or []):
                if k and k not in kws:
                    kws.append(k)
                if len(kws) >= limit:
                    return kws

    comp = payload.get("compare", {}) or payload.get("compare_result", {}) or {}
    for k in ensure_str_list(comp.get("stable_top10", []) or comp.get("stable", []) or []):
        if k and k not in kws:
            kws.append(k)
        if len(kws) >= limit:
            return kws

    for k in ensure_str_list(comp.get("emerging_top10", []) or comp.get("emerging", []) or []):
        if k and k not in kws:
            kws.append(k)
        if len(kws) >= limit:
            return kws

    # 혹시 스키마에 이런게 있으면 보조로
    for k in ensure_str_list(payload.get("candidate_keywords", []) or []):
        if k and k not in kws:
            kws.append(k)
        if len(kws) >= limit:
            return kws

    return kws[:limit]


def year_payload_to_block(payload: dict, ref_kw_limit: int = 30) -> Tuple[str, List[str]]:
    """
    LLM 입력 블록 문자열 + 참고키워드 리스트 반환(출력 섹션에도 재사용)
    """
    y = int(payload.get("Year") or payload.get("year") or 0)

    one_line = (payload.get("one_line_kr") or payload.get("one_line") or "").strip()
    note = (payload.get("final_note_kr") or payload.get("note") or "").strip()

    topics = payload.get("topics", []) or []
    comp = payload.get("compare", {}) or payload.get("compare_result", {}) or {}

    stable = ensure_str_list(comp.get("stable_top10", []) or comp.get("stable", []) or [])[:10]
    emerging = ensure_str_list(comp.get("emerging_top10", []) or comp.get("emerging", []) or [])[:10]

    topic_lines = []
    if isinstance(topics, list):
        for t in topics[:5]:
            if not isinstance(t, dict):
                continue
            kr = (t.get("kr_name") or t.get("topic_kr") or t.get("name_kr") or "").strip()
            en = (t.get("en_name") or t.get("topic_en") or t.get("name_en") or "").strip()
            kws = ensure_str_list(t.get("keywords", []) or [])[:10]
            head = f"{kr}/{en}".strip("/")
            if not head:
                head = "(topic)"
            if kws:
                topic_lines.append(f"- {head}: {', '.join(kws)}")
            else:
                topic_lines.append(f"- {head}")

    ref_kws = extract_ref_keywords(payload, limit=ref_kw_limit)

    lines = []
    lines.append(f"[YEAR {y}]")
    if one_line:
        lines.append(f"- one_line: {one_line}")
    if note:
        lines.append(f"- note: {note}")
    if topic_lines:
        lines.append("- topics:")
        lines.extend(topic_lines)
    if stable:
        lines.append(f"- stable_top10: {', '.join(stable)}")
    if emerging:
        lines.append(f"- emerging_top10: {', '.join(emerging)}")
    if ref_kws:
        lines.append(f"- ref_keywords({len(ref_kws)}): {', '.join(ref_kws)}")

    return "\n".join(lines).strip(), ref_kws


PROMPT_CROSSYEAR_NARRATIVE_KR = """너는 "연도별 트렌드 변화"를 서술형으로 정리하는 분석가다.
아래는 동일 중분류(class)의 2021~2025 연도별 결과 요약 블록이다.

요구사항(매우 중요):
- 출력은 반드시 '자연스러운 한국어 줄글'로 작성한다. 표/리스트/불릿/번호매기기 금지.
- 시스템/구현 용어(예: fallback, 파싱 실패, 입력 신호 부족, 파일 없음)를 절대 쓰지 마라.
- 각 연도의 핵심을 1~2문장으로 이어서 서술하고,
  2021→2022→2023→2024→2025로 넘어갈 때 어떤 변화가 있었는지(관심 축 이동, 등장/확대/감소)도 문장으로 자연스럽게 녹여라.
- 마지막에는 3~5문장으로 "총평(흐름)"을 작성한다.
- 키워드/토픽명은 입력 블록에 등장한 것만 사용한다(새 키워드 생성 금지).
- 과장 금지, 근거 없는 단정 금지. 입력에 있는 단서에 기반해 해석하라.

중분류: {class_name}

연도별 입력 블록:
{blocks}
"""


def build_index(out_parallel_dir: str, debug: bool = False) -> Dict[Tuple[int, str], dict]:
    files = find_all_json_like_files(out_parallel_dir)

    if debug:
        print(f"[DEBUG] found json-like files: {len(files)}")
        for p in files[:20]:
            print(f"[DEBUG] sample: {p}")

    idx: Dict[Tuple[int, str], dict] = {}
    src: Dict[Tuple[int, str], str] = {}
    skipped_no_year = 0
    skipped_no_class = 0
    loaded_records = 0

    for path in files:
        lower = path.lower()
        if lower.endswith(".jsonl"):
            for payload in iter_json_records_from_jsonl(path):
                year, cls = pick_year_class(payload, path)
                if year is None:
                    skipped_no_year += 1
                    continue
                if not cls:
                    skipped_no_class += 1
                    continue

                key = (year, cls)
                pri = "return_final" if is_return_final_path(path) else "other"
                if key not in idx:
                    idx[key] = payload
                    src[key] = pri
                else:
                    if pri == "return_final" and src.get(key) != "return_final":
                        idx[key] = payload
                        src[key] = pri
                loaded_records += 1
        else:
            payload = load_json_file(path)
            if not payload or not isinstance(payload, dict):
                continue
            year, cls = pick_year_class(payload, path)
            if year is None:
                skipped_no_year += 1
                continue
            if not cls:
                skipped_no_class += 1
                continue

            key = (year, cls)
            pri = "return_final" if is_return_final_path(path) else "other"
            if key not in idx:
                idx[key] = payload
                src[key] = pri
            else:
                if pri == "return_final" and src.get(key) != "return_final":
                    idx[key] = payload
                    src[key] = pri
            loaded_records += 1

    if debug:
        print(f"[DEBUG] loaded_records(total seen): {loaded_records}")
        print(f"[DEBUG] indexed keys: {len(idx)}")
        print(f"[DEBUG] skipped_no_year: {skipped_no_year}, skipped_no_class: {skipped_no_class}")

    return idx


def debug_counts(idx: Dict[Tuple[int, str], dict], years: List[int]) -> str:
    by_year = {y: 0 for y in years}
    for (y, _cls) in idx.keys():
        if y in by_year:
            by_year[y] += 1
    return " | ".join([f"{y}:{by_year[y]}" for y in years])


def run(
    out_parallel_dir: str,
    out_dir: str,
    years: List[int],
    model: str,
    max_tokens: int,
    sleep_sec: float,
    preview: bool,
    retry: int,
    debug: bool,
    ref_kw_limit: int,
):
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY 환경변수가 없습니다. export OPENAI_API_KEY=...")

    client = OpenAI()

    idx = build_index(out_parallel_dir, debug=debug)

    if debug:
        print(f"[INFO] indexed counts by year: {debug_counts(idx, years)}")

    if not idx:
        files = find_all_json_like_files(out_parallel_dir)
        msg = [
            "JSON을 하나도 못 읽었습니다.",
            f"- out_parallel_dir: {out_parallel_dir}",
            f"- 재귀 탐색으로 찾은 *.json/*.jsonl 파일 수: {len(files)}",
            "=> 파일 수가 0이면 경로가 틀린 겁니다.",
            "=> 파일 수가 있는데도 index가 0이면 JSON 안에 Year/NODE_CLSS_02 필드명이 다를 수 있습니다.",
        ]
        raise RuntimeError("\n".join(msg))

    os.makedirs(out_dir, exist_ok=True)
    raw_dir = os.path.join(out_dir, "_raw")
    os.makedirs(raw_dir, exist_ok=True)

    classes = sorted(set([cls for (_y, cls) in idx.keys()]))

    all_md_parts = []
    for cls in classes:
        blocks = []
        used_years = []
        ref_by_year: Dict[int, List[str]] = {}

        for y in years:
            key = (y, cls)
            if key in idx:
                block, ref_kws = year_payload_to_block(idx[key], ref_kw_limit=ref_kw_limit)
                blocks.append(block)
                used_years.append(y)
                ref_by_year[y] = ref_kws

        if not blocks:
            continue

        prompt = PROMPT_CROSSYEAR_NARRATIVE_KR.format(
            class_name=cls,
            blocks="\n\n---\n\n".join(blocks),
        )

        text = ""
        raw_all = ""
        for attempt in range(1, retry + 1):
            try:
                text = call_llm(client, model, prompt, max_output_tokens=max_tokens)
                raw_all = text
                if text.strip():
                    break
            except Exception as e:
                raw_all = f"[ERR] attempt={attempt}: {e}"
                time.sleep(2 * attempt)

        if preview:
            print(f"\n----- PREVIEW class={cls} years={used_years} -----")
            print((strip_code_fences(text)[:900] + "...") if text else "(EMPTY)")
            print("----- END PREVIEW -----\n")

        text = strip_code_fences(text).strip()
        if not text:
            text = f"{cls} (연도 {used_years}) 서술 생성 실패. 입력 블록을 확인하세요.\n\n" + "\n\n".join(blocks)

        # ✅ 최종 출력: 줄글 + 연도별 참고키워드(추가 섹션)
        cls_safe = safe_filename(cls)
        md_path = os.path.join(out_dir, f"{cls_safe}_2021-2025_narrative.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# {cls} (2021~2025 변화)\n\n")
            f.write(text.strip() + "\n\n")
            f.write("## 연도별 참고 키워드\n")
            for y in used_years:
                kws = ref_by_year.get(y, [])
                if kws:
                    f.write(f"- {y}: {', '.join(kws)}\n")
                else:
                    f.write(f"- {y}: (없음)\n")
        print(f"[OK] {md_path}")

        raw_path = os.path.join(raw_dir, f"{cls_safe}_raw.txt")
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(raw_all or "")

        # ALL 파일에도 같이 합치기
        sec = [f"# {cls} (2021~2025 변화)", "", text.strip(), "", "## 연도별 참고 키워드"]
        for y in used_years:
            kws = ref_by_year.get(y, [])
            sec.append(f"- {y}: {', '.join(kws) if kws else '(없음)'}")
        all_md_parts.append("\n".join(sec).strip() + "\n")

        time.sleep(sleep_sec)

    if all_md_parts:
        all_path = os.path.join(out_dir, "ALL_classes_2021-2025_narratives.md")
        with open(all_path, "w", encoding="utf-8") as f:
            f.write("\n\n---\n\n".join(all_md_parts).strip() + "\n")
        print(f"[OK] {all_path}")

    print("\nDONE")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out_parallel_dir", required=True, help="out_parallel 폴더")
    p.add_argument("--out_dir", required=True, help="출력 폴더")
    p.add_argument("--years", default="2021,2022,2023,2024,2025", help="쉼표로 구분")
    p.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    p.add_argument("--max_tokens", type=int, default=1800)
    p.add_argument("--sleep", type=float, default=0.2)
    p.add_argument("--preview", action="store_true")
    p.add_argument("--retry", type=int, default=3)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--ref_kw_limit", type=int, default=30, help="연도별 참고키워드 최대 개수")
    args = p.parse_args()

    years = [int(x.strip()) for x in args.years.split(",") if x.strip()]

    run(
        out_parallel_dir=args.out_parallel_dir,
        out_dir=args.out_dir,
        years=years,
        model=args.model,
        max_tokens=args.max_tokens,
        sleep_sec=args.sleep,
        preview=args.preview,
        retry=args.retry,
        debug=args.debug,
        ref_kw_limit=args.ref_kw_limit,
    )
