#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_reports_parallel_conductor.py

정책(요구사항 반영):
1) validator issue가 "처음에 있었던 중분류"만 Conductor final 생성/저장
   - out_dir/rerun_final/<year>/<중분류>_final.json
   - out_dir/rerun_final/<year>/<중분류>_final.md
2) 처음부터 issue가 없던 중분류는 "요약본"만 저장
   - out_dir/ok_summary/<year>/<중분류>_summary.json
   - out_dir/ok_summary/<year>/<중분류>_summary.md
3) meta는 항상 저장(추적/검증/로그)
   - out_dir/_meta/<year>/<중분류>_meta.json
   - out_dir/<year>_index.csv (연도별 전체 상태 인덱스)
4) 연도별 최종 트렌드 분석용 통합 코퍼스 자동 생성(JSONL/CSV)
   - out_dir/<year>_final_corpus.jsonl
   - out_dir/<year>_final_corpus.csv

추가(디버깅/검증 강화):
5) Agent A/B/C 및 Conductor plan의 "파싱된 JSON"과 "원문(raw)" 저장
   - out_dir/_agents/<year>/<중분류>_A_r0.json ...
   - out_dir/_raw/<year>/<중분류>_A_r0.txt ...

핵심 수정:
- 프롬프트 문자열에 JSON 예시 { } 가 들어가면 .format()에서 KeyError 발생
  => .format() 완전 제거 + fill_prompt()로 안전 치환
- evidence_top3가 dict list가 아닌 string list로 올 수 있음
  => Markdown 변환/Validator에서 dict/str 모두 안전 처리
- freq 입력은 merged top100이 아니라 {year}_freq_merged_all.csv를 기본 사용
"""

import os
import re
import json
import time
import argparse
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import ast


# =========================
# 안전한 템플릿 치환 (format() 금지)
# =========================
def fill_prompt(tpl: str, **kwargs) -> str:
    """
    {csv_text} 같은 placeholder만 단순 치환.
    JSON 예시의 { }는 건드리지 않아서 KeyError 방지.
    """
    for k, v in kwargs.items():
        tpl = tpl.replace("{" + k + "}", v)
    return tpl


def safe_filename(s: str) -> str:
    s = str(s)
    return "".join(c if c.isalnum() or c in " _-." else "_" for c in s).strip()


def df_to_csv_text(df: pd.DataFrame, limit_rows: int = 160) -> str:
    if df is None or df.empty:
        return ""
    if len(df) > limit_rows:
        df = df.head(limit_rows)
    return df.to_csv(index=False)


def parse_json_loose(text: str) -> dict:
    """
    LLM이 앞뒤에 설명/코드펜스를 붙여도 최대한 JSON만 뽑아냄
    - strict json.loads
    - 실패하면 ast.literal_eval(싱글쿼트/True/False 대응)
    """
    if not text:
        return {}
    text = re.sub(r"```[a-zA-Z]*\n?", "", text).replace("```", "").strip()

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return {}
    chunk = m.group(0).strip()

    try:
        return json.loads(chunk)
    except Exception:
        pass

    try:
        obj = ast.literal_eval(chunk)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    return {}


# =========================
# OpenAI 호출
# =========================
def call_llm(client: OpenAI, model: str, prompt: str, max_output_tokens: int = 1400) -> str:
    resp = client.responses.create(
        model=model,
        input=prompt,
        max_output_tokens=max_output_tokens,
    )
    if getattr(resp, "output_text", None):
        return (resp.output_text or "").strip()

    parts: List[str] = []
    for item in getattr(resp, "output", []) or []:
        for c in getattr(item, "content", []) or []:
            if getattr(c, "type", "") in ("output_text", "text"):
                parts.append(getattr(c, "text", "") or "")
    return "\n".join([p for p in parts if p]).strip()


# =========================
# 프롬프트: 에이전트 A/B/C
# =========================
PROMPT_AGENT_A_TOPIC = """너는 공학 분야 학술 트렌드 분석가다(Agent A).
아래 CSV(trend mix)는 특정 연도/중분류의 키워드 후보 목록이다.
반드시 CSV의 keyword 컬럼에 있는 값만 사용하라. 새 키워드 생성 금지.

요구:
- 토픽 최대 5개. 부족하면 1~4개로 줄여도 됨(억지 금지).
- 각 토픽은 (kr_name, en_name, keywords[6~10], evidence_top3 예: [{"keyword":"...","burst":0.0123}, ...]) 포함.
- 마지막에 one_line_kr, candidate_stats(total_rows, unique_keywords, positive_burst_keywords, concentration_hint), note_kr 포함.

출력은 JSON만(설명 금지).
JSON 스키마:
{
  "class_name": "...",
  "topics": [
    {
      "kr_name": "...",
      "en_name": "...",
      "keywords": ["k1","k2",...],
      "evidence_top3": [{"keyword":"...","burst":0.0123}, ...],
      "summary_kr": "..."
    }
  ],
  "one_line_kr": "...",
  "candidate_stats": {
    "total_rows": 0,
    "unique_keywords": 0,
    "positive_burst_keywords": 0,
    "concentration_hint": "상위 키워드 쏠림/다양/희소 등"
  },
  "note_kr": "토픽 수가 줄었다면 이유를 1문장"
}

CSV:
{csv_text}
"""

PROMPT_AGENT_B_COMPARE = """너는 공학 분야 학술 트렌드 분석가다(Agent B).
아래 CSV A는 freq(누적 관심사), CSV B는 trend mix(변화/급증) 후보이다.
반드시 두 CSV의 keyword 컬럼에 있는 키워드만 사용하라. 새 키워드 생성 금지.

요구:
- stable_top10: A에는 있고 B에는 없는 키워드 10개(가능하면 freq 큰 순)
- emerging_top10: B에는 있고 A에는 없는 키워드 10개(가능하면 burst>0 우선)
- comparison_3lines: 3줄(각 1문장) 요약
- change_point_kr: 1문장

출력은 JSON만.
스키마:
{
  "class_name": "...",
  "stable_top10": ["..."],
  "emerging_top10": ["..."],
  "comparison_3lines": ["...","...","..."],
  "change_point_kr": "..."
}

CSV A (freq):
{csv_freq}

CSV B (trend mix):
{csv_trend}
"""

PROMPT_AGENT_C_DIAG = """너는 품질 점검 담당자다(Agent C).
아래 CSV(trend mix)를 보고 '토픽이 적게 나올 위험'과 '키워드 분포 문제'를 진단하라.
반드시 CSV의 값만 근거로 하라.

출력은 JSON만.
스키마:
{
  "class_name":"...",
  "risk_flags": {
    "low_positive_burst": true/false,
    "high_concentration": true/false,
    "too_many_noise_tokens": true/false
  },
  "observations_kr": ["...","...","..."],
  "recommendations_kr": ["...","...","..."]
}

CSV:
{csv_text}
"""


# =========================
# 프롬프트: Conductor (rerun plan)
# =========================
PROMPT_CONDUCTOR_PLAN = """너는 여러 에이전트 결과를 통합/검증하는 Conductor다.
아래 입력에는 (A=topic, B=compare, C=diagnosis) 결과(JSON)와 validator 이슈 목록이 있다.

너의 역할:
1) 어떤 에이전트를 다시 실행할지 결정한다(최대 1개 agent만 rerun 권장).
2) rerun한다면, 구체적인 수정 지시를 만든다(키워드 규칙/토픽 수/근거 강화 등).
3) rerun이 불필요하면 "rerun_agents":[] 로 둔다.

출력은 JSON만.
스키마:
{
  "rerun_agents": ["A"|"B"|"C" ...],
  "instructions": {
    "A": {
      "focus_kr": "...",
      "topic_max": 5,
      "min_keywords_per_topic": 6,
      "allow_reduce_topics": true
    },
    "B": {
      "focus_kr": "..."
    },
    "C": {
      "focus_kr": "..."
    }
  },
  "stop_reason_kr": "왜 rerun 하거나 안 하는지 1문장"
}

A_JSON:
{a_json}

B_JSON:
{b_json}

C_JSON:
{c_json}

VALIDATOR_ISSUES:
{issues}
"""


# =========================
# 프롬프트: Conductor (final merge)
# =========================
PROMPT_CONDUCTOR_FINAL = """너는 Conductor다. 아래 A/B/C 결과를 바탕으로 최종 리포트를 만든다.
반드시 keyword는 입력된 A/B 결과 내 keyword들만 사용하라(새 키워드 생성 금지).

출력은 JSON만.
스키마:
{
  "class_name":"...",
  "final_topics": [... A의 topics를 기반으로 정리 ...],
  "final_compare": {... B 기반 ...},
  "final_note_kr":"희소성/쏠림/해석 주의사항 1~2문장",
  "conductor_confidence": 0.0
}

A_JSON:
{a_json}

B_JSON:
{b_json}

C_JSON:
{c_json}
"""


# =========================
# evidence_top3 안전 문자열화 (dict list / str list 둘 다 대응)
# =========================
def format_evidence_list(ev_list: Any) -> str:
    """
    evidence_top3가
    - [{"keyword":"...", "burst":0.01}, ...] (dict list)
    - ["kw1","kw2", ...] (str list)
    - "kw" (str)
    이런 식으로 와도 안전하게 문자열로 변환
    """
    if not ev_list:
        return ""

    if isinstance(ev_list, (str, int, float)):
        return str(ev_list)

    out = []
    if isinstance(ev_list, list):
        for e in ev_list:
            if isinstance(e, dict):
                kw = str(e.get("keyword", "") or "").strip()
                b = e.get("burst", None)
                if kw:
                    try:
                        bb = float(b) if b is not None else None
                    except Exception:
                        bb = None
                    if bb is None:
                        out.append(f"{kw}")
                    else:
                        out.append(f"{kw} (burst={bb:.4f})")
            else:
                kw = str(e).strip()
                if kw:
                    out.append(kw)

    return ", ".join(out)


def ensure_str_list(x: Any) -> List[str]:
    if not x:
        return []
    if isinstance(x, list):
        return [str(i) for i in x if str(i).strip()]
    return [str(x).strip()] if str(x).strip() else []


# =========================
# Validator
# =========================
def validate_outputs(a: dict, b: dict, c: dict, allowed_keywords: set[str]) -> list[str]:
    issues: List[str] = []

    # A 검사
    if not a or "topics" not in a:
        issues.append("A: topics 누락 또는 JSON 파싱 실패")
    else:
        topics = a.get("topics", []) or []
        if not isinstance(topics, list):
            issues.append("A: topics 타입 이상(list 아님)")
            topics = []

        if len(topics) == 0:
            issues.append("A: topics=0")
        if len(topics) > 5:
            issues.append("A: topics가 5개 초과")

        for i, t in enumerate(topics, start=1):
            if not isinstance(t, dict):
                issues.append(f"A: Topic{i}가 dict 아님")
                continue

            kws = t.get("keywords", []) or []
            kws = ensure_str_list(kws)

            if len(kws) < 4:
                issues.append(f"A: Topic{i} keywords 너무 적음({len(kws)})")

            bad = [k for k in kws if k not in allowed_keywords]
            if bad:
                issues.append(f"A: Topic{i} CSV에 없는 키워드 사용: {bad[:5]}")

            ev = t.get("evidence_top3", []) or []
            if not ev:
                issues.append(f"A: Topic{i} evidence_top3 누락")
            else:
                # dict list / str list 모두 안전 처리
                if isinstance(ev, list):
                    for e in ev:
                        if isinstance(e, dict):
                            kk = str(e.get("keyword", "") or "").strip()
                        else:
                            kk = str(e).strip()
                        if kk and kk not in allowed_keywords:
                            issues.append(f"A: evidence에 CSV 없는 키워드: {kk}")

    # B 검사
    if not b or "stable_top10" not in b:
        issues.append("B: stable_top10 누락 또는 JSON 파싱 실패")
    else:
        stable = ensure_str_list((b.get("stable_top10", []) or []))
        emerging = ensure_str_list((b.get("emerging_top10", []) or []))
        keys = stable + emerging
        for key in keys:
            if key and key not in allowed_keywords:
                issues.append(f"B: CSV에 없는 키워드 사용: {key}")

    # C 검사(형식만)
    if not c or "risk_flags" not in c:
        issues.append("C: risk_flags 누락 또는 JSON 파싱 실패")

    return issues


# =========================
# 에이전트 출력 저장 헬퍼
# =========================
def save_text(path: Optional[str], text: str) -> None:
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text or "")
    except Exception:
        pass


def save_json(path: Optional[str], payload: Any) -> None:
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, indent=2))
    except Exception:
        pass


# =========================
# 에이전트 실행 함수
# =========================
def _sort_for_prompt(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    x = df.copy()

    # burst 우선 정렬
    if "burst" in x.columns:
        x["burst"] = pd.to_numeric(x["burst"], errors="coerce").fillna(0.0)
    if "freq" in x.columns:
        x["freq"] = pd.to_numeric(x["freq"], errors="coerce").fillna(0).astype(int)

    if "burst" in x.columns and "freq" in x.columns:
        x = x.sort_values(["burst", "freq"], ascending=[False, False])
    elif "freq" in x.columns:
        x = x.sort_values(["freq"], ascending=[False])
    return x


def run_agent_A(
    client,
    model,
    cls,
    trend_df_sub,
    max_tokens,
    extra_inst: str = "",
    raw_save_path: Optional[str] = None,
    json_save_path: Optional[str] = None,
) -> dict:
    trend_df_sub = _sort_for_prompt(trend_df_sub)

    cols = [c for c in ["Year", "NODE_CLSS_02", "keyword", "freq", "paper_count", "share",
                       "share_prev", "burst", "source", "min_freq_used", "merged_variants"]
            if c in trend_df_sub.columns]
    csv_text = df_to_csv_text(trend_df_sub[cols] if cols else trend_df_sub)

    prompt = fill_prompt(PROMPT_AGENT_A_TOPIC, csv_text=csv_text)
    if extra_inst:
        prompt += "\n\n[추가 지시]\n" + extra_inst.strip()

    raw = call_llm(client, model, prompt, max_output_tokens=max_tokens)
    save_text(raw_save_path, raw)
    parsed = parse_json_loose(raw)
    save_json(json_save_path, parsed)
    return parsed


def run_agent_B(
    client,
    model,
    cls,
    freq_sub,
    trend_sub,
    max_tokens,
    extra_inst: str = "",
    raw_save_path: Optional[str] = None,
    json_save_path: Optional[str] = None,
) -> dict:
    freq_sub = _sort_for_prompt(freq_sub)
    trend_sub = _sort_for_prompt(trend_sub)

    cols_f = [c for c in ["Year", "NODE_CLSS_02", "keyword", "freq", "paper_count", "share", "merged_variants"]
              if c in freq_sub.columns]
    cols_t = [c for c in ["Year", "NODE_CLSS_02", "keyword", "freq", "paper_count", "share",
                         "share_prev", "burst", "source", "min_freq_used", "merged_variants"]
              if c in trend_sub.columns]
    csv_freq = df_to_csv_text(freq_sub[cols_f] if cols_f else freq_sub)
    csv_trend = df_to_csv_text(trend_sub[cols_t] if cols_t else trend_sub)

    prompt = fill_prompt(PROMPT_AGENT_B_COMPARE, csv_freq=csv_freq, csv_trend=csv_trend)
    if extra_inst:
        prompt += "\n\n[추가 지시]\n" + extra_inst.strip()

    raw = call_llm(client, model, prompt, max_output_tokens=max_tokens)
    save_text(raw_save_path, raw)
    parsed = parse_json_loose(raw)
    save_json(json_save_path, parsed)
    return parsed


def run_agent_C(
    client,
    model,
    cls,
    trend_df_sub,
    max_tokens,
    extra_inst: str = "",
    raw_save_path: Optional[str] = None,
    json_save_path: Optional[str] = None,
) -> dict:
    trend_df_sub = _sort_for_prompt(trend_df_sub)

    cols = [c for c in ["Year", "NODE_CLSS_02", "keyword", "freq", "paper_count", "share",
                       "share_prev", "burst", "source", "min_freq_used", "merged_variants"]
            if c in trend_df_sub.columns]
    csv_text = df_to_csv_text(trend_df_sub[cols] if cols else trend_df_sub)

    prompt = fill_prompt(PROMPT_AGENT_C_DIAG, csv_text=csv_text)
    if extra_inst:
        prompt += "\n\n[추가 지시]\n" + extra_inst.strip()

    raw = call_llm(client, model, prompt, max_output_tokens=max_tokens)
    save_text(raw_save_path, raw)
    parsed = parse_json_loose(raw)
    save_json(json_save_path, parsed)
    return parsed


def run_conductor_plan(
    client,
    model,
    a,
    b,
    c,
    issues,
    max_tokens,
    raw_save_path: Optional[str] = None,
    json_save_path: Optional[str] = None,
) -> dict:
    prompt = fill_prompt(
        PROMPT_CONDUCTOR_PLAN,
        a_json=json.dumps(a, ensure_ascii=False),
        b_json=json.dumps(b, ensure_ascii=False),
        c_json=json.dumps(c, ensure_ascii=False),
        issues=json.dumps(issues, ensure_ascii=False),
    )
    raw = call_llm(client, model, prompt, max_output_tokens=max_tokens)
    save_text(raw_save_path, raw)
    parsed = parse_json_loose(raw)
    save_json(json_save_path, parsed)
    return parsed


def run_conductor_final(client, model, a, b, c, max_tokens) -> Tuple[dict, str]:
    prompt = fill_prompt(
        PROMPT_CONDUCTOR_FINAL,
        a_json=json.dumps(a, ensure_ascii=False),
        b_json=json.dumps(b, ensure_ascii=False),
        c_json=json.dumps(c, ensure_ascii=False),
    )

    raw1 = call_llm(client, model, prompt, max_output_tokens=max_tokens)
    d1 = parse_json_loose(raw1)

    # 파싱 실패/키 누락이면 1회 강제 재시도
    if (not d1) or ("final_topics" not in d1) or ("final_compare" not in d1):
        hard = prompt + "\n\nIMPORTANT: 반드시 스키마를 만족하는 '유효한 JSON 객체'만 출력. 설명/코드펜스/추가텍스트 금지."
        raw2 = call_llm(client, model, hard, max_output_tokens=max_tokens)
        d2 = parse_json_loose(raw2)
        return d2, (raw1 + "\n\n---RETRY---\n\n" + raw2)

    return d1, raw1


# =========================
# Markdown 변환
# =========================
def final_json_to_md(year: int, cls: str, out: dict) -> str:
    lines: List[str] = []
    lines.append(f"[중분류] {cls} ({year})")
    lines.append("")

    topics = out.get("final_topics", []) or []
    if not isinstance(topics, list):
        topics = []

    for i, t in enumerate(topics, start=1):
        if not isinstance(t, dict):
            continue
        lines.append(f"- Topic {i}: {t.get('kr_name','')} / {t.get('en_name','')}")
        lines.append(f"  - Keywords: {', '.join(ensure_str_list(t.get('keywords', []) or []))}")

        ev_str = format_evidence_list(t.get("evidence_top3", []) or [])
        lines.append(f"  - Evidence(burst): {ev_str}")
        lines.append(f"  - Summary: {t.get('summary_kr','')}")
        lines.append("")

    comp = out.get("final_compare", {}) or {}
    if isinstance(comp, dict) and comp:
        lines.append("- Stable(Top10): " + ", ".join(ensure_str_list(comp.get("stable_top10", []) or [])))
        lines.append("- Emerging(Top10): " + ", ".join(ensure_str_list(comp.get("emerging_top10", []) or [])))
        lines.append("- 3-line comparison:")
        for j, s in enumerate(ensure_str_list(comp.get("comparison_3lines", []) or []), start=1):
            lines.append(f"  {j}) {s}")
        lines.append("- Change point: " + (comp.get("change_point_kr", "") or ""))
        lines.append("")

    lines.append("- Note: " + (out.get("final_note_kr", "") or ""))
    lines.append(f"- Conductor confidence: {out.get('conductor_confidence', 0.0)}")
    return "\n".join(lines).strip() + "\n"


def ok_summary_to_md(year: int, cls: str, a: dict, b: dict) -> str:
    lines: List[str] = []
    lines.append(f"[중분류] {cls} ({year})")
    lines.append("")

    one = (a or {}).get("one_line_kr", "") or ""
    if one:
        lines.append(f"- One-line: {one}")
        lines.append("")

    topics = (a or {}).get("topics", []) or []
    if not isinstance(topics, list):
        topics = []

    for i, t in enumerate(topics, start=1):
        if not isinstance(t, dict):
            continue
        lines.append(f"- Topic {i}: {t.get('kr_name','')} / {t.get('en_name','')}")
        lines.append(f"  - Keywords: {', '.join(ensure_str_list(t.get('keywords', []) or []))}")

        ev_str = format_evidence_list(t.get("evidence_top3", []) or [])
        if ev_str:
            lines.append(f"  - Evidence(burst): {ev_str}")

        lines.append(f"  - Summary: {t.get('summary_kr','')}")
        lines.append("")

    comp = b or {}
    if isinstance(comp, dict) and comp:
        lines.append("- Stable(Top10): " + ", ".join(ensure_str_list(comp.get("stable_top10", []) or [])))
        lines.append("- Emerging(Top10): " + ", ".join(ensure_str_list(comp.get("emerging_top10", []) or [])))
        lines.append("- 3-line comparison:")
        for j, s in enumerate(ensure_str_list(comp.get("comparison_3lines", []) or []), start=1):
            lines.append(f"  {j}) {s}")
        lines.append("- Change point: " + (comp.get("change_point_kr", "") or ""))
        lines.append("")

    note = (a or {}).get("note_kr", "") or ""
    if note:
        lines.append(f"- Note: {note}")

    return "\n".join(lines).strip() + "\n"


# =========================
# 연도별 최종 코퍼스(JSONL/CSV) 생성
# =========================
def _read_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _join_list(x: Any, sep: str = " | ") -> str:
    if not x:
        return ""
    if isinstance(x, list):
        return sep.join([str(i) for i in x if str(i).strip()])
    return str(x)


def build_year_corpus(out_dir: str, year: int, preview: bool = False) -> None:
    ok_dir = os.path.join(out_dir, "ok_summary", str(year))
    issue_dir = os.path.join(out_dir, "rerun_final", str(year))
    meta_dir = os.path.join(out_dir, "_meta", str(year))

    records: List[Dict[str, Any]] = []

    # OK summaries
    if os.path.isdir(ok_dir):
        for fn in os.listdir(ok_dir):
            if not fn.endswith("_summary.json"):
                continue
            path = os.path.join(ok_dir, fn)
            payload = _read_json(path)
            cls = payload.get("class_name") or fn.replace("_summary.json", "")
            cls_safe = safe_filename(cls)

            a = payload.get("topic_result_A", {}) or {}
            b = payload.get("compare_result_B", {}) or {}
            c = payload.get("diagnosis_C", {}) or {}

            meta_path = os.path.join(meta_dir, f"{cls_safe}_meta.json")
            meta = _read_json(meta_path)

            topics = a.get("topics", []) or []
            comp = b or {}

            rec = {
                "Year": year,
                "NODE_CLSS_02": cls,
                "status": "ok",
                "rerun_taken": bool(meta.get("rerun_taken", False)),
                "rerun_rounds": int(meta.get("rerun_rounds", 0) or 0),
                "issues_initial": meta.get("issues_initial", []),
                "issues_after_loop": meta.get("issues_after_loop", []),
                "one_line_kr": a.get("one_line_kr", ""),
                "note_kr": a.get("note_kr", ""),
                "diagnosis_risk_flags": (c.get("risk_flags", {}) if isinstance(c, dict) else {}),
                "topics": topics,
                "compare": comp,
                "conductor_confidence": None,
                "final_note_kr": None,
                "source_file": os.path.relpath(path, out_dir),
            }
            records.append(rec)

    # ISSUE finals (Conductor outputs)
    if os.path.isdir(issue_dir):
        for fn in os.listdir(issue_dir):
            if not fn.endswith("_final.json"):
                continue
            path = os.path.join(issue_dir, fn)
            final = _read_json(path)
            cls = final.get("class_name") or fn.replace("_final.json", "")
            cls_safe = safe_filename(cls)

            meta_path = os.path.join(meta_dir, f"{cls_safe}_meta.json")
            meta = _read_json(meta_path)

            rec = {
                "Year": year,
                "NODE_CLSS_02": cls,
                "status": "issue",
                "rerun_taken": bool(meta.get("rerun_taken", True)),
                "rerun_rounds": int(meta.get("rerun_rounds", 0) or 0),
                "issues_initial": meta.get("issues_initial", []),
                "issues_after_loop": meta.get("issues_after_loop", []),
                "one_line_kr": None,
                "note_kr": None,
                "diagnosis_risk_flags": None,
                "topics": final.get("final_topics", []) or [],
                "compare": final.get("final_compare", {}) or {},
                "conductor_confidence": final.get("conductor_confidence", None),
                "final_note_kr": final.get("final_note_kr", ""),
                "source_file": os.path.relpath(path, out_dir),
            }
            records.append(rec)

    if not records:
        if preview:
            print(f"[WARN] No corpus records for year={year}")
        return

    # JSONL
    jsonl_path = os.path.join(out_dir, f"{year}_final_corpus.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as fw:
        for r in sorted(records, key=lambda x: str(x.get("NODE_CLSS_02", ""))):
            fw.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] {jsonl_path}")

    # CSV (class-level row)
    rows: List[Dict[str, Any]] = []
    for r in records:
        comp = r.get("compare", {}) or {}
        stable = ensure_str_list(comp.get("stable_top10", []) or [])
        emerging = ensure_str_list(comp.get("emerging_top10", []) or [])
        comp3 = ensure_str_list(comp.get("comparison_3lines", []) or [])

        kw_set: List[str] = []
        for t in (r.get("topics", []) or []):
            if not isinstance(t, dict):
                continue
            for k in ensure_str_list(t.get("keywords", []) or []):
                if k and k not in kw_set:
                    kw_set.append(k)

        rows.append({
            "Year": r.get("Year"),
            "NODE_CLSS_02": r.get("NODE_CLSS_02"),
            "status": r.get("status"),
            "rerun_taken": r.get("rerun_taken"),
            "rerun_rounds": r.get("rerun_rounds"),
            "issues_initial_count": len(r.get("issues_initial", []) or []),
            "issues_after_loop_count": len(r.get("issues_after_loop", []) or []),
            "one_line_kr": r.get("one_line_kr") or "",
            "final_note_kr": r.get("final_note_kr") or "",
            "conductor_confidence": r.get("conductor_confidence"),
            "keywords_flat": ";".join(kw_set),
            "stable_top10": ";".join(stable),
            "emerging_top10": ";".join(emerging),
            "comparison_3lines": _join_list(comp3, sep=" | "),
            "change_point_kr": comp.get("change_point_kr", "") or "",
            "topics_json": json.dumps(r.get("topics", []) or [], ensure_ascii=False),
            "compare_json": json.dumps(comp, ensure_ascii=False),
            "source_file": r.get("source_file", ""),
        })

    csv_path = os.path.join(out_dir, f"{year}_final_corpus.csv")
    pd.DataFrame(rows).sort_values(["Year", "NODE_CLSS_02"]).to_csv(
        csv_path, index=False, encoding="utf-8-sig"
    )
    print(f"[OK] {csv_path}")


# =========================
# 파일 찾기 헬퍼 (호환성)
# =========================
def find_first_existing(base_dir: str, candidates: List[str]) -> Optional[str]:
    for fn in candidates:
        p = os.path.join(base_dir, fn)
        if os.path.exists(p):
            return p
    return None


# =========================
# main runner
# =========================
def run(
    input_dir: str,
    out_dir: str,
    years: list[int],
    sleep_sec: float,
    max_tokens: int,
    worker_model: str,
    conductor_model: str,
    preview: bool,
    max_rounds: int,
    top_n: int,
    min_freq: int,
):
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY 환경변수가 없습니다. export OPENAI_API_KEY=...")

    client = OpenAI()
    os.makedirs(out_dir, exist_ok=True)

    for year in years:
        # ✅ freq는 merged_all을 기본으로 사용
        freq_path = find_first_existing(input_dir, [
            f"{year}_freq_merged_all.csv",
            f"{year}_freq_merged_top{top_n}_min{min_freq}.csv",
            f"{year}_freq_top{top_n}_min{min_freq}.csv",
            f"{year}_freq_merged_top100_min5.csv",
            f"{year}_freq_top100_min5.csv",
        ])

        # trend mix는 보통 topN 파일이 생성됨
        trend_path = find_first_existing(input_dir, [
            f"{year}_trend_mix_merged_top{top_n}_min{min_freq}.csv",
            f"{year}_trend_mix_top{top_n}_min{min_freq}.csv",
            f"{year}_trend_mix_merged_top100_min5.csv",
            f"{year}_trend_mix_top100_min5.csv",
        ])

        if not freq_path or not trend_path:
            print(f"[SKIP] {year} 입력 파일 없음")
            print("  - freq_path :", freq_path)
            print("  - trend_path:", trend_path)
            continue

        freq_df = pd.read_csv(freq_path)
        trend_df = pd.read_csv(trend_path)

        # 컬럼명 공백/BOM 대비
        freq_df.columns = freq_df.columns.astype(str).str.strip()
        trend_df.columns = trend_df.columns.astype(str).str.strip()

        if "NODE_CLSS_02" not in freq_df.columns or "NODE_CLSS_02" not in trend_df.columns:
            print(f"[SKIP] {year} NODE_CLSS_02 없음")
            continue

        classes = sorted(
            set(freq_df["NODE_CLSS_02"].dropna().unique().tolist())
            | set(trend_df["NODE_CLSS_02"].dropna().unique().tolist())
        )

        index_rows: List[Dict[str, Any]] = []

        # 디버그 저장 폴더
        raw_dir_year = os.path.join(out_dir, "_raw", str(year))
        agents_dir_year = os.path.join(out_dir, "_agents", str(year))
        os.makedirs(raw_dir_year, exist_ok=True)
        os.makedirs(agents_dir_year, exist_ok=True)

        for cls in classes:
            f_all = freq_df[freq_df["NODE_CLSS_02"] == cls].copy()
            t_mix = trend_df[trend_df["NODE_CLSS_02"] == cls].copy()
            if f_all.empty and t_mix.empty:
                continue

            allowed_keywords: set[str] = set()
            if not f_all.empty and "keyword" in f_all.columns:
                allowed_keywords |= set(f_all["keyword"].astype(str).tolist())
            if not t_mix.empty and "keyword" in t_mix.columns:
                allowed_keywords |= set(t_mix["keyword"].astype(str).tolist())

            # 폴더 준비
            ok_dir = os.path.join(out_dir, "ok_summary", str(year))
            issue_dir = os.path.join(out_dir, "rerun_final", str(year))
            meta_dir = os.path.join(out_dir, "_meta", str(year))
            os.makedirs(ok_dir, exist_ok=True)
            os.makedirs(issue_dir, exist_ok=True)
            os.makedirs(meta_dir, exist_ok=True)

            cls_safe = safe_filename(cls)

            # ---------- round 0: A/B/C 실행 ----------
            a = run_agent_A(
                client, worker_model, cls,
                t_mix if not t_mix.empty else f_all,
                max_tokens=max_tokens,
                raw_save_path=os.path.join(raw_dir_year, f"{cls_safe}_A_r0.txt"),
                json_save_path=os.path.join(agents_dir_year, f"{cls_safe}_A_r0.json"),
            )
            b = run_agent_B(
                client, worker_model, cls,
                f_all if not f_all.empty else t_mix,
                t_mix if not t_mix.empty else f_all,
                max_tokens=max_tokens,
                raw_save_path=os.path.join(raw_dir_year, f"{cls_safe}_B_r0.txt"),
                json_save_path=os.path.join(agents_dir_year, f"{cls_safe}_B_r0.json"),
            )
            c = run_agent_C(
                client, worker_model, cls,
                t_mix if not t_mix.empty else f_all,
                max_tokens=max_tokens,
                raw_save_path=os.path.join(raw_dir_year, f"{cls_safe}_C_r0.txt"),
                json_save_path=os.path.join(agents_dir_year, f"{cls_safe}_C_r0.json"),
            )

            issues0 = validate_outputs(a, b, c, allowed_keywords)
            issues = list(issues0)

            if preview:
                print(f"\n=== {year} / {cls} issues0 ===")
                print(issues0 if issues0 else "(no issues)")

            # ---------- conductor loop (issue가 있는 경우에만) ----------
            cur_round = 0
            rerun_taken = False
            plan_history: List[dict] = []

            while issues and cur_round < max_rounds:
                plan = run_conductor_plan(
                    client, conductor_model, a, b, c, issues,
                    max_tokens=max_tokens,
                    raw_save_path=os.path.join(raw_dir_year, f"{cls_safe}_PLAN_r{cur_round}.txt"),
                    json_save_path=os.path.join(agents_dir_year, f"{cls_safe}_PLAN_r{cur_round}.json"),
                )

                if isinstance(plan, dict) and plan:
                    plan_history.append(plan)
                else:
                    plan_history.append({"_parse_failed_or_empty": True, "plan": plan})

                rerun = (plan.get("rerun_agents", []) if isinstance(plan, dict) else []) or []

                if preview:
                    print(f"\n--- Conductor plan round={cur_round} ---")
                    if isinstance(plan, dict) and plan:
                        print(json.dumps(plan, ensure_ascii=False, indent=2))
                    else:
                        print("[WARN] Conductor plan parse failed or empty -> fallback by issues")
                        print("issues =", issues)

                # plan이 비었거나 rerun_agents가 없으면 issues 기반 강제 rerun
                if not rerun:
                    if any(x.startswith("A:") for x in issues):
                        rerun = ["A"]
                    elif any(x.startswith("B:") for x in issues):
                        rerun = ["B"]
                    elif any(x.startswith("C:") for x in issues):
                        rerun = ["C"]
                    else:
                        rerun = []

                if not rerun:
                    break

                rerun_taken = True
                agent = rerun[0]

                inst = ""
                try:
                    inst = (plan.get("instructions", {}).get(agent, {}).get("focus_kr", "") or "").strip()
                    if agent == "A":
                        tmax = plan.get("instructions", {}).get("A", {}).get("topic_max", 5)
                        mink = plan.get("instructions", {}).get("A", {}).get("min_keywords_per_topic", 6)
                        allow_reduce = plan.get("instructions", {}).get("A", {}).get("allow_reduce_topics", True)
                        inst = (inst + f"\n- topic_max={tmax}\n- min_keywords_per_topic={mink}\n- allow_reduce_topics={allow_reduce}").strip()
                except Exception:
                    inst = ""

                # rerun round tag
                rr = cur_round + 1  # rerun round index (1..)
                if agent == "A":
                    a = run_agent_A(
                        client, worker_model, cls,
                        t_mix if not t_mix.empty else f_all,
                        max_tokens=max_tokens,
                        extra_inst=inst,
                        raw_save_path=os.path.join(raw_dir_year, f"{cls_safe}_A_r{rr}.txt"),
                        json_save_path=os.path.join(agents_dir_year, f"{cls_safe}_A_r{rr}.json"),
                    )
                elif agent == "B":
                    b = run_agent_B(
                        client, worker_model, cls,
                        f_all if not f_all.empty else t_mix,
                        t_mix if not t_mix.empty else f_all,
                        max_tokens=max_tokens,
                        extra_inst=inst,
                        raw_save_path=os.path.join(raw_dir_year, f"{cls_safe}_B_r{rr}.txt"),
                        json_save_path=os.path.join(agents_dir_year, f"{cls_safe}_B_r{rr}.json"),
                    )
                elif agent == "C":
                    c = run_agent_C(
                        client, worker_model, cls,
                        t_mix if not t_mix.empty else f_all,
                        max_tokens=max_tokens,
                        extra_inst=inst,
                        raw_save_path=os.path.join(raw_dir_year, f"{cls_safe}_C_r{rr}.txt"),
                        json_save_path=os.path.join(agents_dir_year, f"{cls_safe}_C_r{rr}.json"),
                    )

                issues = validate_outputs(a, b, c, allowed_keywords)
                cur_round += 1
                time.sleep(sleep_sec)

            # ---------- 저장 정책 ----------
            had_issues_initial = bool(issues0)

            # 1) 이슈 없으면 -> 요약본만 ok_summary에 저장
            if not had_issues_initial:
                ok_payload = {
                    "class_name": cls,
                    "topic_result_A": a,
                    "compare_result_B": b,
                    "diagnosis_C": c,
                    "note_kr": "validator issues 없음 → 요약본만 저장",
                }

                out_json = os.path.join(ok_dir, f"{cls_safe}_summary.json")
                with open(out_json, "w", encoding="utf-8") as fw:
                    fw.write(json.dumps(ok_payload, ensure_ascii=False, indent=2))
                print(f"[OK] {out_json}")

                out_md = os.path.join(ok_dir, f"{cls_safe}_summary.md")
                md = ok_summary_to_md(year, cls, a, b)
                with open(out_md, "w", encoding="utf-8") as fw:
                    fw.write(md)
                print(f"[OK] {out_md}")

                status = "ok"

            # 2) 이슈 있으면 -> Conductor final만 issue_dir에 저장
            else:
                final, raw_final = run_conductor_final(client, conductor_model, a, b, c, max_tokens=max_tokens)

                save_text(os.path.join(raw_dir_year, f"{cls_safe}_CONDUCTOR_FINAL.txt"), raw_final or "")

                if (not final) or ("final_topics" not in final) or ("final_compare" not in final):
                    final = {
                        "class_name": cls,
                        "final_topics": (a.get("topics", []) or [])[:5] if isinstance(a, dict) else [],
                        "final_compare": b if isinstance(b, dict) else {},
                        "final_note_kr": "Conductor 최종 JSON 파싱 실패 → A/B 결과로 fallback 생성",
                        "conductor_confidence": 0.2
                    }

                out_json = os.path.join(issue_dir, f"{cls_safe}_final.json")
                with open(out_json, "w", encoding="utf-8") as fw:
                    fw.write(json.dumps(final, ensure_ascii=False, indent=2))
                print(f"[OK] {out_json}")

                out_md = os.path.join(issue_dir, f"{cls_safe}_final.md")
                md = final_json_to_md(year, cls, final)
                with open(out_md, "w", encoding="utf-8") as fw:
                    fw.write(md)
                print(f"[OK] {out_md}")

                status = "issue"

            # ---------- meta는 항상 저장 ----------
            meta = {
                "year": year,
                "class_name": cls,
                "cls_safe": cls_safe,
                "status": status,
                "had_issues_initial": had_issues_initial,
                "rerun_taken": rerun_taken,
                "rerun_rounds": cur_round,
                "issues_initial": issues0,
                "issues_after_loop": issues,
                "plan_history": plan_history,
                "models": {
                    "worker_model": worker_model,
                    "conductor_model": conductor_model,
                },
                "input_files": {
                    "freq_path": os.path.basename(freq_path),
                    "trend_path": os.path.basename(trend_path),
                },
                "debug_outputs": {
                    "agents_dir": os.path.relpath(agents_dir_year, out_dir),
                    "raw_dir": os.path.relpath(raw_dir_year, out_dir),
                }
            }
            out_meta = os.path.join(meta_dir, f"{cls_safe}_meta.json")
            with open(out_meta, "w", encoding="utf-8") as fw:
                fw.write(json.dumps(meta, ensure_ascii=False, indent=2))
            if preview:
                print(f"[META] {out_meta}")

            # ---------- 연도 인덱스 누적 ----------
            index_rows.append({
                "Year": year,
                "NODE_CLSS_02": cls,
                "status": status,
                "had_issues_initial": had_issues_initial,
                "rerun_taken": rerun_taken,
                "rerun_rounds": cur_round,
                "issues_initial_count": len(issues0),
                "issues_after_loop_count": len(issues),
            })

            time.sleep(sleep_sec)

        # 연도별 인덱스 저장
        if index_rows:
            idx_path = os.path.join(out_dir, f"{year}_index.csv")
            pd.DataFrame(index_rows).sort_values(["Year", "NODE_CLSS_02"]).to_csv(
                idx_path, index=False, encoding="utf-8-sig"
            )
            print(f"[OK] {idx_path}")

        # 연도별 통합 코퍼스 생성(JSONL/CSV)
        build_year_corpus(out_dir=out_dir, year=year, preview=preview)

    print("\nDONE")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True, help="merged 산출물 폴더")
    p.add_argument("--out_dir", default="./reports_parallel", help="출력 폴더")
    p.add_argument("--years", default="2021,2022,2023,2024,2025", help="쉼표로 구분")
    p.add_argument("--sleep", type=float, default=0.3)
    p.add_argument("--max_tokens", type=int, default=1400)
    p.add_argument("--worker_model", default=os.getenv("OPENAI_MODEL", "gpt-5-nano"))
    p.add_argument("--conductor_model", default=os.getenv("OPENAI_CONDUCTOR_MODEL", "gpt-5-mini"))
    p.add_argument("--preview", action="store_true")
    p.add_argument("--max_rounds", type=int, default=1, help="재시도 라운드(0~2 추천)")
    p.add_argument("--top_n", type=int, default=100, help="trend mix 파일 topN 값(파일명 매칭용)")
    p.add_argument("--min_freq", type=int, default=5, help="trend mix 파일 min_freq 값(파일명 매칭용)")
    args = p.parse_args()

    years = [int(x.strip()) for x in args.years.split(",") if x.strip()]

    run(
        input_dir=args.input_dir,
        out_dir=args.out_dir,
        years=years,
        sleep_sec=args.sleep,
        max_tokens=args.max_tokens,
        worker_model=args.worker_model,
        conductor_model=args.conductor_model,
        preview=args.preview,
        max_rounds=args.max_rounds,
        top_n=args.top_n,
        min_freq=args.min_freq,
    )
