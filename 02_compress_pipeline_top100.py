#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
02_compress_pipeline_top100.py

입력:
out_dir/keywords_by_year_class/{year}/{class}_keywords_raw.csv

동작:
1) 중분류별로 2021~2025 전체 키워드 유니온 수집
2) 중분류별 merge_map(variant->canonical)을 LLM으로 1회 생성(캐시 저장)
   - 너무 많은 키워드는 llm_max_keywords까지만 LLM에 넣고,
     나머지는 identity로 처리(과도한 입력 방지)
3) 각 (year,class) 파일에 merge_map 적용하여 freq 합산/merged_variants 생성
4) 전체 years를 합친 뒤 burst=share(Y)-share(Y-1) 계산
5) (year,class)별로
   - freq_top100
   - burst_top100 (burst>0)
   - mix_top100 (burst 먼저 + 부족하면 freq로 fill)
   저장

출력:
out_dir/compressed_by_year_class/{year}/{class}_keywords_merged_all.csv
out_dir/top100_by_year_class/{year}/{class}_freq_top100.csv
out_dir/top100_by_year_class/{year}/{class}_burst_top100.csv
out_dir/top100_by_year_class/{year}/{class}_mix_top100.csv

실행 예시:
export OPENAI_API_KEY="..."
export OPENAI_MODEL="gpt-5-nano"

python 02_compress_pipeline_top100.py \
  --base_dir "/Users/idonghyeon/Desktop/dathon_result/keywords_by_year_class" \
  --out_dir "/Users/idonghyeon/Desktop/dathon_result" \
  --years "2021,2022,2023,2024,2025" \
  --top_n 100 \
  --min_freq 5 \
  --llm_max_keywords 800 \
  --sleep 0.25
"""

import os
import re
import json
import time
import argparse
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from typing import Any, Dict, List
import ast


# -------------------------
# LLM Prompt (ONLY normalize)
# -------------------------
PROMPT_MERGE_MAP_ONLY_TSV = """너는 키워드 정규화 담당자다.
아래는 특정 중분류(class)의 키워드 목록이다.

표기만 다른 경우(대소문자/공백/하이픈/복수형),
한/영 번역쌍, 약어-풀네임, 명백한 오탈자만 병합하라.
과병합 금지(상위 개념으로 넓게 합치는 것 금지).

중요 규칙:
- canonical은 가능하면 영어(라틴문자) 키워드로 선택하라. (있으면 영어 우선)
- variant는 반드시 입력 목록에 실제로 존재하는 키워드만 써라.
- 모든 variant에 대해 canonical을 하나 지정하라. (병합하지 않는 경우 canonical=variant)
- 절대 코드블록(```) 쓰지 말고, 설명문 쓰지 말고, 오직 TSV 본문만 출력하라.
- 출력은 TSV(탭으로 구분)만 출력하라. (CSV 금지)

출력 TSV 헤더(반드시 그대로):
variant\tcanonical\treason\tconfidence

reason 값은 아래 중 하나:
casing|spacing|plural|typo|acronym|ko-en|same

입력(키워드 목록):
{kw_list}
"""

ALLOWED_REASON = {"casing", "spacing", "plural", "typo", "acronym", "ko-en", "same"}


# -------------------------
# utils
# -------------------------
def safe_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in " _-." else "_" for c in str(s)).strip()


def strip_code_fences(text: str) -> str:
    if not text:
        return ""
    lines = []
    for ln in text.splitlines():
        if ln.strip().startswith("```"):
            continue
        lines.append(ln)
    return "\n".join(lines).strip()


def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _is_float(x: str) -> bool:
    try:
        float(str(x).strip())
        return True
    except Exception:
        return False


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


def parse_merge_map_tsv_robust(text: str, keywords: List[str]) -> pd.DataFrame:
    cleaned = strip_code_fences(text or "")
    cleaned = cleaned.replace("\ufeff", "").strip()
    if not cleaned:
        return pd.DataFrame()

    lines = [ln.rstrip("\n") for ln in cleaned.splitlines() if str(ln).strip()]
    if not lines:
        return pd.DataFrame()

    # 헤더 제거
    if lines[0].lower().startswith("variant"):
        lines = lines[1:]

    records = []
    for raw in lines:
        line = str(raw).strip().rstrip("\t")
        parts = line.split("\t")
        while parts and parts[-1] == "":
            parts = parts[:-1]
        if len(parts) < 2:
            continue

        variant = parts[0].strip()
        canonical = parts[1].strip()
        reason = "same"
        conf = "1.0"

        if len(parts) >= 3 and parts[2].strip() in ALLOWED_REASON:
            reason = parts[2].strip()

        # confidence는 뒤에서 숫자 찾기
        for p in reversed(parts[2:]):
            if _is_float(p.strip()):
                cf = float(p.strip())
                if cf < 0:
                    conf = "0.0"
                elif cf > 1:
                    conf = "1.0"
                else:
                    conf = str(cf)
                break

        if reason not in ALLOWED_REASON:
            reason = "same"

        records.append({
            "variant": variant,
            "canonical": canonical,
            "reason": reason,
            "confidence": conf,
        })

    df = pd.DataFrame(records)
    if df.empty:
        return df

    df["variant"] = df["variant"].astype(str).str.strip()
    df["canonical"] = df["canonical"].astype(str).str.strip()
    df["variant_norm"] = df["variant"].map(_norm)
    df = df.drop_duplicates(subset=["variant_norm"], keep="first").copy()

    # 누락 키워드는 identity로 보완
    existing = set(df["variant_norm"].dropna().tolist())
    add_rows = []
    for k in keywords:
        n = _norm(k)
        if n not in existing:
            add_rows.append({
                "variant": k,
                "canonical": k,
                "reason": "same",
                "confidence": "1.0",
                "variant_norm": n,
            })
    if add_rows:
        df = pd.concat([df, pd.DataFrame(add_rows)], ignore_index=True)

    df = df.drop_duplicates(subset=["variant_norm"], keep="first").copy()
    return df


def build_merge_map_for_class(
    client: OpenAI,
    model: str,
    cls: str,
    keywords: List[str],
    cache_path: str,
    llm_max_keywords: int = 800,
    max_tokens: int = 1400,
    sleep_sec: float = 0.25,
    preview: bool = False,
) -> pd.DataFrame:
    """
    - keyword가 너무 많으면 상위 llm_max_keywords만 LLM에 투입
    - 나머지는 identity mapping으로 자동 보완
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # 캐시 재사용
    if os.path.exists(cache_path):
        mm = pd.read_csv(cache_path, sep="\t", dtype=str, engine="python")
        if "variant" in mm.columns and "canonical" in mm.columns:
            mm["variant"] = mm["variant"].astype(str).str.strip()
            mm["canonical"] = mm["canonical"].astype(str).str.strip()
            mm["variant_norm"] = mm["variant"].map(_norm)
            # 캐시에도 누락 보완
            existing = set(mm["variant_norm"].dropna().tolist())
            add_rows = []
            for k in keywords:
                n = _norm(k)
                if n not in existing:
                    add_rows.append({
                        "variant": k, "canonical": k, "reason": "same", "confidence": "1.0", "variant_norm": n
                    })
            if add_rows:
                mm = pd.concat([mm, pd.DataFrame(add_rows)], ignore_index=True)
            mm = mm.drop_duplicates(subset=["variant_norm"], keep="first").copy()
            return mm

    # LLM에 넣을 키워드 제한
    kw_for_llm = keywords[:llm_max_keywords]

    kw_list = "\n".join([k for k in kw_for_llm if str(k).strip()])
    prompt = PROMPT_MERGE_MAP_ONLY_TSV.replace("{kw_list}", kw_list)

    raw = call_llm(client, model, prompt, max_output_tokens=max_tokens)
    if preview:
        print(f"\n--- merge_map preview [{cls}] ---")
        print((raw[:900] + "...") if raw else "(EMPTY)")

    mm = parse_merge_map_tsv_robust(raw, keywords=keywords)

    # 파싱 실패 시 identity
    if mm.empty:
        mm = pd.DataFrame({
            "variant": keywords,
            "canonical": keywords,
            "reason": ["same"] * len(keywords),
            "confidence": ["1.0"] * len(keywords),
        })
        mm["variant_norm"] = mm["variant"].map(_norm)

    # 저장 (variant_norm 제외)
    mm_save = mm.drop(columns=["variant_norm"], errors="ignore").copy()
    mm_save.to_csv(cache_path, index=False, sep="\t", encoding="utf-8-sig")
    print(f"[OK] merge_map saved: {cache_path}")

    time.sleep(sleep_sec)
    return mm


def apply_merge_map_to_year_class(df: pd.DataFrame, mm: pd.DataFrame) -> pd.DataFrame:
    """
    df columns: Year, NODE_CLSS_02, keyword, freq, paper_count, share
    mm columns: variant_norm -> canonical
    """
    x = df.copy()
    x["keyword"] = x["keyword"].astype(str).str.strip()
    x["keyword_norm"] = x["keyword"].map(_norm)

    mp = dict(zip(mm["variant_norm"], mm["canonical"]))
    x["canonical"] = x["keyword_norm"].map(lambda z: mp.get(z, None))
    x["canonical"] = x["canonical"].fillna(x["keyword"])

    # merged_variants
    merged_variants = (
        x.groupby("canonical")["keyword"]
         .apply(lambda s: ";".join(sorted(set([str(i) for i in s if str(i).strip()]))))
         .reset_index()
         .rename(columns={"keyword": "merged_variants"})
    )

    x["freq"] = pd.to_numeric(x["freq"], errors="coerce").fillna(0).astype(int)
    x["paper_count"] = pd.to_numeric(x["paper_count"], errors="coerce").fillna(0).astype(int)

    out = (
        x.groupby(["Year", "NODE_CLSS_02", "canonical"], as_index=False)
         .agg(freq=("freq", "sum"), paper_count=("paper_count", "max"))
         .rename(columns={"canonical": "keyword"})
    )

    out = out.merge(
        merged_variants.rename(columns={"canonical": "keyword"}),
        on="keyword",
        how="left"
    )

    out["share"] = out["freq"] / out["paper_count"].replace(0, pd.NA)
    out["share"] = out["share"].fillna(0.0)
    return out


def recalc_burst(all_merged: pd.DataFrame) -> pd.DataFrame:
    base = all_merged.copy()
    base["Year"] = pd.to_numeric(base["Year"], errors="coerce").fillna(0).astype(int)
    base["share"] = pd.to_numeric(base["share"], errors="coerce").fillna(0.0)

    prev = base[["Year", "NODE_CLSS_02", "keyword", "share"]].copy()
    prev["Year"] = prev["Year"] + 1
    prev = prev.rename(columns={"share": "share_prev"})

    out = base.merge(prev, on=["Year", "NODE_CLSS_02", "keyword"], how="left")
    out["share_prev"] = pd.to_numeric(out["share_prev"], errors="coerce").fillna(0.0)
    out["burst"] = out["share"] - out["share_prev"]
    return out


def topn_freq(df: pd.DataFrame, top_n: int, min_freq: int) -> pd.DataFrame:
    x = df.copy()
    x["freq"] = pd.to_numeric(x["freq"], errors="coerce").fillna(0).astype(int)
    x = x[x["freq"] >= min_freq].copy()
    return x.sort_values("freq", ascending=False).head(top_n)


def topn_burst(df: pd.DataFrame, top_n: int, min_freq: int) -> pd.DataFrame:
    x = df.copy()
    x["freq"] = pd.to_numeric(x["freq"], errors="coerce").fillna(0).astype(int)
    x["burst"] = pd.to_numeric(x["burst"], errors="coerce").fillna(0.0)
    x = x[(x["freq"] >= min_freq) & (x["burst"] > 0)].copy()
    return x.sort_values("burst", ascending=False).head(top_n)


def topn_mix(df: pd.DataFrame, top_n: int, min_freq: int) -> pd.DataFrame:
    """
    burst>0 우선 TopN, 부족분은 freq로 채움
    """
    x = df.copy()
    x["freq"] = pd.to_numeric(x["freq"], errors="coerce").fillna(0).astype(int)
    x["burst"] = pd.to_numeric(x["burst"], errors="coerce").fillna(0.0)

    burst_part = x[(x["freq"] >= min_freq) & (x["burst"] > 0)].sort_values("burst", ascending=False).head(top_n)
    picked = set(burst_part["keyword"].astype(str).tolist())

    need = top_n - len(burst_part)
    if need > 0:
        fill = x[(x["freq"] >= min_freq) & (~x["keyword"].astype(str).isin(picked))] \
            .sort_values("freq", ascending=False) \
            .head(need)
        out = pd.concat([burst_part, fill], ignore_index=True)
    else:
        out = burst_part
    return out


def main(base_dir: str, out_dir: str, years: List[int], top_n: int, min_freq: int,
         llm_max_keywords: int, sleep_sec: float, max_tokens: int, preview: bool):

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY 환경변수가 없습니다. export OPENAI_API_KEY=...")

    model = os.getenv("OPENAI_MODEL", "gpt-5-nano")
    client = OpenAI()

    # 1) raw 파일 로드
    raw_items = []  # each = raw_df
    for y in years:
        y_dir = os.path.join(base_dir, str(y))
        if not os.path.isdir(y_dir):
            print(f"[SKIP] no dir: {y_dir}")
            continue
        for fn in os.listdir(y_dir):
            if not fn.endswith("_keywords_raw.csv"):
                continue
            p = os.path.join(y_dir, fn)
            try:
                df = pd.read_csv(p)
                df.columns = df.columns.astype(str).str.strip()
                if "Year" not in df.columns:
                    df["Year"] = y
                df["Year"] = pd.to_numeric(df["Year"], errors="coerce").fillna(y).astype(int)
                raw_items.append(df)
            except Exception as e:
                print(f"[WARN] fail read {p}: {e}")

    if not raw_items:
        print("[WARN] no raw csv loaded.")
        return

    raw_all = pd.concat(raw_items, ignore_index=True)

    # 2) 중분류별 전체 키워드 유니온 수집 (freq합 기반으로 정렬)
    cls_keywords: Dict[str, List[str]] = {}
    for cls, g in raw_all.groupby("NODE_CLSS_02"):
        g2 = g.copy()
        g2["freq"] = pd.to_numeric(g2["freq"], errors="coerce").fillna(0).astype(int)
        # 전체 년도 freq 합
        sumfreq = g2.groupby("keyword", as_index=False).agg(sum_freq=("freq", "sum"))
        sumfreq = sumfreq.sort_values("sum_freq", ascending=False)
        kws = sumfreq["keyword"].astype(str).tolist()
        cls_keywords[str(cls)] = kws

    # 3) class별 merge_map 생성/캐시
    mm_dir = os.path.join(out_dir, "_merge_maps_by_class")
    os.makedirs(mm_dir, exist_ok=True)

    cls_mm: Dict[str, pd.DataFrame] = {}
    for cls, kws in sorted(cls_keywords.items(), key=lambda x: x[0]):
        cache_path = os.path.join(mm_dir, f"{safe_filename(cls)}_merge_map.tsv")
        mm = build_merge_map_for_class(
            client=client,
            model=model,
            cls=cls,
            keywords=kws,
            cache_path=cache_path,
            llm_max_keywords=llm_max_keywords,
            max_tokens=max_tokens,
            sleep_sec=sleep_sec,
            preview=preview,
        )
        cls_mm[cls] = mm

    # 4) year/class별 압축 적용 + 저장
    merged_dir = os.path.join(out_dir, "compressed_by_year_class")
    os.makedirs(merged_dir, exist_ok=True)

    merged_all_list = []
    for (y, cls), g in raw_all.groupby(["Year", "NODE_CLSS_02"]):
        mm = cls_mm.get(str(cls))
        if mm is None:
            continue
        merged = apply_merge_map_to_year_class(g, mm)
        merged_all_list.append(merged)

        out_y_dir = os.path.join(merged_dir, str(int(y)))
        os.makedirs(out_y_dir, exist_ok=True)
        out_path = os.path.join(out_y_dir, f"{safe_filename(cls)}_keywords_merged_all.csv")
        merged.sort_values("freq", ascending=False).to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[OK] {out_path} rows={len(merged)}")

    if not merged_all_list:
        print("[WARN] no merged results.")
        return

    merged_all = pd.concat(merged_all_list, ignore_index=True)

    # 5) burst 계산(전체 통합 후 prev 사용)
    merged_burst = recalc_burst(merged_all)

    # 6) year/class별 top100 생성 저장
    top_dir = os.path.join(out_dir, "top100_by_year_class")
    os.makedirs(top_dir, exist_ok=True)

    for (y, cls), g in merged_burst.groupby(["Year", "NODE_CLSS_02"]):
        y = int(y)
        cls = str(cls)

        out_y_dir = os.path.join(top_dir, str(y))
        os.makedirs(out_y_dir, exist_ok=True)

        freq100 = topn_freq(g, top_n=top_n, min_freq=min_freq)
        burst100 = topn_burst(g, top_n=top_n, min_freq=min_freq)
        mix100 = topn_mix(g, top_n=top_n, min_freq=min_freq)

        p1 = os.path.join(out_y_dir, f"{safe_filename(cls)}_freq_top{top_n}.csv")
        p2 = os.path.join(out_y_dir, f"{safe_filename(cls)}_burst_top{top_n}.csv")
        p3 = os.path.join(out_y_dir, f"{safe_filename(cls)}_mix_top{top_n}.csv")

        freq100.to_csv(p1, index=False, encoding="utf-8-sig")
        burst100.to_csv(p2, index=False, encoding="utf-8-sig")
        mix100.to_csv(p3, index=False, encoding="utf-8-sig")

        print(f"[OK] {p1} rows={len(freq100)}")
        print(f"[OK] {p2} rows={len(burst100)}")
        print(f"[OK] {p3} rows={len(mix100)}")

    # 전체 통합 파일도 저장
    merged_all_path = os.path.join(out_dir, "merged_all_years_all_classes.csv")
    merged_burst.sort_values(["Year", "NODE_CLSS_02", "freq"], ascending=[True, True, False]).to_csv(
        merged_all_path, index=False, encoding="utf-8-sig"
    )
    print(f"[OK] {merged_all_path}")

    print("\nDONE ✅ split(raw) -> class-global merge_map -> year/class compress -> freq/burst/mix top100 완료")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", required=True, help="keywords_by_year_class 폴더")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--years", default="2021,2022,2023,2024,2025")
    p.add_argument("--top_n", type=int, default=100)
    p.add_argument("--min_freq", type=int, default=5)
    p.add_argument("--llm_max_keywords", type=int, default=800, help="LLM에 넣을 최대 키워드 수(초과분은 identity)")
    p.add_argument("--sleep", type=float, default=0.25)
    p.add_argument("--max_tokens", type=int, default=1400)
    p.add_argument("--preview", action="store_true")
    args = p.parse_args()

    years = [int(x.strip()) for x in args.years.split(",") if x.strip()]

    main(
        base_dir=args.base_dir,
        out_dir=args.out_dir,
        years=years,
        top_n=args.top_n,
        min_freq=args.min_freq,
        llm_max_keywords=args.llm_max_keywords,
        sleep_sec=args.sleep,
        max_tokens=args.max_tokens,
        preview=args.preview,
    )
