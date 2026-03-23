#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
01_split_keywords_by_year_class.py

원본 JSON(DBpia/Datathon)에서
- Year(PBSH 앞 4자리)
- NODE_CLSS_02(중분류)
- KYWD(키워드 문자열)
를 사용해,

✅ (year, class) 단위로 "키워드 빈도(freq=등장 논문 수)"를 계산하고
✅ 년도/중분류별 CSV로 저장한다.

freq 정의:
- 한 논문(row) 안에서 동일 키워드가 여러번 있어도 1회로 카운트
- 즉, keyword가 등장한 "논문 수" 기준

출력:
out_dir/keywords_by_year_class/{year}/{class}_keywords_raw.csv
(out 통합 인덱스) out_dir/keywords_by_year_class/_index.csv

실행 예시:
python 01_split_keywords_by_year_class.py \
  --input_json "/Users/idonghyeon/Desktop/datathon/SSU_Datathon2025_공학분야_62199.json" \
  --out_dir "/Users/idonghyeon/Desktop/dathon_result" \
  --year_min 2021 --year_max 2025 \
  --only_engineering
"""

import os
import re
import json
import argparse
from collections import defaultdict
import pandas as pd


# -------------------------
# Keyword parsing
# -------------------------
_acronym_pat = re.compile(r'\(([A-Za-z0-9\-]{2,10})\)')  # (CNN), (LSTM) 등

def normalize_token(t: str) -> str:
    t = (t or "").strip().lower()
    if not t:
        return ""
    t = t.replace("·", " ").replace("_", " ").replace("-", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t

def extract_keywords(kw_str: str) -> list[str]:
    text = str(kw_str or "")

    # 1) 괄호 약어 추출
    acronyms = _acronym_pat.findall(text)
    acronyms = [normalize_token(a) for a in acronyms if a]

    # 2) 괄호 내용 제거
    text = re.sub(r"\([^)]*\)", "", text)

    # 3) split
    tokens = re.split(r"[,;/]", text)
    tokens = [normalize_token(t) for t in tokens if normalize_token(t)]

    # 4) 약어 추가
    tokens.extend([a for a in acronyms if a])

    # 5) 너무 짧은 토큰 제거
    tokens = [t for t in tokens if len(t) > 1]

    # 중복 제거(논문 내)
    return list(dict.fromkeys(tokens))


def safe_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in " _-." else "_" for c in str(s)).strip()


def main(input_json: str, out_dir: str, year_min: int, year_max: int, only_engineering: bool):
    if not os.path.exists(input_json):
        raise FileNotFoundError(f"input_json not found: {input_json}")

    os.makedirs(out_dir, exist_ok=True)

    with open(input_json, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if isinstance(raw_data, dict) and "NODE_LIST" in raw_data:
        data_list = raw_data["NODE_LIST"]
    elif isinstance(raw_data, list):
        data_list = raw_data
    else:
        data_list = [raw_data]

    df = pd.DataFrame(data_list)

    need_cols = ["PBSH", "NODE_CLSS_02", "KYWD"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # 공학만 필터
    if only_engineering and "NODE_CLSS_01" in df.columns:
        df = df[df["NODE_CLSS_01"].fillna("") == "공학"].copy()

    # Year 파싱
    df["Year"] = df["PBSH"].fillna("").astype(str).str[:4]
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df = df[(df["Year"].notna()) & (df["Year"] >= year_min) & (df["Year"] <= year_max)].copy()
    df["Year"] = df["Year"].astype(int)

    df["NODE_CLSS_02"] = df["NODE_CLSS_02"].fillna("미분류")
    df["KYWD"] = df["KYWD"].fillna("")

    # (year,class)별 paper_count, keyword_freq
    paper_count = defaultdict(int)         # (y, cls) -> 논문 수
    kw_freq = defaultdict(int)             # (y, cls, kw) -> 등장 논문 수

    for _, row in df.iterrows():
        y = int(row["Year"])
        cls = str(row["NODE_CLSS_02"])
        paper_count[(y, cls)] += 1

        kws = set(extract_keywords(row["KYWD"]))
        for k in kws:
            kw_freq[(y, cls, k)] += 1

    # 저장
    base_dir = os.path.join(out_dir, "keywords_by_year_class")
    os.makedirs(base_dir, exist_ok=True)

    index_rows = []
    for (y, cls, kw), freq in kw_freq.items():
        pc = paper_count[(y, cls)]
        index_rows.append({
            "Year": y,
            "NODE_CLSS_02": cls,
            "keyword": kw,
            "freq": int(freq),
            "paper_count": int(pc),
            "share": float(freq) / float(pc) if pc else 0.0
        })

    all_df = pd.DataFrame(index_rows)
    if all_df.empty:
        print("[WARN] no extracted keywords.")
        return

    # year/class별로 분할 저장
    for (y, cls), g in all_df.groupby(["Year", "NODE_CLSS_02"]):
        out_y_dir = os.path.join(base_dir, str(y))
        os.makedirs(out_y_dir, exist_ok=True)
        out_path = os.path.join(out_y_dir, f"{safe_filename(cls)}_keywords_raw.csv")

        g = g.sort_values("freq", ascending=False).reset_index(drop=True)
        g.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[OK] {out_path} rows={len(g)}")

    # 전체 인덱스
    idx_path = os.path.join(base_dir, "_index.csv")
    all_df.sort_values(["Year", "NODE_CLSS_02", "freq"], ascending=[True, True, False]).to_csv(
        idx_path, index=False, encoding="utf-8-sig"
    )
    print(f"[OK] {idx_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_json", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--year_min", type=int, default=2021)
    p.add_argument("--year_max", type=int, default=2025)
    p.add_argument("--only_engineering", action="store_true")
    args = p.parse_args()

    main(
        input_json=args.input_json,
        out_dir=args.out_dir,
        year_min=args.year_min,
        year_max=args.year_max,
        only_engineering=args.only_engineering
    )
