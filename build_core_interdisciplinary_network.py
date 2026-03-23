#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_core_interdisciplinary_network.py

목표
1) Spread TopN ∩ Bridge TopM = "핵심 융합 키워드(Core Keywords)" 산출
2) Core Keywords만 사용해서 중분류-중분류 Edge(융합 네트워크) 재계산
3) 연도별 TopK core 키워드 + core edge + hub(연결 중심 중분류) 산출

입력
- merged_dir/{year}_freq_merged_all.csv
- spread_csv = interdisciplinary_spread_by_year.csv (Spread 모듈 산출물)
- bridge_dir/bridge_keywords_{year}.json (Bridge 모듈 산출물)

출력(out_dir)
- core_keywords_{year}.csv                : 연도별 Core Keywords(TopK)
- core_keywords_all.csv                   : 전체 연도 Core Keywords(TopK 누적)
- core_edges_{year}.json / .csv           : Core Keywords 기반 class-edge
- core_hubs_{year}.csv                    : 중분류 hub(가중치 합)
- core_meta.json                          : 설정/통계

실행 예시
python build_core_interdisciplinary_network.py \
  --merged_dir "/Users/idonghyeon/Desktop/dathon_result/merged" \
  --spread_csv "/Users/idonghyeon/Desktop/dathon_result/out_inter/spread/interdisciplinary_spread_by_year.csv" \
  --bridge_dir "/Users/idonghyeon/Desktop/dathon_result/out_inter/bridge" \
  --out_dir "/Users/idonghyeon/Desktop/dathon_result/out_inter/core" \
  --years "2021,2022,2023,2024,2025" \
  --spread_top_n 20 \
  --bridge_top_n 200 \
  --core_top_k 30 \
  --min_freq_per_class 1 \
  --top_edges 2000
"""

import os
import json
import math
import argparse
from itertools import combinations
from typing import Dict, List, Set, Tuple

import pandas as pd


# --------------------------
# IO helpers
# --------------------------
def safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)


def read_csv_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.astype(str).str.replace("\ufeff", "").str.strip()
    return df


def to_int(x, default=0) -> int:
    try:
        return int(float(x))
    except Exception:
        return int(default)


def to_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


# --------------------------
# Load freq merged
# --------------------------
def load_year_freq_merged(merged_dir: str, year: int) -> pd.DataFrame:
    """
    Expect: {year}_freq_merged_all.csv
    Need: Year, NODE_CLSS_02, keyword, freq
    """
    path = os.path.join(merged_dir, f"{year}_freq_merged_all.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing file: {path}")

    df = read_csv_clean(path)

    if "Year" not in df.columns:
        df["Year"] = year
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").fillna(year).astype(int)

    need = {"NODE_CLSS_02", "keyword", "freq"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"{path} missing columns: {miss}")

    df["NODE_CLSS_02"] = df["NODE_CLSS_02"].astype(str).fillna("미분류").str.strip()
    df["keyword"] = df["keyword"].astype(str).fillna("").str.strip()
    df["freq"] = pd.to_numeric(df["freq"], errors="coerce").fillna(0).astype(int)

    df = df[(df["freq"] > 0) & (df["keyword"] != "")].copy()
    return df


# --------------------------
# Spread TopN
# --------------------------
def pick_spread_topN(spread_df: pd.DataFrame, year: int, top_n: int) -> pd.DataFrame:
    """
    spread_df: interdisciplinary_spread_by_year.csv output
    Must include: Year, keyword, delta_classes, entropy_norm, total_freq
    """
    sub = spread_df[spread_df["Year"] == int(year)].copy()
    if sub.empty:
        return sub

    # ensure numeric
    for col in ["delta_classes", "entropy_norm", "total_freq", "n_classes"]:
        if col in sub.columns:
            sub[col] = pd.to_numeric(sub[col], errors="coerce").fillna(0)

    # rank 기준(Spread 코드와 동일)
    sub = sub.sort_values(
        ["delta_classes", "entropy_norm", "total_freq"],
        ascending=[False, False, False]
    ).head(int(top_n))

    sub["spread_rank"] = range(1, len(sub) + 1)
    return sub


# --------------------------
# Bridge TopM
# --------------------------
def load_bridge_json(bridge_path: str) -> pd.DataFrame:
    """
    bridge_keywords_{y}.json:
    list of dicts, each should have keyword / total_freq / n_classes / max_share / inter_score / (optional score)
    """
    if not os.path.exists(bridge_path):
        return pd.DataFrame()

    with open(bridge_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        return pd.DataFrame()

    df = pd.DataFrame(data)
    if df.empty:
        return df

    if "keyword" not in df.columns:
        return pd.DataFrame()

    # numeric normalize
    for col in ["total_freq", "n_classes", "max_share", "inter_score", "score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # score 없으면 계산
    if "score" not in df.columns:
        df["score"] = df["inter_score"] * df["total_freq"].map(lambda z: math.log(1 + float(z)))

    # 안전
    df["keyword"] = df["keyword"].astype(str).str.strip()
    df = df[df["keyword"] != ""].copy()
    return df


def pick_bridge_topM(bridge_df: pd.DataFrame, top_m: int) -> pd.DataFrame:
    if bridge_df.empty:
        return bridge_df
    bridge_df = bridge_df.sort_values(["score", "inter_score", "total_freq"], ascending=[False, False, False])
    bridge_df = bridge_df.head(int(top_m)).copy()
    bridge_df["bridge_rank"] = range(1, len(bridge_df) + 1)
    return bridge_df


# --------------------------
# Core Keywords = Spread ∩ Bridge
# --------------------------
def build_core_keywords(
    spread_top: pd.DataFrame,
    bridge_top: pd.DataFrame,
    core_top_k: int
) -> pd.DataFrame:
    """
    반환: 연도별 Core Keywords TopK (정렬=Bridge score 우선)
    """
    if spread_top.empty or bridge_top.empty:
        return pd.DataFrame()

    sset = set(spread_top["keyword"].astype(str).str.strip().tolist())
    bset = set(bridge_top["keyword"].astype(str).str.strip().tolist())
    core = sorted(list(sset & bset))

    if not core:
        return pd.DataFrame()

    # merge spread info + bridge info
    s = spread_top.copy()
    b = bridge_top.copy()

    s = s.rename(columns={
        "total_freq": "spread_total_freq",
        "n_classes": "spread_n_classes",
        "entropy_norm": "spread_entropy_norm",
        "delta_classes": "spread_delta_classes"
    })

    b = b.rename(columns={
        "total_freq": "bridge_total_freq",
        "n_classes": "bridge_n_classes",
        "max_share": "bridge_max_share",
        "inter_score": "bridge_inter_score",
        "score": "bridge_score"
    })

    s_keep = ["keyword", "spread_rank", "spread_total_freq", "spread_n_classes", "spread_entropy_norm", "spread_delta_classes"]
    for c in ["top_classes", "class_ratio_json"]:
        if c in s.columns:
            s_keep.append(c)

    b_keep = ["keyword", "bridge_rank", "bridge_total_freq", "bridge_n_classes", "bridge_max_share", "bridge_inter_score", "bridge_score"]
    if "class_dist" in b.columns:
        b_keep.append("class_dist")

    core_df = pd.DataFrame({"keyword": core})
    core_df = core_df.merge(s[s_keep], on="keyword", how="left").merge(b[b_keep], on="keyword", how="left")

    # TopK 정렬 기준:
    # 1) bridge_score desc (연결자 강함)
    # 2) spread_delta_classes desc (확산성)
    # 3) spread_entropy_norm desc (균등 확산)
    # 4) bridge_total_freq desc (근거)
    core_df = core_df.sort_values(
        ["bridge_score", "spread_delta_classes", "spread_entropy_norm", "bridge_total_freq"],
        ascending=[False, False, False, False]
    ).head(int(core_top_k)).copy()

    core_df["core_rank"] = range(1, len(core_df) + 1)
    return core_df


# --------------------------
# Core edges (recompute using core keywords only)
# --------------------------
def recompute_edges_with_core_keywords(
    df_year_freq: pd.DataFrame,
    core_keywords: Set[str],
    min_freq_per_class: int = 1,
    top_edges: int = 2000
) -> pd.DataFrame:
    """
    df_year_freq: (Year, NODE_CLSS_02, keyword, freq)
    edge_add(c1,c2) = total_freq(k)*p(c1|k)*p(c2|k)
    """
    if df_year_freq.empty or not core_keywords:
        return pd.DataFrame(columns=["Year", "c1", "c2", "weight"])

    y = int(df_year_freq["Year"].iloc[0])

    d = df_year_freq.copy()
    d = d[d["keyword"].isin(core_keywords)].copy()
    d = d[d["freq"] >= int(min_freq_per_class)].copy()
    if d.empty:
        return pd.DataFrame(columns=["Year", "c1", "c2", "weight"])

    # aggregate per (keyword, class)
    d = d.groupby(["Year", "keyword", "NODE_CLSS_02"], as_index=False)["freq"].sum()

    # total per keyword
    total = d.groupby(["Year", "keyword"], as_index=False).agg(total_freq=("freq", "sum"))
    d = d.merge(total, on=["Year", "keyword"], how="left")

    # p(c|k,y)
    d["p"] = d["freq"] / d["total_freq"].replace(0, pd.NA)
    d["p"] = d["p"].fillna(0.0)

    edges: Dict[Tuple[str, str], float] = {}
    for (yy, k), g in d.groupby(["Year", "keyword"]):
        rows = [(str(r["NODE_CLSS_02"]), float(r["p"])) for _, r in g.iterrows() if float(r["p"]) > 0]
        if len(rows) < 2:
            continue

        tf = float(g["total_freq"].iloc[0])
        for (c1, p1), (c2, p2) in combinations(sorted(rows, key=lambda x: x[0]), 2):
            w = tf * p1 * p2
            if w <= 0:
                continue
            key = (c1, c2)
            edges[key] = edges.get(key, 0.0) + w

    out = pd.DataFrame([{"Year": y, "c1": k[0], "c2": k[1], "weight": v} for k, v in edges.items()])
    if out.empty:
        return out

    out = out.sort_values("weight", ascending=False).head(int(top_edges)).reset_index(drop=True)
    return out


def compute_hubs(edges_df: pd.DataFrame) -> pd.DataFrame:
    """
    hub score = sum of incident edge weights per class
    """
    if edges_df.empty:
        return pd.DataFrame(columns=["Year", "class", "hub_weight_sum"])

    y = int(edges_df["Year"].iloc[0])
    a = edges_df[["c1", "weight"]].rename(columns={"c1": "class"}).copy()
    b = edges_df[["c2", "weight"]].rename(columns={"c2": "class"}).copy()
    x = pd.concat([a, b], ignore_index=True)

    hubs = x.groupby("class", as_index=False).agg(hub_weight_sum=("weight", "sum"))
    hubs["Year"] = y
    hubs = hubs.sort_values("hub_weight_sum", ascending=False).reset_index(drop=True)
    hubs["hub_rank"] = range(1, len(hubs) + 1)
    return hubs[["Year", "hub_rank", "class", "hub_weight_sum"]]


# --------------------------
# Main
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged_dir", required=True, help="폴더: {year}_freq_merged_all.csv 존재")
    ap.add_argument("--spread_csv", required=True, help="interdisciplinary_spread_by_year.csv 경로")
    ap.add_argument("--bridge_dir", required=True, help="bridge_keywords_{y}.json들이 있는 폴더")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--years", default="2021,2022,2023,2024,2025")

    ap.add_argument("--spread_top_n", type=int, default=20, help="연도별 Spread TopN")
    ap.add_argument("--bridge_top_n", type=int, default=200, help="연도별 Bridge TopM")
    ap.add_argument("--core_top_k", type=int, default=30, help="연도별 Core Keywords TopK(교집합 내에서)")
    ap.add_argument("--min_freq_per_class", type=int, default=1)
    ap.add_argument("--top_edges", type=int, default=2000)

    args = ap.parse_args()
    safe_mkdir(args.out_dir)

    years = [int(x.strip()) for x in str(args.years).split(",") if x.strip()]

    # load spread csv once
    if not os.path.exists(args.spread_csv):
        raise FileNotFoundError(f"--spread_csv not found: {args.spread_csv}")

    spread_df = read_csv_clean(args.spread_csv)
    need = {"Year", "keyword", "delta_classes", "entropy_norm", "total_freq"}
    miss = need - set(spread_df.columns)
    if miss:
        raise ValueError(f"spread_csv missing columns: {miss}")

    spread_df["Year"] = pd.to_numeric(spread_df["Year"], errors="coerce").fillna(0).astype(int)
    spread_df["keyword"] = spread_df["keyword"].astype(str).str.strip()

    all_core_rows = []
    meta = {
        "config": {
            "spread_top_n": int(args.spread_top_n),
            "bridge_top_n": int(args.bridge_top_n),
            "core_top_k": int(args.core_top_k),
            "min_freq_per_class": int(args.min_freq_per_class),
            "top_edges": int(args.top_edges),
        },
        "per_year_stats": {}
    }

    for y in years:
        # 1) Spread TopN
        spread_top = pick_spread_topN(spread_df, year=y, top_n=args.spread_top_n)
        spread_set = set(spread_top["keyword"].tolist()) if not spread_top.empty else set()

        # 2) Bridge TopM
        bridge_path = os.path.join(args.bridge_dir, f"bridge_keywords_{y}.json")
        bridge_df = load_bridge_json(bridge_path)
        bridge_top = pick_bridge_topM(bridge_df, top_m=args.bridge_top_n)
        bridge_set = set(bridge_top["keyword"].tolist()) if not bridge_top.empty else set()

        # 3) Core = intersection
        core_df = build_core_keywords(spread_top, bridge_top, core_top_k=args.core_top_k)
        core_set = set(core_df["keyword"].tolist()) if not core_df.empty else set()

        # 4) Save core keywords
        core_kw_csv = os.path.join(args.out_dir, f"core_keywords_{y}.csv")
        core_df.to_csv(core_kw_csv, index=False, encoding="utf-8-sig")

        # 5) Recompute edges using core keywords only
        edges_df = pd.DataFrame(columns=["Year", "c1", "c2", "weight"])
        hubs_df = pd.DataFrame(columns=["Year", "hub_rank", "class", "hub_weight_sum"])

        if core_set:
            dfy = load_year_freq_merged(args.merged_dir, y)
            edges_df = recompute_edges_with_core_keywords(
                df_year_freq=dfy,
                core_keywords=core_set,
                min_freq_per_class=args.min_freq_per_class,
                top_edges=args.top_edges
            )

            hubs_df = compute_hubs(edges_df)

        # save edges
        edges_json = os.path.join(args.out_dir, f"core_edges_{y}.json")
        edges_csv = os.path.join(args.out_dir, f"core_edges_{y}.csv")
        with open(edges_json, "w", encoding="utf-8") as f:
            json.dump(edges_df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)
        edges_df.to_csv(edges_csv, index=False, encoding="utf-8-sig")

        # save hubs
        hubs_csv = os.path.join(args.out_dir, f"core_hubs_{y}.csv")
        hubs_df.to_csv(hubs_csv, index=False, encoding="utf-8-sig")

        # accumulate
        if not core_df.empty:
            core_df = core_df.copy()
            core_df.insert(0, "Year", y)
            all_core_rows.append(core_df)

        meta["per_year_stats"][str(y)] = {
            "spread_top_n_count": int(len(spread_set)),
            "bridge_top_n_count": int(len(bridge_set)),
            "core_intersection_count": int(len(core_set)),
            "saved": {
                "core_keywords_csv": os.path.basename(core_kw_csv),
                "core_edges_json": os.path.basename(edges_json),
                "core_edges_csv": os.path.basename(edges_csv),
                "core_hubs_csv": os.path.basename(hubs_csv),
            }
        }

        print(f"[OK] year={y} | spread={len(spread_set)} bridge={len(bridge_set)} core={len(core_set)}")
        print(f"     -> {core_kw_csv}")
        print(f"     -> {edges_json}")
        print(f"     -> {hubs_csv}")

    # save all years combined
    all_core_df = pd.concat(all_core_rows, ignore_index=True) if all_core_rows else pd.DataFrame()
    all_core_path = os.path.join(args.out_dir, "core_keywords_all.csv")
    all_core_df.to_csv(all_core_path, index=False, encoding="utf-8-sig")

    # save meta
    meta_path = os.path.join(args.out_dir, "core_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] all core keywords -> {all_core_path}")
    print(f"[OK] meta -> {meta_path}")
    print("DONE")


if __name__ == "__main__":
    main()
