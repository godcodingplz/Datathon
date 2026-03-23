#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse
import pandas as pd
import numpy as np

def entropy_norm(probs: np.ndarray) -> float:
    probs = probs[probs > 0]
    if len(probs) <= 1:
        return 0.0
    h = -np.sum(probs * np.log(probs))
    return float(h / np.log(len(probs)))  # 0~1

def load_year_freq(merged_dir: str, year: int) -> pd.DataFrame:
    path = os.path.join(merged_dir, f"{year}_freq_merged_all.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df.columns = df.columns.astype(str).str.strip()
    need = {"Year","NODE_CLSS_02","keyword","freq"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"{path} missing columns: {miss}")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").fillna(year).astype(int)
    df["freq"] = pd.to_numeric(df["freq"], errors="coerce").fillna(0).astype(int)
    df["keyword"] = df["keyword"].astype(str).str.strip()
    df["NODE_CLSS_02"] = df["NODE_CLSS_02"].astype(str).str.strip()
    return df

def build_spread_table(df_y: pd.DataFrame, fmin_class: int, topk_classes: int = 5) -> pd.DataFrame:
    # keyword x class freq
    x = df_y[df_y["freq"] > 0].copy()
    x = x.groupby(["Year","keyword","NODE_CLSS_02"], as_index=False)["freq"].sum()

    # class coverage with threshold
    x["valid_class"] = x["freq"] >= fmin_class

    # total per keyword
    tot = x.groupby(["Year","keyword"], as_index=False).agg(
        total_freq=("freq","sum"),
        n_classes=("valid_class","sum")
    )

    # probs per keyword
    x2 = x.merge(tot[["Year","keyword","total_freq"]], on=["Year","keyword"], how="left")
    x2["p"] = x2["freq"] / x2["total_freq"].replace(0, np.nan)
    x2["p"] = x2["p"].fillna(0.0)

    # entropy_norm per keyword (only among valid classes is OK, but here use all classes where freq>0)
    ent = x2.groupby(["Year","keyword"]).apply(lambda g: entropy_norm(g["p"].values)).reset_index(name="entropy_norm")

    # top classes string
    def top_classes_str(g):
        g = g.sort_values("p", ascending=False).head(topk_classes)
        parts = [f"{r.NODE_CLSS_02}:{r.p:.2f}" for r in g.itertuples(index=False)]
        return "|".join(parts)

    topc = x2.groupby(["Year","keyword"]).apply(top_classes_str).reset_index(name="top_classes")

    # full ratio json
    def ratio_json(g):
        g = g.sort_values("p", ascending=False)
        d = {str(r.NODE_CLSS_02): float(r.p) for r in g.itertuples(index=False)}
        return json.dumps(d, ensure_ascii=False)

    ratios = x2.groupby(["Year","keyword"]).apply(ratio_json).reset_index(name="class_ratio_json")

    out = tot.merge(ent, on=["Year","keyword"], how="left").merge(topc, on=["Year","keyword"], how="left").merge(ratios, on=["Year","keyword"], how="left")
    out["entropy_norm"] = out["entropy_norm"].fillna(0.0)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged_dir", required=True, help="폴더: {year}_freq_merged_all.csv 존재")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--years", default="2021,2022,2023,2024,2025")
    ap.add_argument("--fmin_class", type=int, default=3, help="한 중분류에서 유효로 치는 최소 freq")
    ap.add_argument("--min_classes", type=int, default=3, help="학제간 후보 최소 중분류 수 m_y(k)")
    ap.add_argument("--min_entropy", type=float, default=0.6, help="몰빵 제거용 entropy_norm 하한")
    ap.add_argument("--min_total_freq", type=int, default=10, help="너무 희소한 키워드 제거")
    ap.add_argument("--min_delta_classes", type=int, default=1, help="확산: 전년 대비 중분류 수 증가")
    ap.add_argument("--top_n", type=int, default=20, help="연도별 출력 topN(본문용)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    years = [int(x.strip()) for x in args.years.split(",") if x.strip()]

    # build per year tables
    per_year = {}
    for y in years:
        dfy = load_year_freq(args.merged_dir, y)
        per_year[y] = build_spread_table(dfy, fmin_class=args.fmin_class)

    # add delta_classes (vs previous year)
    all_rows = []
    for y in years:
        cur = per_year[y].copy()
        prev = per_year.get(y-1, pd.DataFrame(columns=["keyword","n_classes"]))[["keyword","n_classes"]].copy()
        prev = prev.rename(columns={"n_classes":"n_classes_prev"})
        cur = cur.merge(prev, on="keyword", how="left")
        cur["n_classes_prev"] = cur["n_classes_prev"].fillna(0).astype(int)
        cur["delta_classes"] = cur["n_classes"] - cur["n_classes_prev"]
        all_rows.append(cur)

    all_df = pd.concat(all_rows, ignore_index=True)
    all_df = all_df.sort_values(["Year","delta_classes","entropy_norm","total_freq"], ascending=[True,False,False,False])

    # filter for "spread"
    filt = (
        (all_df["n_classes"] >= args.min_classes) &
        (all_df["entropy_norm"] >= args.min_entropy) &
        (all_df["total_freq"] >= args.min_total_freq) &
        (all_df["delta_classes"] >= args.min_delta_classes)
    )
    spread = all_df[filt].copy()

    # save one consolidated file
    out_all = os.path.join(args.out_dir, "interdisciplinary_spread_by_year.csv")
    spread.to_csv(out_all, index=False, encoding="utf-8-sig")
    print("[OK]", out_all)

    # save per-year TopN
    for y in years:
        sub = spread[spread["Year"] == y].copy()
        if sub.empty:
            continue
        sub_top = sub.head(args.top_n).copy()
        out_y = os.path.join(args.out_dir, f"{y}_spread_keywords_top{args.top_n}.csv")
        sub_top.to_csv(out_y, index=False, encoding="utf-8-sig")
        print("[OK]", out_y)

if __name__ == "__main__":
    main()
