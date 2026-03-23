import os
import json
import math
import argparse
from itertools import combinations
import pandas as pd


def safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)


def load_year_freq_merged(merged_dir: str, year: int) -> pd.DataFrame:
    """
    Expect a file like: {year}_freq_merged_all.csv
    Columns needed: Year, NODE_CLSS_02, keyword, freq
    Optional: paper_count, share, merged_variants
    """
    path = os.path.join(merged_dir, f"{year}_freq_merged_all.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing file: {path}")

    df = pd.read_csv(path)
    # normalize columns
    if "Year" not in df.columns:
        df["Year"] = year
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").fillna(year).astype(int)

    for col in ["NODE_CLSS_02", "keyword", "freq"]:
        if col not in df.columns:
            raise ValueError(f"required column missing: {col} in {path}")

    df["NODE_CLSS_02"] = df["NODE_CLSS_02"].astype(str).fillna("미분류")
    df["keyword"] = df["keyword"].astype(str).str.strip()
    df["freq"] = pd.to_numeric(df["freq"], errors="coerce").fillna(0).astype(int)

    # keep only positive freq
    df = df[df["freq"] > 0].copy()
    return df


def compute_inter_scores(df_year: pd.DataFrame, min_freq_per_class: int = 1) -> pd.DataFrame:
    """
    df_year: rows of (Year, NODE_CLSS_02, keyword, freq)
    Returns per (Year, keyword) with:
      total_freq, n_classes, max_share, inter_score, class_dist_json
    """
    y = int(df_year["Year"].iloc[0])

    # optional per-class min freq filter (to reduce noise)
    d = df_year[df_year["freq"] >= int(min_freq_per_class)].copy()
    if d.empty:
        return pd.DataFrame(columns=[
            "Year", "keyword", "total_freq", "n_classes", "max_share", "inter_score", "class_dist"
        ])

    # total per keyword
    total = d.groupby(["Year", "keyword"], as_index=False).agg(total_freq=("freq", "sum"))

    # join to compute p(c|k,y)
    d2 = d.merge(total, on=["Year", "keyword"], how="left")
    d2["p"] = d2["freq"] / d2["total_freq"].replace(0, pd.NA)
    d2["p"] = d2["p"].fillna(0.0)

    # inter_score = -sum p log p
    def _ent(ps):
        s = 0.0
        for v in ps:
            if v > 0:
                s += -v * math.log(v)
        return s

    inter = (
        d2.groupby(["Year", "keyword"], as_index=False)
          .agg(
              total_freq=("total_freq", "max"),
              n_classes=("NODE_CLSS_02", "nunique"),
              max_share=("p", "max"),
              inter_score=("p", _ent),
          )
    )

    # class distribution dict: {class: p}
    dist = (
        d2.groupby(["Year", "keyword"])
          .apply(lambda g: {str(r["NODE_CLSS_02"]): float(r["p"]) for _, r in g.sort_values("p", ascending=False).iterrows()})
          .reset_index(name="class_dist")
    )

    out = inter.merge(dist, on=["Year", "keyword"], how="left")
    out["Year"] = y
    return out


def pick_bridge_keywords(inter_df: pd.DataFrame,
                         min_classes: int,
                         max_share: float,
                         min_total_freq: int,
                         top_n: int) -> pd.DataFrame:
    """
    Filter + rank bridge keywords.
    Score = inter_score * log(1+total_freq)
    """
    x = inter_df.copy()
    x = x[
        (x["n_classes"] >= int(min_classes)) &
        (x["max_share"] <= float(max_share)) &
        (x["total_freq"] >= int(min_total_freq))
    ].copy()

    if x.empty:
        return x

    x["score"] = x["inter_score"] * x["total_freq"].map(lambda z: math.log(1 + float(z)))
    x = x.sort_values(["score", "inter_score", "total_freq"], ascending=[False, False, False]).head(int(top_n))
    return x


def build_class_edges(df_year: pd.DataFrame,
                      min_freq_per_class: int = 1,
                      top_edges: int = 2000) -> pd.DataFrame:
    """
    Build class-class edges using shared keywords.
    For each (k,y), for all class pairs (c1,c2):
      edge_add = total_freq(k,y) * p(c1|k,y) * p(c2|k,y)
    Then sum over keywords.
    """
    d = df_year[df_year["freq"] >= int(min_freq_per_class)].copy()
    if d.empty:
        return pd.DataFrame(columns=["Year", "c1", "c2", "weight"])

    y = int(d["Year"].iloc[0])

    # totals
    total = d.groupby(["Year", "keyword"], as_index=False).agg(total_freq=("freq", "sum"))
    d = d.merge(total, on=["Year", "keyword"], how="left")
    d["p"] = d["freq"] / d["total_freq"].replace(0, pd.NA)
    d["p"] = d["p"].fillna(0.0)

    edges = {}
    for (yy, k), g in d.groupby(["Year", "keyword"]):
        rows = [(str(r["NODE_CLSS_02"]), float(r["p"])) for _, r in g.iterrows() if float(r["p"]) > 0]
        if len(rows) < 2:
            continue
        tf = float(g["total_freq"].iloc[0])
        # pair contribution
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged_dir", required=True, help="merged 출력 폴더(예: .../dathon_result/merged)")
    ap.add_argument("--out_dir", required=True, help="산출물 저장 폴더")
    ap.add_argument("--years", default="2021,2022,2023,2024,2025")
    ap.add_argument("--min_freq_per_class", type=int, default=1, help="(k,c,y) 최소 freq 필터")
    ap.add_argument("--min_classes", type=int, default=3, help="브릿지 키워드: 등장 중분류 수 하한")
    ap.add_argument("--max_share", type=float, default=0.6, help="브릿지 키워드: max p(c|k,y) 상한(쏠림 제한)")
    ap.add_argument("--min_total_freq", type=int, default=20, help="브릿지 키워드: total_freq 하한")
    ap.add_argument("--top_n", type=int, default=200, help="연도별 브릿지 키워드 Top-N")
    ap.add_argument("--top_edges", type=int, default=2000, help="연도별 class-edge Top-N")
    args = ap.parse_args()

    years = [int(x.strip()) for x in args.years.split(",") if x.strip()]
    safe_mkdir(args.out_dir)

    for y in years:
        dfy = load_year_freq_merged(args.merged_dir, y)

        inter = compute_inter_scores(dfy, min_freq_per_class=args.min_freq_per_class)
        inter_csv = os.path.join(args.out_dir, f"interdisciplinary_summary_{y}.csv")
        inter.to_csv(inter_csv, index=False, encoding="utf-8-sig")

        bridge = pick_bridge_keywords(
            inter,
            min_classes=args.min_classes,
            max_share=args.max_share,
            min_total_freq=args.min_total_freq,
            top_n=args.top_n
        )

        bridge_json = os.path.join(args.out_dir, f"bridge_keywords_{y}.json")
        with open(bridge_json, "w", encoding="utf-8") as f:
            json.dump(bridge.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

        edges = build_class_edges(dfy, min_freq_per_class=args.min_freq_per_class, top_edges=args.top_edges)
        edges_json = os.path.join(args.out_dir, f"class_edges_{y}.json")
        with open(edges_json, "w", encoding="utf-8") as f:
            json.dump(edges.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

        print(f"[OK] year={y} -> {inter_csv}, {bridge_json}, {edges_json}")


if __name__ == "__main__":
    main()
