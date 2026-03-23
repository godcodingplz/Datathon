#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_keyword_assets_json.py

목표:
1) (연도×중분류×키워드) 가중치(JSON) 생성
   - out_dir/weights_by_year_class/<year>/<cls_safe>_weights.json
   - out_dir/keyword_weights_all.jsonl  (전체 레코드, 1줄=1키워드)
   - out_dir/keyword_weights_all.json   (전체 레코드 배열)

2) 룰베이스 매핑 테이블(JSON) 생성 (variant -> canonical)
   - out_dir/keyword_mapping_global.json

입력(merged_dir):
- {year}_freq_merged_all.csv   (필수)
  columns: Year, NODE_CLSS_02, keyword, freq, (optional: paper_count, share, merged_variants)
- {year}_burst_merged_all.csv  (선택, 있으면 burst 반영)
  columns: Year, NODE_CLSS_02, keyword, burst, (optional: share_prev, share, freq)
- {year}_merge_map_all.csv     (선택, 있으면 mapping 신뢰도 보강)
  columns: Year, NODE_CLSS_02, variant, canonical, (optional: reason, confidence)

가중치 정의(기본):
- group = (Year, NODE_CLSS_02)
- freq_norm = freq / max(freq in group)
- burst_norm = max(burst, 0) / max(max(burst,0) in group)  (분모 0이면 0)
- weight_raw = alpha*freq_norm + beta*burst_norm
- weight = weight_raw / sum(weight_raw in group)  (합 0이면 uniform)

실행 예시:
python build_keyword_assets_json.py \
  --merged_dir "/Users/idonghyeon/Desktop/dathon_result/merged" \
  --out_dir "/Users/idonghyeon/Desktop/dathon_result/assets" \
  --years "2021,2022,2023,2024,2025" \
  --alpha 0.35 \
  --beta 0.65
"""

import os
import re
import json
import argparse
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd


# -------------------------
# Helpers
# -------------------------
def _safe_filename(s: str) -> str:
    s = str(s)
    return re.sub(r"[^0-9A-Za-z가-힣 _\-.]", "_", s).strip()


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _read_csv(path: str) -> pd.DataFrame:
    # BOM/공백 컬럼명 방어
    df = pd.read_csv(path)
    df.columns = df.columns.astype(str).str.replace("\ufeff", "").str.strip()
    return df


def _find_first(base_dir: str, candidates: List[str]) -> Optional[str]:
    for fn in candidates:
        p = os.path.join(base_dir, fn)
        if os.path.exists(p):
            return p
    return None


def _to_int_series(x, default=0) -> pd.Series:
    return pd.to_numeric(x, errors="coerce").fillna(default).astype(int)


def _to_float_series(x, default=0.0) -> pd.Series:
    return pd.to_numeric(x, errors="coerce").fillna(default).astype(float)


# -------------------------
# Weight computation
# -------------------------
def compute_weights(
    freq_df: pd.DataFrame,
    burst_df: Optional[pd.DataFrame],
    alpha: float,
    beta: float,
    min_freq: int = 1,
) -> pd.DataFrame:
    """
    Returns weights_df with columns:
    Year, NODE_CLSS_02, keyword, freq, burst, weight_raw, weight
    plus optional: paper_count, share, merged_variants
    """
    df = freq_df.copy()

    # Required cols
    for col in ["Year", "NODE_CLSS_02", "keyword", "freq"]:
        if col not in df.columns:
            raise RuntimeError(f"freq_df missing required column: {col}")

    df["Year"] = _to_int_series(df["Year"], default=0)
    df["NODE_CLSS_02"] = df["NODE_CLSS_02"].astype(str)
    df["keyword"] = df["keyword"].astype(str).str.strip()
    df["freq"] = _to_int_series(df["freq"], default=0)

    # Optional
    if "paper_count" in df.columns:
        df["paper_count"] = _to_int_series(df["paper_count"], default=0)
    if "share" in df.columns:
        df["share"] = _to_float_series(df["share"], default=0.0)
    if "merged_variants" in df.columns:
        df["merged_variants"] = df["merged_variants"].astype(str)

    # Filter
    df = df[df["freq"] >= int(min_freq)].copy()

    # Attach burst if available
    df["burst"] = 0.0
    if burst_df is not None and not burst_df.empty:
        b = burst_df.copy()
        # Required in burst to join
        needed = {"Year", "NODE_CLSS_02", "keyword"}
        if needed.issubset(set(b.columns)):
            b["Year"] = _to_int_series(b["Year"], default=0)
            b["NODE_CLSS_02"] = b["NODE_CLSS_02"].astype(str)
            b["keyword"] = b["keyword"].astype(str).str.strip()
            if "burst" in b.columns:
                b["burst"] = _to_float_series(b["burst"], default=0.0)
            else:
                b["burst"] = 0.0

            df = df.merge(
                b[["Year", "NODE_CLSS_02", "keyword", "burst"]],
                on=["Year", "NODE_CLSS_02", "keyword"],
                how="left",
                suffixes=("", "_b"),
            )
            df["burst"] = _to_float_series(df["burst"], default=0.0)
        else:
            # burst 파일이 형식이 다르면 그냥 무시
            pass

    # Groupwise normalization + weight
    gcols = ["Year", "NODE_CLSS_02"]

    # max freq per group
    df["_max_freq"] = df.groupby(gcols)["freq"].transform(lambda s: max(int(s.max()), 1))

    # positive burst + max positive burst per group
    df["_burst_pos"] = df["burst"].clip(lower=0.0)
    df["_max_burst_pos"] = df.groupby(gcols)["_burst_pos"].transform(lambda s: float(s.max()) if float(s.max()) > 0 else 0.0)

    df["freq_norm"] = df["freq"] / df["_max_freq"]
    df["burst_norm"] = 0.0
    mask = df["_max_burst_pos"] > 0
    df.loc[mask, "burst_norm"] = df.loc[mask, "_burst_pos"] / df.loc[mask, "_max_burst_pos"]

    df["weight_raw"] = (float(alpha) * df["freq_norm"]) + (float(beta) * df["burst_norm"])

    # normalize to sum=1 per group (fallback uniform if sum=0)
    df["_w_sum"] = df.groupby(gcols)["weight_raw"].transform(lambda s: float(s.sum()))
    df["weight"] = 0.0
    ok = df["_w_sum"] > 0
    df.loc[ok, "weight"] = df.loc[ok, "weight_raw"] / df.loc[ok, "_w_sum"]

    # uniform fallback
    df["_n_in_group"] = df.groupby(gcols)["keyword"].transform("count").astype(int)
    df.loc[~ok, "weight"] = 1.0 / df.loc[~ok, "_n_in_group"].replace(0, 1)

    # cleanup
    df = df.drop(columns=["_max_freq", "_burst_pos", "_max_burst_pos", "_w_sum", "_n_in_group"])

    # stable sort
    df = df.sort_values(["Year", "NODE_CLSS_02", "weight", "freq", "burst"], ascending=[True, True, False, False, False])

    return df


# -------------------------
# Write per (year, class) JSON
# -------------------------
def write_weights_by_year_class(
    weights_df: pd.DataFrame,
    out_dir: str,
    alpha: float,
    beta: float,
    top_k: Optional[int] = None,
    round_ndigits: int = 6,
) -> None:
    base_dir = os.path.join(out_dir, "weights_by_year_class")
    _ensure_dir(base_dir)

    required = {"Year", "NODE_CLSS_02", "keyword", "weight", "weight_raw", "freq", "burst"}
    missing = [c for c in required if c not in weights_df.columns]
    if missing:
        raise RuntimeError(f"weights_df missing required cols: {missing}")

    for (year, cls), g in weights_df.groupby(["Year", "NODE_CLSS_02"], dropna=False):
        year_dir = os.path.join(base_dir, str(int(year)))
        _ensure_dir(year_dir)

        g2 = g.sort_values("weight", ascending=False).copy()
        if top_k is not None and int(top_k) > 0:
            g2 = g2.head(int(top_k))

        items = []
        for _, r in g2.iterrows():
            item = {
                "keyword": str(r["keyword"]),
                "weight": round(float(r["weight"]), round_ndigits),
                "weight_raw": round(float(r["weight_raw"]), round_ndigits),
                "freq": int(r["freq"]),
                "burst": round(float(r["burst"]), round_ndigits),
            }
            if "share" in g2.columns:
                item["share"] = round(float(r.get("share", 0.0) or 0.0), round_ndigits)
            if "paper_count" in g2.columns:
                item["paper_count"] = int(float(r.get("paper_count", 0) or 0))
            if "merged_variants" in g2.columns:
                mv = r.get("merged_variants", "")
                item["merged_variants"] = mv if isinstance(mv, str) else str(mv)
            items.append(item)

        payload = {
            "year": int(year),
            "class_name": str(cls),
            "alpha": float(alpha),
            "beta": float(beta),
            "count": len(items),
            "keywords": items,
        }

        cls_safe = _safe_filename(cls)
        out_path = os.path.join(year_dir, f"{cls_safe}_weights.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[OK] weights split saved: {base_dir}")


def write_weights_all(weights_df: pd.DataFrame, out_dir: str) -> None:
    _ensure_dir(out_dir)
    jsonl_path = os.path.join(out_dir, "keyword_weights_all.jsonl")
    json_path = os.path.join(out_dir, "keyword_weights_all.json")

    # JSONL
    with open(jsonl_path, "w", encoding="utf-8") as fw:
        for _, r in weights_df.iterrows():
            obj = {k: (r[k].item() if hasattr(r[k], "item") else r[k]) for k in weights_df.columns}
            fw.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # JSON array
    arr = []
    for _, r in weights_df.iterrows():
        obj = {k: (r[k].item() if hasattr(r[k], "item") else r[k]) for k in weights_df.columns}
        arr.append(obj)

    with open(json_path, "w", encoding="utf-8") as fw:
        json.dump(arr, fw, ensure_ascii=False, indent=2)

    print(f"[OK] weights all saved: {jsonl_path}")
    print(f"[OK] weights all saved: {json_path}")


# -------------------------
# Build global mapping (variant -> canonical)
# -------------------------
def _add_variant_score(scores: Dict[Tuple[str, str], float], variant: str, canonical: str, score: float) -> None:
    v = str(variant).strip()
    c = str(canonical).strip()
    if not v or not c:
        return
    scores[(v, c)] += float(score)


def build_global_mapping(
    year_freq_dfs: List[pd.DataFrame],
    year_merge_map_dfs: List[pd.DataFrame],
) -> Dict[str, Any]:
    """
    Strategy:
    - Use merged_variants from freq_merged_all to vote variant->canonical with score=freq
    - Use merge_map_all to vote variant->canonical with score=max(0.1, confidence)  (if confidence exists)
    - Resolve conflicts by picking canonical with max total score per variant
    """
    scores: Dict[Tuple[str, str], float] = defaultdict(float)

    # 1) From freq merged_variants
    for f in year_freq_dfs:
        if f is None or f.empty:
            continue
        if not {"keyword", "freq"}.issubset(set(f.columns)):
            continue

        df = f.copy()
        df["keyword"] = df["keyword"].astype(str).str.strip()
        df["freq"] = _to_int_series(df["freq"], default=0)

        if "merged_variants" in df.columns:
            df["merged_variants"] = df["merged_variants"].astype(str)
            for _, r in df.iterrows():
                canonical = r["keyword"]
                freq = int(r["freq"])
                mv = r.get("merged_variants", "") or ""
                variants = [x.strip() for x in str(mv).split(";") if str(x).strip()]
                # canonical 자체도 variant로 포함
                variants.append(str(canonical))
                for v in set(variants):
                    _add_variant_score(scores, v, canonical, score=max(freq, 1))

        else:
            # merged_variants가 없으면 keyword 자체만 identity
            for _, r in df.iterrows():
                canonical = r["keyword"]
                freq = int(r["freq"])
                _add_variant_score(scores, canonical, canonical, score=max(freq, 1))

    # 2) From merge_map_all
    for mm in year_merge_map_dfs:
        if mm is None or mm.empty:
            continue
        if not {"variant", "canonical"}.issubset(set(mm.columns)):
            continue

        df = mm.copy()
        df["variant"] = df["variant"].astype(str).str.strip()
        df["canonical"] = df["canonical"].astype(str).str.strip()

        if "confidence" in df.columns:
            df["confidence"] = _to_float_series(df["confidence"], default=0.7)
        else:
            df["confidence"] = 0.7

        for _, r in df.iterrows():
            v = r["variant"]
            c = r["canonical"]
            conf = float(r.get("confidence", 0.7) or 0.7)
            _add_variant_score(scores, v, c, score=max(conf, 0.1))

    # Resolve
    per_variant: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for (v, c), sc in scores.items():
        per_variant[v].append((c, float(sc)))

    mapping: Dict[str, str] = {}
    for v, cand_list in per_variant.items():
        # sort by score desc, then canonical length asc (more concise tie-break)
        cand_list.sort(key=lambda x: (-x[1], len(x[0])))
        mapping[v] = cand_list[0][0]

    # Reverse index
    canon_to_vars: Dict[str, List[str]] = defaultdict(list)
    for v, c in mapping.items():
        canon_to_vars[c].append(v)

    for c in canon_to_vars.keys():
        canon_to_vars[c] = sorted(list(set(canon_to_vars[c])))

    return {
        "mapping": mapping,
        "canonical_to_variants": dict(canon_to_vars),
        "stats": {
            "num_variants": len(mapping),
            "num_canonicals": len(canon_to_vars),
        }
    }


def write_global_mapping(mapping_obj: Dict[str, Any], out_dir: str, meta: Dict[str, Any]) -> None:
    _ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "keyword_mapping_global.json")
    payload = {
        "meta": meta,
        "stats": mapping_obj.get("stats", {}),
        "mapping": mapping_obj.get("mapping", {}),
        "canonical_to_variants": mapping_obj.get("canonical_to_variants", {}),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[OK] mapping saved: {out_path}")


# -------------------------
# Main
# -------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--merged_dir", required=True, help="merged 산출물 폴더 (freq_merged_all/burst_merged_all/merge_map_all)")
    p.add_argument("--out_dir", required=True, help="출력 폴더")
    p.add_argument("--years", default="2021,2022,2023,2024,2025", help="쉼표로 구분")
    p.add_argument("--alpha", type=float, default=0.35, help="freq 가중치 계수")
    p.add_argument("--beta", type=float, default=0.65, help="burst 가중치 계수")
    p.add_argument("--min_freq", type=int, default=1, help="가중치 계산에 포함할 최소 freq")
    p.add_argument("--top_k", type=int, default=0, help="(연도×중분류) JSON에 상위 K만 저장(0이면 전체)")
    p.add_argument("--no_split_weights", action="store_true", help="연도×중분류별 weights JSON 저장 비활성화")
    p.add_argument("--no_mapping", action="store_true", help="global mapping JSON 생성 비활성화")
    p.add_argument("--preview", action="store_true")
    args = p.parse_args()

    merged_dir = args.merged_dir
    out_dir = args.out_dir
    years = [int(x.strip()) for x in str(args.years).split(",") if x.strip()]

    if not os.path.isdir(merged_dir):
        raise RuntimeError(f"--merged_dir not found: {merged_dir}")

    _ensure_dir(out_dir)

    year_freq_dfs: List[pd.DataFrame] = []
    year_burst_dfs: Dict[int, pd.DataFrame] = {}
    year_mm_dfs: List[pd.DataFrame] = []

    # Load files
    loaded_years = []
    for year in years:
        freq_path = _find_first(merged_dir, [
            f"{year}_freq_merged_all.csv",
            f"{year}_freq_merged_top100_min5.csv",
        ])
        if not freq_path:
            if args.preview:
                print(f"[SKIP] {year} no freq file in {merged_dir}")
            continue

        burst_path = _find_first(merged_dir, [
            f"{year}_burst_merged_all.csv",
            f"{year}_trend_burst_merged_all.csv",
        ])

        mm_path = _find_first(merged_dir, [
            f"{year}_merge_map_all.csv",
        ])

        fdf = _read_csv(freq_path)
        # Ensure Year column exists and consistent
        if "Year" not in fdf.columns:
            fdf["Year"] = year
        fdf["Year"] = _to_int_series(fdf["Year"], default=year)

        year_freq_dfs.append(fdf)
        loaded_years.append(year)

        if burst_path and os.path.exists(burst_path):
            bdf = _read_csv(burst_path)
            if "Year" not in bdf.columns:
                bdf["Year"] = year
            bdf["Year"] = _to_int_series(bdf["Year"], default=year)
            year_burst_dfs[year] = bdf

        if mm_path and os.path.exists(mm_path):
            mdf = _read_csv(mm_path)
            # merge_map_all에는 Year가 있을 수도 없을 수도 있음
            if "Year" not in mdf.columns:
                mdf["Year"] = year
            mdf["Year"] = _to_int_series(mdf["Year"], default=year)
            year_mm_dfs.append(mdf)

        if args.preview:
            print(f"[LOAD] {year}")
            print(f"  freq:  {os.path.basename(freq_path)} rows={len(fdf)}")
            print(f"  burst: {os.path.basename(burst_path) if burst_path else '(none)'}")
            print(f"  mm:    {os.path.basename(mm_path) if mm_path else '(none)'}")

    if not year_freq_dfs:
        raise RuntimeError("No freq merged files loaded. check --merged_dir")

    # Build one combined freq df and compute weights year-by-year with matching burst
    weights_all_parts = []
    for fdf in year_freq_dfs:
        # infer year from data
        yvals = sorted(set(_to_int_series(fdf["Year"]).tolist()))
        # usually only one year per file
        for y in yvals:
            sub_freq = fdf[fdf["Year"] == y].copy()
            sub_burst = year_burst_dfs.get(int(y), None)
            wdf = compute_weights(
                freq_df=sub_freq,
                burst_df=sub_burst,
                alpha=float(args.alpha),
                beta=float(args.beta),
                min_freq=int(args.min_freq),
            )
            weights_all_parts.append(wdf)

    weights_df = pd.concat(weights_all_parts, ignore_index=True) if weights_all_parts else pd.DataFrame()
    if weights_df.empty:
        raise RuntimeError("weights_df empty after compute_weights (check min_freq / input columns).")

    # Save all weights
    write_weights_all(weights_df, out_dir=out_dir)

    # Save split weights
    if not args.no_split_weights:
        top_k = int(args.top_k) if int(args.top_k) > 0 else None
        write_weights_by_year_class(
            weights_df=weights_df,
            out_dir=out_dir,
            alpha=float(args.alpha),
            beta=float(args.beta),
            top_k=top_k,
            round_ndigits=6,
        )

    # Global mapping
    if not args.no_mapping:
        mapping_obj = build_global_mapping(
            year_freq_dfs=year_freq_dfs,
            year_merge_map_dfs=year_mm_dfs,
        )
        meta = {
            "merged_dir": merged_dir,
            "years": loaded_years,
            "alpha": float(args.alpha),
            "beta": float(args.beta),
            "min_freq": int(args.min_freq),
            "mapping_sources": {
                "used_merged_variants": any(("merged_variants" in df.columns) for df in year_freq_dfs),
                "used_merge_map_all": bool(year_mm_dfs),
            }
        }
        write_global_mapping(mapping_obj, out_dir=out_dir, meta=meta)

    print("\nDONE")


if __name__ == "__main__":
    main()
