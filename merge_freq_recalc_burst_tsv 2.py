import os
import re
import io
import time
import argparse
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


# =========================================================
# LLM: merge_map(variant->canonical)만 생성 (TSV로 강제)
#  - 합산(freq)/share/burst는 파이썬이 정확히 수행
# =========================================================
PROMPT_MERGE_MAP_ONLY_TSV = """너는 키워드 정규화 담당자다. 아래는 특정 연도(year) 및 중분류(class)의 키워드 목록이다.
표기만 다른 경우(대소문자/공백/하이픈/복수형), 한/영 번역쌍, 약어-풀네임, 명백한 오탈자만 병합하라. 과병합 금지.

중요 규칙:
- canonical은 가능하면 영어(라틴문자) 키워드로 선택하라. (있으면 영어 우선)
- variant는 반드시 입력 목록에 실제로 존재하는 키워드만 써라.
- 모든 variant에 대해 canonical을 하나 지정하라. (병합하지 않는 경우 canonical=variant로 둬도 됨)
- 절대 코드블록(```) 쓰지 말고, 설명문 쓰지 말고, 오직 TSV 본문만 출력하라.
- 출력은 TSV(탭으로 구분)만 출력하라. (쉼표 CSV 금지)
- 각 필드 안에 탭 문자는 절대 넣지 마라.

출력 TSV 헤더(반드시 그대로):
variant\tcanonical\treason\tconfidence

reason 값은 아래 중 하나:
casing|spacing|plural|typo|acronym|ko-en|same

입력(키워드 목록):
{kw_list}
"""


# =========================================================
# OpenAI 호출 + 응답 텍스트 추출
# =========================================================
def call_llm(client: OpenAI, model: str, prompt: str, max_output_tokens: int = 1200) -> str:
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


def strip_code_fences(text: str) -> str:
    if not text:
        return ""
    lines = []
    for ln in text.splitlines():
        if ln.strip().startswith("```"):
            continue
        lines.append(ln)
    return "\n".join(lines).strip()


def read_tsv_strip_fence_text(text: str) -> pd.DataFrame:
    cleaned = strip_code_fences(text)
    if not cleaned:
        return pd.DataFrame()
    # engine="python"으로 관대하게 파싱
    return pd.read_csv(io.StringIO(cleaned), sep="\t", dtype=str, engine="python")


def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def safe_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in " _-." else "_" for c in s).strip()


# =========================================================
# 1) (year, class)별 merge_map 생성(LLM)
# 2) merge_map 기준으로 파이썬이 freq 합산(정확)
# =========================================================
def build_merge_map_llm(
    client: OpenAI,
    model: str,
    keywords: list[str],
    sleep_sec: float = 0.2,
    max_output_tokens: int = 1200,
    preview: bool = False,
    dump_on_fail_path: str | None = "last_llm_mergemap_raw.txt",
) -> pd.DataFrame:
    kw_list = "\n".join([k for k in keywords if str(k).strip()])
    prompt = PROMPT_MERGE_MAP_ONLY_TSV.format(kw_list=kw_list)

    text = ""
    for attempt in range(3):
        try:
            text = call_llm(client, model, prompt, max_output_tokens=max_output_tokens)
            break
        except Exception as e:
            print(f"[ERR] merge_map LLM attempt={attempt+1}: {e}")
            time.sleep(2 * (attempt + 1))

    if preview:
        print("\n----- MERGE_MAP TSV PREVIEW -----")
        print((text[:900] + "...") if text else "(EMPTY)")
        print("----- END PREVIEW -----\n")

    # 원본 덤프(디버깅)
    if dump_on_fail_path:
        try:
            with open(dump_on_fail_path, "w", encoding="utf-8") as f:
                f.write(text or "")
        except Exception:
            pass

    text = strip_code_fences(text)
    mm = read_tsv_strip_fence_text(text)

    if mm.empty or not {"variant", "canonical"}.issubset(set(mm.columns)):
        print("[WARN] merge_map parse failed. fallback to identity mapping.")
        mm = pd.DataFrame({
            "variant": keywords,
            "canonical": keywords,
            "reason": ["same"] * len(keywords),
            "confidence": ["1.0"] * len(keywords)
        })

    mm["variant"] = mm["variant"].astype(str).str.strip()
    mm["canonical"] = mm["canonical"].astype(str).str.strip()
    mm["variant_norm"] = mm["variant"].map(_norm)

    # variant_norm 중복 제거(첫번째 우선)
    mm = mm.drop_duplicates(subset=["variant_norm"], keep="first").copy()

    time.sleep(sleep_sec)
    return mm


def apply_merge_map_and_aggregate(freq_df: pd.DataFrame, mm: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    freq_df columns expected: Year, NODE_CLSS_02, keyword, freq, paper_count
    mm columns: variant, canonical, variant_norm
    """
    df = freq_df.copy()
    df["keyword"] = df["keyword"].astype(str).str.strip()
    df["keyword_norm"] = df["keyword"].map(_norm)

    mp = dict(zip(mm["variant_norm"], mm["canonical"]))

    df["canonical"] = df["keyword_norm"].map(lambda x: mp.get(x, None))
    df["canonical"] = df["canonical"].fillna(df["keyword"])

    match_num = int(df["keyword_norm"].isin(set(mp.keys())).sum())
    match_den = int(len(df))

    # merged_variants(원래 키워드들;로 나열)
    merged_variants = (
        df.groupby("canonical")["keyword"]
          .apply(lambda s: ";".join(sorted(set([x for x in s if str(x).strip()]))))
          .reset_index()
          .rename(columns={"keyword": "merged_variants"})
    )

    # freq 합산 + paper_count는 최대값 유지(연도×중분류 단위라 동일해야 정상)
    df["freq"] = pd.to_numeric(df["freq"], errors="coerce").fillna(0).astype(int)
    if "paper_count" in df.columns:
        df["paper_count"] = pd.to_numeric(df["paper_count"], errors="coerce").fillna(0).astype(int)

    agg_ops = {"freq": "sum"}
    if "paper_count" in df.columns:
        agg_ops["paper_count"] = "max"

    out = (
        df.groupby(["Year", "NODE_CLSS_02", "canonical"], as_index=False)
          .agg(agg_ops)
          .rename(columns={"canonical": "keyword"})
    )

    out = out.merge(
        merged_variants.rename(columns={"canonical": "keyword"}),
        on="keyword",
        how="left"
    )

    # share 재계산
    if "paper_count" in out.columns:
        out["share"] = out["freq"] / out["paper_count"].replace(0, pd.NA)
        out["share"] = out["share"].fillna(0.0)
    else:
        out["share"] = 0.0

    # merge stats
    group_sizes = df.groupby("canonical")["keyword"].nunique()
    merged_groups = int((group_sizes >= 2).sum())
    merged_variants_n = int(group_sizes.sum() - group_sizes.size)

    stats = {
        "match_rate": f"{match_num}/{match_den}",
        "merged_groups": merged_groups,
        "merged_variants": merged_variants_n,
    }
    return out, stats


# =========================================================
# burst 재계산: share(Y) - share(Y-1)
# =========================================================
def recalc_burst(merged_all_years: pd.DataFrame) -> pd.DataFrame:
    base_cols = ["Year", "NODE_CLSS_02", "keyword", "freq", "paper_count", "share", "merged_variants"]
    base = merged_all_years[base_cols].copy()

    prev = base[["Year", "NODE_CLSS_02", "keyword", "share"]].copy()
    prev["Year"] = prev["Year"].astype(int) + 1
    prev = prev.rename(columns={"share": "share_prev"})

    out = base.merge(prev, on=["Year", "NODE_CLSS_02", "keyword"], how="left")
    out["share_prev"] = pd.to_numeric(out["share_prev"], errors="coerce").fillna(0.0)
    out["share"] = pd.to_numeric(out["share"], errors="coerce").fillna(0.0)
    out["burst"] = out["share"] - out["share_prev"]
    return out


def make_topn_per_class(df: pd.DataFrame, top_n: int, sort_col: str, min_freq: int) -> pd.DataFrame:
    x = df.copy()
    x["freq"] = pd.to_numeric(x["freq"], errors="coerce").fillna(0).astype(int)

    if sort_col in x.columns:
        x[sort_col] = pd.to_numeric(x[sort_col], errors="coerce").fillna(0.0)

    x = x[x["freq"] >= min_freq].copy()
    x = (
        x.sort_values(["NODE_CLSS_02", sort_col], ascending=[True, False])
         .groupby("NODE_CLSS_02", as_index=False, group_keys=False)
         .head(top_n)
    )
    return x


def make_mix_topn(freq_all: pd.DataFrame, burst_all: pd.DataFrame, top_n: int, min_freq: int) -> pd.DataFrame:
    """
    1) burst>0 topN
    2) 부족하면 freq top으로 채움(이미 포함된 keyword 제외)
    """
    out_rows = []
    for cls in sorted(freq_all["NODE_CLSS_02"].dropna().unique().tolist()):
        f = freq_all[freq_all["NODE_CLSS_02"] == cls].copy()
        b = burst_all[burst_all["NODE_CLSS_02"] == cls].copy()

        b["burst"] = pd.to_numeric(b["burst"], errors="coerce").fillna(0.0)
        b["freq"] = pd.to_numeric(b["freq"], errors="coerce").fillna(0).astype(int)
        f["freq"] = pd.to_numeric(f["freq"], errors="coerce").fillna(0).astype(int)

        b_sel = (
            b[(b["burst"] > 0) & (b["freq"] >= min_freq)]
            .sort_values("burst", ascending=False)
            .head(top_n)
        )
        picked = set(b_sel["keyword"].astype(str).tolist())

        need = top_n - len(b_sel)
        if need > 0:
            f_fill = (
                f[(f["freq"] >= min_freq) & (~f["keyword"].astype(str).isin(picked))]
                .sort_values("freq", ascending=False)
                .head(need)
            )
            mix = pd.concat([b_sel, f_fill], ignore_index=True)
        else:
            mix = b_sel

        out_rows.append(mix)

    if not out_rows:
        return pd.DataFrame()
    return pd.concat(out_rows, ignore_index=True)


def save_year_class_splits(df: pd.DataFrame, year: int, out_dir: str, subfolder_name: str, sort_col: str):
    """
    df: Year 단일 값으로 필터된 DF
    sort_col: 'freq' 또는 'burst'
    """
    per_class_dir = os.path.join(out_dir, subfolder_name)
    os.makedirs(per_class_dir, exist_ok=True)

    for cls in sorted(df["NODE_CLSS_02"].dropna().unique().tolist()):
        sub_cls = df[df["NODE_CLSS_02"] == cls].copy()

        if sort_col in sub_cls.columns:
            sub_cls[sort_col] = pd.to_numeric(sub_cls[sort_col], errors="coerce").fillna(0)
            sub_cls = sub_cls.sort_values(sort_col, ascending=False)

        out_cls = os.path.join(per_class_dir, f"{year}_{safe_filename(str(cls))}_{subfolder_name}.csv")
        sub_cls.to_csv(out_cls, index=False, encoding="utf-8-sig")


def make_summary_table(merged_all_years_df: pd.DataFrame, burst_all_df: pd.DataFrame) -> pd.DataFrame:
    """
    year×class 요약표:
    - num_keywords(병합 후 키워드 수)
    - total_freq(합산 freq)
    - num_merged_groups(merged_variants에 ';' 포함 = 실제 병합 발생)
    - positive_burst_keywords( burst>0 키워드 수 )
    """
    df = merged_all_years_df.copy()
    df["freq"] = pd.to_numeric(df["freq"], errors="coerce").fillna(0).astype(int)
    df["merged_flag"] = df["merged_variants"].astype(str).str.contains(";")

    g1 = df.groupby(["Year", "NODE_CLSS_02"], as_index=False).agg(
        num_keywords=("keyword", "nunique"),
        total_freq=("freq", "sum"),
        num_merged_keywords=("merged_flag", "sum"),
        paper_count=("paper_count", "max")
    )

    b = burst_all_df.copy()
    b["burst"] = pd.to_numeric(b["burst"], errors="coerce").fillna(0.0)
    g2 = b.groupby(["Year", "NODE_CLSS_02"], as_index=False).agg(
        positive_burst_keywords=("burst", lambda s: int((s > 0).sum())),
        max_burst=("burst", "max")
    )

    out = g1.merge(g2, on=["Year", "NODE_CLSS_02"], how="left")
    out["positive_burst_keywords"] = out["positive_burst_keywords"].fillna(0).astype(int)
    out["max_burst"] = out["max_burst"].fillna(0.0)
    return out.sort_values(["Year", "NODE_CLSS_02"])


# =========================================================
# main
# =========================================================
def run(
    input_dir: str,
    out_dir: str,
    years: list[int],
    top_n: int,
    min_freq: int,
    sleep_sec: float,
    max_tokens: int,
    preview: bool,
    dump_raw_llm: bool,
):
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY 환경변수가 없습니다. export OPENAI_API_KEY=... 로 설정하세요.")

    model = os.getenv("OPENAI_MODEL", "gpt-5-nano")
    client = OpenAI()

    os.makedirs(out_dir, exist_ok=True)
    mm_dir = os.path.join(out_dir, "_merge_maps")
    os.makedirs(mm_dir, exist_ok=True)

    merged_all_years = []

    # ---------- 1) 연도별 freq merge ----------
    for year in years:
        freq_path = os.path.join(input_dir, f"{year}_freq_top100_min5.csv")
        if not os.path.exists(freq_path):
            print(f"[SKIP] {freq_path} 없음")
            continue

        freq_df = pd.read_csv(freq_path)

        if "Year" not in freq_df.columns:
            freq_df["Year"] = year
        freq_df["Year"] = pd.to_numeric(freq_df["Year"], errors="coerce").fillna(year).astype(int)

        # paper_count 없으면 share 계산이 약해짐(가능하면 생성단계에서 포함 권장)
        if "paper_count" not in freq_df.columns:
            freq_df["paper_count"] = 0

        merged_rows = []
        merge_map_rows = []

        for cls in sorted(freq_df["NODE_CLSS_02"].dropna().unique().tolist()):
            sub = freq_df[freq_df["NODE_CLSS_02"] == cls].copy()
            sub = sub[["Year", "NODE_CLSS_02", "keyword", "freq", "paper_count"]].copy()
            sub["keyword"] = sub["keyword"].astype(str).str.strip()

            keywords = sub["keyword"].astype(str).tolist()

            mm_path = os.path.join(mm_dir, f"{year}_{safe_filename(str(cls))}_merge_map.tsv")

            # 캐시된 merge_map 있으면 재사용
            if os.path.exists(mm_path):
                mm = pd.read_csv(mm_path, sep="\t", dtype=str, engine="python")
                # 내부 매칭용 norm 컬럼 보정
                if "variant_norm" not in mm.columns:
                    if "variant" not in mm.columns or "canonical" not in mm.columns:
                        print(f"[WARN] invalid merge_map file: {mm_path} -> identity fallback")
                        mm = pd.DataFrame({
                            "variant": keywords,
                            "canonical": keywords,
                            "reason": ["same"] * len(keywords),
                            "confidence": ["1.0"] * len(keywords)
                        })
                    mm["variant"] = mm["variant"].astype(str).str.strip()
                    mm["canonical"] = mm["canonical"].astype(str).str.strip()
                    mm["variant_norm"] = mm["variant"].map(_norm)
            else:
                dump_path = None
                if dump_raw_llm:
                    dump_path = os.path.join(out_dir, "last_llm_mergemap_raw.txt")

                mm = build_merge_map_llm(
                    client, model, keywords,
                    sleep_sec=sleep_sec,
                    max_output_tokens=max_tokens,
                    preview=preview,
                    dump_on_fail_path=dump_path
                )

                # TSV로 저장(variant_norm은 내부용이라 제외)
                mm_save = mm.drop(columns=["variant_norm"], errors="ignore").copy()
                mm_save.to_csv(mm_path, index=False, sep="\t", encoding="utf-8-sig")
                print(f"[OK] merge_map saved: {mm_path}")

            merged_sub, stats = apply_merge_map_and_aggregate(sub, mm)

            print(
                f"[OK] {year} / {cls} merge stats: "
                f"match_rate={stats['match_rate']}, merged_groups={stats['merged_groups']}, merged_variants={stats['merged_variants']}"
            )

            merged_rows.append(merged_sub)

            # merge_map 출력용(Year/Class 붙여서 누적)
            mm_out = mm.drop(columns=["variant_norm"], errors="ignore").copy()
            mm_out["Year"] = year
            mm_out["NODE_CLSS_02"] = cls
            keep_cols = ["Year", "NODE_CLSS_02", "variant", "canonical"]
            for c in ["reason", "confidence"]:
                if c in mm_out.columns:
                    keep_cols.append(c)
            merge_map_rows.append(mm_out[keep_cols])

        merged_all = pd.concat(merged_rows, ignore_index=True) if merged_rows else pd.DataFrame()
        merge_map_all = pd.concat(merge_map_rows, ignore_index=True) if merge_map_rows else pd.DataFrame()

        if merged_all.empty:
            print(f"[WARN] {year} merged_all empty")
            continue

        # 저장: 연도별 merged freq(통합)
        out_freq_all = os.path.join(out_dir, f"{year}_freq_merged_all.csv")
        merged_all.to_csv(out_freq_all, index=False, encoding="utf-8-sig")
        print(f"[OK] {out_freq_all}")

        # 저장: 연도×중분류별 merged freq
        save_year_class_splits(
            df=merged_all,
            year=year,
            out_dir=out_dir,
            subfolder_name=f"{year}_freq_merged_by_class",
            sort_col="freq"
        )
        print(f"[OK] per-class freq saved: {os.path.join(out_dir, f'{year}_freq_merged_by_class')}")

        # 저장: merge_map(연도 통합본)
        out_mm_all = os.path.join(out_dir, f"{year}_merge_map_all.csv")
        merge_map_all.to_csv(out_mm_all, index=False, encoding="utf-8-sig")
        print(f"[OK] {out_mm_all}")

        # topN(freq)
        freq_top = make_topn_per_class(merged_all, top_n=top_n, sort_col="freq", min_freq=min_freq)
        out_freq_top = os.path.join(out_dir, f"{year}_freq_merged_top{top_n}_min{min_freq}.csv")
        freq_top.to_csv(out_freq_top, index=False, encoding="utf-8-sig")
        print(f"[OK] {out_freq_top}")

        merged_all_years.append(merged_all)

    # ---------- 2) burst 재계산(연도 전체 합쳐서 prev year 사용) ----------
    if not merged_all_years:
        print("[WARN] no merged data, stop.")
        return

    merged_all_years_df = pd.concat(merged_all_years, ignore_index=True)
    burst_all = recalc_burst(merged_all_years_df)

    # 요약표 저장(year×class)
    summary = make_summary_table(merged_all_years_df, burst_all)
    out_summary = os.path.join(out_dir, "summary_year_class.csv")
    summary.to_csv(out_summary, index=False, encoding="utf-8-sig")
    print(f"[OK] {out_summary}")

    # ---------- 3) 연도별 burst/topN/mix 저장 ----------
    for year in years:
        yb = burst_all[burst_all["Year"] == year].copy()
        if yb.empty:
            continue

        out_burst_all = os.path.join(out_dir, f"{year}_burst_merged_all.csv")
        yb.to_csv(out_burst_all, index=False, encoding="utf-8-sig")
        print(f"[OK] {out_burst_all}")

        # 저장: 연도×중분류별 burst
        save_year_class_splits(
            df=yb,
            year=year,
            out_dir=out_dir,
            subfolder_name=f"{year}_burst_merged_by_class",
            sort_col="burst"
        )
        print(f"[OK] per-class burst saved: {os.path.join(out_dir, f'{year}_burst_merged_by_class')}")

        # burst TopN
        yb2 = yb.copy()
        yb2["burst"] = pd.to_numeric(yb2["burst"], errors="coerce").fillna(0.0)
        burst_top = make_topn_per_class(yb2[yb2["burst"] > 0].copy(), top_n=top_n, sort_col="burst", min_freq=min_freq)
        out_burst_top = os.path.join(out_dir, f"{year}_trend_burst_merged_top{top_n}_min{min_freq}.csv")
        burst_top.to_csv(out_burst_top, index=False, encoding="utf-8-sig")
        print(f"[OK] {out_burst_top}")

        # mix TopN (burst 부족하면 freq로 채움)
        yf = merged_all_years_df[merged_all_years_df["Year"] == year].copy()
        mix_top = make_mix_topn(yf, yb, top_n=top_n, min_freq=min_freq)
        out_mix = os.path.join(out_dir, f"{year}_trend_mix_merged_top{top_n}_min{min_freq}.csv")
        mix_top.to_csv(out_mix, index=False, encoding="utf-8-sig")
        print(f"[OK] {out_mix}")

    print("\nDONE: (LLM merge_map=TSV) freq merge -> share 재계산 -> burst 재계산 -> topN/mix 생성 + 연도×중분류 split 저장 완료")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True, help="원본 freq CSV들이 있는 폴더 (예: YYYY_freq_top100_min5.csv)")
    p.add_argument("--out_dir", required=True, help="출력 폴더")
    p.add_argument("--years", default="2021,2022,2023,2024,2025", help="쉼표로 구분")
    p.add_argument("--top_n", type=int, default=100)
    p.add_argument("--min_freq", type=int, default=5)
    p.add_argument("--sleep", type=float, default=0.2)
    p.add_argument("--max_tokens", type=int, default=1200)
    p.add_argument("--preview", action="store_true")
    p.add_argument("--dump_raw_llm", action="store_true", help="LLM 원문 출력(디버깅용) 저장")
    args = p.parse_args()

    years = [int(x.strip()) for x in args.years.split(",") if x.strip()]
    run(
        input_dir=args.input_dir,
        out_dir=args.out_dir,
        years=years,
        top_n=args.top_n,
        min_freq=args.min_freq,
        sleep_sec=args.sleep,
        max_tokens=args.max_tokens,
        preview=args.preview,
        dump_raw_llm=args.dump_raw_llm
    )
