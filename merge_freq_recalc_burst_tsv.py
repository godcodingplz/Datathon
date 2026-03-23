import os
import re
import time
import argparse
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


# =========================================================
# ✅ Confidence Gate Policy (UPDATED)
#  - conf >= 0.85 : 병합 적용(그대로 canonical 사용) + auto 리스트 저장
#  - 0.60 <= conf < 0.85 : 병합 적용(그대로 canonical 사용) + review 리스트 저장
#  - conf < 0.60 : 병합 미적용(canonical=variant 강제) + hold 리스트 저장
# =========================================================
CONF_HIGH = 0.85
CONF_MID = 0.60


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


def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def safe_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in " _-." else "_" for c in s).strip()


# =========================================================
# ✅ TSV Robust Parser (LLM output 안정화 핵심)
# =========================================================
ALLOWED_REASON = {"casing", "spacing", "plural", "typo", "acronym", "ko-en", "same"}


def _is_float(x: str) -> bool:
    try:
        float(str(x).strip())
        return True
    except Exception:
        return False


def parse_merge_map_tsv_robust(text: str, keywords: list[str] | None = None, verbose: bool = True) -> pd.DataFrame:
    cleaned = strip_code_fences(text or "")
    cleaned = cleaned.replace("\ufeff", "").strip()  # BOM 제거

    if not cleaned:
        return pd.DataFrame()

    lines = [ln.rstrip("\n") for ln in cleaned.splitlines() if str(ln).strip()]
    if not lines:
        return pd.DataFrame()

    # 헤더 처리
    header = lines[0].strip()
    data_lines = lines[1:] if header.lower().replace(" ", "")[:7] == "variant" else lines

    records = []
    dropped = 0
    fixed = 0

    for idx, raw in enumerate(data_lines, start=1):
        line = str(raw).strip()
        line = line.rstrip("\t")  # trailing tab 제거

        parts = line.split("\t")
        while parts and parts[-1] == "":
            parts = parts[:-1]

        if len(parts) < 2:
            dropped += 1
            if verbose:
                print(f"[WARN] drop line (too few cols) #{idx}: {line}")
            continue

        variant = parts[0].strip()
        canonical = parts[1].strip()
        reason = "same"
        confidence = "1.0"

        if len(parts) == 2:
            pass
        elif len(parts) == 3:
            reason = parts[2].strip() or "same"
        elif len(parts) == 4:
            reason = parts[2].strip() or "same"
            confidence = parts[3].strip() or "1.0"
        else:
            fixed += 1
            variant = parts[0].strip()
            canonical = parts[1].strip()

            reason_candidate = None
            for p in parts[2:]:
                if p.strip() in ALLOWED_REASON:
                    reason_candidate = p.strip()
                    break
            reason = reason_candidate or "same"

            conf_candidate = None
            for p in reversed(parts[2:]):
                if _is_float(p.strip()):
                    conf_candidate = p.strip()
                    break
            confidence = conf_candidate or "0.7"

            if verbose:
                print(f"[WARN] fix malformed TSV line #{idx}: cols={len(parts)} -> recovered")

        if reason not in ALLOWED_REASON:
            reason = "same"

        if not _is_float(confidence):
            confidence = "0.7"
        else:
            cf = float(confidence)
            if cf < 0:
                confidence = "0.0"
            elif cf > 1.0:
                confidence = "1.0"
            else:
                confidence = str(cf)

        variant = variant.replace("\t", " ").strip()
        canonical = canonical.replace("\t", " ").strip()

        records.append({
            "variant": variant,
            "canonical": canonical,
            "reason": reason,
            "confidence": confidence
        })

    df = pd.DataFrame(records)
    if df.empty:
        return df

    # coverage 보정(누락 키워드는 identity 추가)
    if keywords:
        kw_norms = {_norm(k): k for k in keywords if str(k).strip()}
        df["variant"] = df["variant"].astype(str).str.strip()
        df["canonical"] = df["canonical"].astype(str).str.strip()
        df["variant_norm"] = df["variant"].map(_norm)

        existing = set(df["variant_norm"].dropna().tolist())
        missing_norms = [n for n in kw_norms.keys() if n not in existing]

        if missing_norms:
            add_rows = []
            for n in missing_norms:
                k = kw_norms[n]
                add_rows.append({
                    "variant": k,
                    "canonical": k,
                    "reason": "same",
                    "confidence": "1.0",
                    "variant_norm": n
                })
            df = pd.concat([df, pd.DataFrame(add_rows)], ignore_index=True)

        df = df.drop_duplicates(subset=["variant_norm"], keep="first").copy()
    else:
        df["variant_norm"] = df["variant"].map(_norm)
        df = df.drop_duplicates(subset=["variant_norm"], keep="first").copy()

    if verbose:
        print(f"[INFO] merge_map parsed: rows={len(df)}, dropped={dropped}, fixed={fixed}")
    return df


# =========================================================
# ✅ Confidence Gate Tagger (UPDATED POLICY)
#  - auto/review는 병합 적용(그대로 canonical 사용)
#  - hold만 병합 미적용(canonical=variant)
#  - 리스트 저장을 위해 gate/canonical_suggested/canonical_used를 넣음
# =========================================================
def tag_confidence_gate(
    mm: pd.DataFrame,
    high: float = CONF_HIGH,
    mid: float = CONF_MID,
) -> pd.DataFrame:
    if mm is None or mm.empty:
        return mm

    mm = mm.copy()
    mm["variant"] = mm.get("variant", "").astype(str).str.strip()
    mm["canonical"] = mm.get("canonical", "").astype(str).str.strip()

    # LLM 제안 canonical 보존
    if "canonical_suggested" not in mm.columns:
        mm["canonical_suggested"] = mm["canonical"]

    # confidence 정규화
    if "confidence" not in mm.columns:
        mm["confidence"] = 1.0
    mm["confidence"] = pd.to_numeric(mm["confidence"], errors="coerce").fillna(0.7)
    mm["confidence"] = mm["confidence"].clip(lower=0.0, upper=1.0)

    # gate 부여
    mm["gate"] = "review"  # 기본을 review로 둬도 무방
    mm.loc[mm["confidence"] >= high, "gate"] = "auto"
    mm.loc[mm["confidence"] < mid, "gate"] = "hold"

    # 실제 적용 canonical(used)
    mm["canonical_used"] = mm["canonical_suggested"]

    # hold만 병합 미적용
    hold_mask = (mm["gate"] == "hold")
    mm.loc[hold_mask, "canonical_used"] = mm.loc[hold_mask, "variant"]

    # downstream은 canonical 컬럼을 사용하므로 canonical=canonical_used로 맞춤
    mm["canonical"] = mm["canonical_used"]

    return mm


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

    if dump_on_fail_path:
        try:
            with open(dump_on_fail_path, "w", encoding="utf-8") as f:
                f.write(text or "")
        except Exception:
            pass

    mm = parse_merge_map_tsv_robust(text, keywords=keywords, verbose=preview)

    if mm.empty or not {"variant", "canonical", "variant_norm"}.issubset(set(mm.columns)):
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

    time.sleep(sleep_sec)
    return mm


def apply_merge_map_and_aggregate(freq_df: pd.DataFrame, mm: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    freq_df columns expected: Year, NODE_CLSS_02, keyword, freq, paper_count
    mm columns: variant, canonical, variant_norm (canonical is "used" canonical after gate)
    """
    df = freq_df.copy()
    df["keyword"] = df["keyword"].astype(str).str.strip()
    df["keyword_norm"] = df["keyword"].map(_norm)

    mp = dict(zip(mm["variant_norm"], mm["canonical"]))
    df["canonical"] = df["keyword_norm"].map(lambda x: mp.get(x, None))
    df["canonical"] = df["canonical"].fillna(df["keyword"])

    match_num = int(df["keyword_norm"].isin(set(mp.keys())).sum())
    match_den = int(len(df))

    merged_variants = (
        df.groupby("canonical")["keyword"]
          .apply(lambda s: ";".join(sorted(set([x for x in s if str(x).strip()]))))
          .reset_index()
          .rename(columns={"keyword": "merged_variants"})
    )

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

    if "paper_count" in out.columns:
        out["share"] = out["freq"] / out["paper_count"].replace(0, pd.NA)
        out["share"] = out["share"].fillna(0.0)
    else:
        out["share"] = 0.0

    group_sizes = df.groupby("canonical")["keyword"].nunique()
    merged_groups = int((group_sizes >= 2).sum())
    merged_variants_n = int(group_sizes.sum() - group_sizes.size)

    stats = {
        "match_rate": f"{match_num}/{match_den}",
        "merged_groups": merged_groups,
        "merged_variants": merged_variants_n,
    }
    return out, stats


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

        if "paper_count" not in freq_df.columns:
            freq_df["paper_count"] = 0  # ⚠️ 가능하면 생성단계에서 포함 권장

        merged_rows = []
        merge_map_rows = []

        for cls in sorted(freq_df["NODE_CLSS_02"].dropna().unique().tolist()):
            sub = freq_df[freq_df["NODE_CLSS_02"] == cls].copy()
            sub = sub[["Year", "NODE_CLSS_02", "keyword", "freq", "paper_count"]].copy()
            sub["keyword"] = sub["keyword"].astype(str).str.strip()

            keywords = sub["keyword"].astype(str).tolist()

            mm_path = os.path.join(mm_dir, f"{year}_{safe_filename(str(cls))}_merge_map.tsv")

            # ✅ 리스트 파일(자동/검토/보류)
            mm_auto_path = os.path.join(mm_dir, f"{year}_{safe_filename(str(cls))}_merge_map_auto.tsv")
            mm_review_path = os.path.join(mm_dir, f"{year}_{safe_filename(str(cls))}_merge_map_review.tsv")
            mm_hold_path = os.path.join(mm_dir, f"{year}_{safe_filename(str(cls))}_merge_map_hold.tsv")

            # 캐시된 merge_map 있으면 재사용
            if os.path.exists(mm_path):
                mm = pd.read_csv(mm_path, sep="\t", dtype=str, engine="python")

                # backward compatibility: gate/canonical_suggested 없으면 채움
                if "canonical_suggested" not in mm.columns and "canonical" in mm.columns:
                    mm["canonical_suggested"] = mm["canonical"]

                if "confidence" not in mm.columns:
                    mm["confidence"] = "1.0"

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

                # ✅ 캐시 파일에도 누락 키워드가 있을 수 있으니 보정
                mm["variant"] = mm["variant"].astype(str).str.strip()
                mm["canonical"] = mm["canonical"].astype(str).str.strip()
                mm["variant_norm"] = mm["variant"].map(_norm)

                existing = set(mm["variant_norm"].dropna().tolist())
                for k in keywords:
                    n = _norm(k)
                    if n not in existing:
                        mm = pd.concat([mm, pd.DataFrame([{
                            "variant": k, "canonical": k, "reason": "same", "confidence": "1.0",
                            "canonical_suggested": k,
                            "variant_norm": n
                        }])], ignore_index=True)

                mm = mm.drop_duplicates(subset=["variant_norm"], keep="first").copy()

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

                # variant_norm은 내부 매칭용
                if "variant_norm" not in mm.columns:
                    mm["variant_norm"] = mm["variant"].map(_norm)

            # ✅ Confidence gate tagging (UPDATED POLICY)
            mm = tag_confidence_gate(mm, high=CONF_HIGH, mid=CONF_MID)

            # ✅ 리스트 저장: (auto/review/hold) — 병합된 것만 보고 싶으면 "suggested!=variant"로 필터링
            def _only_changed(xdf: pd.DataFrame) -> pd.DataFrame:
                if xdf is None or xdf.empty:
                    return xdf
                # LLM이 병합을 제안한(변경) 건만 리스트로 뽑기
                return xdf[xdf["canonical_suggested"].astype(str) != xdf["variant"].astype(str)].copy()

            auto_df = _only_changed(mm[mm["gate"] == "auto"].copy())
            review_df = _only_changed(mm[mm["gate"] == "review"].copy())
            hold_df = _only_changed(mm[mm["gate"] == "hold"].copy())

            if not auto_df.empty:
                auto_df.drop(columns=["variant_norm"], errors="ignore").to_csv(
                    mm_auto_path, index=False, sep="\t", encoding="utf-8-sig"
                )

            if not review_df.empty:
                review_df.drop(columns=["variant_norm"], errors="ignore").to_csv(
                    mm_review_path, index=False, sep="\t", encoding="utf-8-sig"
                )

            if not hold_df.empty:
                hold_df.drop(columns=["variant_norm"], errors="ignore").to_csv(
                    mm_hold_path, index=False, sep="\t", encoding="utf-8-sig"
                )

            if preview:
                print(
                    f"[INFO] gate counts {year}/{cls}: "
                    f"auto={int((mm['gate']=='auto').sum())}, "
                    f"review={int((mm['gate']=='review').sum())}, "
                    f"hold={int((mm['gate']=='hold').sum())}"
                )

            # ✅ 메인 merge_map 저장(항상 갱신 저장을 원치 않으면 조건 걸어도 됨)
            mm_save = mm.drop(columns=["variant_norm"], errors="ignore").copy()
            # 권장 저장 컬럼 순서 정렬
            ordered = ["variant", "canonical", "canonical_suggested", "canonical_used", "reason", "confidence", "gate"]
            cols = [c for c in ordered if c in mm_save.columns] + [c for c in mm_save.columns if c not in ordered]
            mm_save = mm_save[cols]
            mm_save.to_csv(mm_path, index=False, sep="\t", encoding="utf-8-sig")

            # ✅ 병합 적용(hold만 미적용 / auto+review는 적용)
            merged_sub, stats = apply_merge_map_and_aggregate(sub, mm)

            print(
                f"[OK] {year} / {cls} merge stats: "
                f"match_rate={stats['match_rate']}, merged_groups={stats['merged_groups']}, merged_variants={stats['merged_variants']}, "
                f"gates(auto/review/hold)={int((mm['gate']=='auto').sum())}/{int((mm['gate']=='review').sum())}/{int((mm['gate']=='hold').sum())}"
            )

            merged_rows.append(merged_sub)

            # merge_map 출력용(Year/Class 붙여서 누적)
            mm_out = mm.drop(columns=["variant_norm"], errors="ignore").copy()
            mm_out["Year"] = year
            mm_out["NODE_CLSS_02"] = cls

            keep_cols = ["Year", "NODE_CLSS_02", "variant", "canonical"]
            for c in ["canonical_suggested", "canonical_used", "reason", "confidence", "gate"]:
                if c in mm_out.columns:
                    keep_cols.append(c)

            merge_map_rows.append(mm_out[keep_cols])

        merged_all = pd.concat(merged_rows, ignore_index=True) if merged_rows else pd.DataFrame()
        merge_map_all = pd.concat(merge_map_rows, ignore_index=True) if merge_map_rows else pd.DataFrame()

        if merged_all.empty:
            print(f"[WARN] {year} merged_all empty")
            continue

        out_freq_all = os.path.join(out_dir, f"{year}_freq_merged_all.csv")
        merged_all.to_csv(out_freq_all, index=False, encoding="utf-8-sig")
        print(f"[OK] {out_freq_all}")

        save_year_class_splits(
            df=merged_all,
            year=year,
            out_dir=out_dir,
            subfolder_name=f"{year}_freq_merged_by_class",
            sort_col="freq"
        )
        print(f"[OK] per-class freq saved: {os.path.join(out_dir, f'{year}_freq_merged_by_class')}")

        out_mm_all = os.path.join(out_dir, f"{year}_merge_map_all.csv")
        merge_map_all.to_csv(out_mm_all, index=False, encoding="utf-8-sig")
        print(f"[OK] {out_mm_all}")

        freq_top = make_topn_per_class(merged_all, top_n=top_n, sort_col="freq", min_freq=min_freq)
        out_freq_top = os.path.join(out_dir, f"{year}_freq_merged_top{top_n}_min{min_freq}.csv")
        freq_top.to_csv(out_freq_top, index=False, encoding="utf-8-sig")
        print(f"[OK] {out_freq_top}")

        merged_all_years.append(merged_all)

    # ---------- 2) burst 재계산 ----------
    if not merged_all_years:
        print("[WARN] no merged data, stop.")
        return

    merged_all_years_df = pd.concat(merged_all_years, ignore_index=True)
    burst_all = recalc_burst(merged_all_years_df)

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

        save_year_class_splits(
            df=yb,
            year=year,
            out_dir=out_dir,
            subfolder_name=f"{year}_burst_merged_by_class",
            sort_col="burst"
        )
        print(f"[OK] per-class burst saved: {os.path.join(out_dir, f'{year}_burst_merged_by_class')}")

        yb2 = yb.copy()
        yb2["burst"] = pd.to_numeric(yb2["burst"], errors="coerce").fillna(0.0)
        burst_top = make_topn_per_class(
            yb2[yb2["burst"] > 0].copy(),
            top_n=top_n,
            sort_col="burst",
            min_freq=min_freq
        )
        out_burst_top = os.path.join(out_dir, f"{year}_trend_burst_merged_top{top_n}_min{min_freq}.csv")
        burst_top.to_csv(out_burst_top, index=False, encoding="utf-8-sig")
        print(f"[OK] {out_burst_top}")

        yf = merged_all_years_df[merged_all_years_df["Year"] == year].copy()
        mix_top = make_mix_topn(yf, yb, top_n=top_n, min_freq=min_freq)
        out_mix = os.path.join(out_dir, f"{year}_trend_mix_merged_top{top_n}_min{min_freq}.csv")
        mix_top.to_csv(out_mix, index=False, encoding="utf-8-sig")
        print(f"[OK] {out_mix}")

    print("\nDONE: merge_map(confidence gate: auto/review apply, hold block) -> freq merge -> share 재계산 -> burst 재계산 -> topN/mix 생성 + split 저장 완료")


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
