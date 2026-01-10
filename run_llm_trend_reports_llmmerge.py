import os
import re
import io
import time
import argparse
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# =========================================================
# Prompt 템플릿
# =========================================================
PROMPT_MERGE_MAP = """너는 키워드 정규화 담당자다. 아래 CSV의 keyword들을 보고,
표기만 다른 경우(대소문자/공백/하이픈/복수형), 한/영 번역쌍, 약어-풀네임, 오탈자, 매우 명확한 동의어만 병합하라.
과병합 금지.

규칙:
- canonical은 가능하면 영어(라틴문자)로 선택하라.
- variant는 CSV에 실제로 존재하는 keyword만 써라.
- confidence 0~1
- reason은 casing|spacing|typo|acronym|ko-en|synonym 중 하나
- 절대 코드블록(``` ... ``` )을 쓰지 말고, 설명 문장도 쓰지 말고, 오직 CSV 본문만 출력하라.

출력은 반드시 CSV로만:
Year,NODE_CLSS_02,canonical,variant,confidence,reason

입력 CSV:
{csv_text}
"""

PROMPT_TREND_TOPICS = """너는 학술 트렌드 분석가다. 아래 CSV는 공학 분야의 특정 연도(year) 및 중분류(NODE_CLSS_02)별 키워드 후보 목록이다.
반드시 CSV에 있는 keyword만 근거로 사용하고, CSV에 없는 키워드/정보를 새로 만들어내지 마라.

중요 규칙:
- 토픽은 '최대 5개'까지만 만든다.
- 후보 키워드가 부족하면(의미적으로 묶을 수 없으면) 토픽 수를 1~4개로 줄여도 된다.
- 억지로 5개를 만들지 마라. 부족하면 부족하다고 명시하라.

작업:
1) 키워드들을 의미적으로 묶어 “트렌드 토픽”을 만든다. (최대 5개)
2) 각 토픽에 대해:
   - 토픽명(한국어 1개 + 영어 1개)
   - 대표 키워드 6~10개(반드시 CSV의 keyword에서만)
   - Evidence(burst): 해당 토픽에 속한 키워드 중 burst 상위 3개와 burst 수치
3) 마지막에 아래 3줄을 반드시 포함하라:
   - Candidate stats: burst_only=<정수>, total_rows=<정수>
   - Merge stats: match_rate=<정수>/<정수>, merged_groups=<정수>, merged_variants=<정수>
   - Note: 후보 부족 여부와 그 이유(1문장)

출력 형식(그대로 지켜):
[중분류] {class_name}
- Topic 1: <토픽명KR> / <TopicNameEN>
  - Keywords: k1, k2, ...
  - Evidence(burst): kA(burst=...), kB(...), kC(...)
  - Summary: ...
...
- One-line: ...
- Candidate stats: burst_only=..., total_rows=...
- Merge stats: match_rate=.../..., merged_groups=..., merged_variants=...
- Note: ...

CSV:
{csv_text}
"""

PROMPT_COMPARE_FREQ_BURST = """너는 공학 논문 트렌드 분석가다. 아래에 두 개의 CSV가 있다.
A는 연도×중분류별 ‘빈도 Top100’, B는 같은 조건의 ‘급증(burst) Top100’이다.
반드시 CSV에 있는 keyword만 사용하라. 외부지식/새 키워드 생성 금지.

작업:
1) “기본 관심사(빈도 상위)”와 “신규 트렌드(burst 상위)”를 비교해 차이를 3줄로 요약한다.
2) 아래를 뽑아라:
   - Stable keywords: A(빈도)에는 있고 B(burst)에는 없는 키워드 중 상위 10개(빈도 기준)
   - Emerging keywords: B에는 있고 A에는 없는 키워드 중 상위 10개(burst 기준)
3) 마지막에 “Change point”로 올해의 변화 포인트를 1문장으로 작성한다.

출력 형식:
[중분류] {class_name}
- Stable(Top10): ...
- Emerging(Top10): ...
- 3-line comparison:
  1) ...
  2) ...
  3) ...
- Change point: ...

CSV A (freq):
{csv_freq}

CSV B (burst):
{csv_burst}
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

    text_parts = []
    for item in getattr(resp, "output", []) or []:
        for c in getattr(item, "content", []) or []:
            if getattr(c, "type", "") in ("output_text", "text"):
                text_parts.append(getattr(c, "text", "") or "")
    return "\n".join([t for t in text_parts if t]).strip()

# =========================================================
# 유틸
# =========================================================
def safe_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in " _-." else "_" for c in s).strip()

def df_to_csv_text(df: pd.DataFrame, limit_rows: int = 160) -> str:
    if len(df) > limit_rows:
        df = df.head(limit_rows)
    return df.to_csv(index=False)

def strip_code_fences(text: str) -> str:
    if not text:
        return ""
    lines = []
    for ln in text.splitlines():
        if ln.strip().startswith("```"):
            continue
        lines.append(ln)
    return "\n".join(lines).strip()

def read_csv_strip_fence(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    cleaned = strip_code_fences(txt)
    if not cleaned:
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(cleaned), dtype=str)

def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

# =========================================================
# mergemap 적용(정규화 매칭 + 안전 필터 + 합산)
# =========================================================
def apply_mergemap_to_df(b: pd.DataFrame, mergemap_path: str, debug: bool = True) -> tuple[pd.DataFrame, dict]:
    if (not mergemap_path) or (not os.path.exists(mergemap_path)):
        return b, {"merge_used": False, "match_num": 0, "match_den": len(b), "merged_groups": 0, "merged_variants": 0}

    mm = read_csv_strip_fence(mergemap_path)
    if mm.empty:
        if debug:
            print(f"[WARN] mergemap empty after strip: {mergemap_path}")
        return b, {"merge_used": False, "match_num": 0, "match_den": len(b), "merged_groups": 0, "merged_variants": 0}

    # 컬럼명 정리
    mm.columns = [c.strip() for c in mm.columns]
    need = {"canonical", "variant"}
    if not need.issubset(set(mm.columns)):
        if debug:
            print(f"[WARN] mergemap columns invalid: {mergemap_path}")
            print(f"       columns={list(mm.columns)}")
        return b, {"merge_used": False, "match_num": 0, "match_den": len(b), "merged_groups": 0, "merged_variants": 0}

    # year/class 타입 정리(필터 안정화)
    if "Year" in mm.columns:
        mm["Year"] = pd.to_numeric(mm["Year"], errors="coerce")
    if "NODE_CLSS_02" in mm.columns:
        mm["NODE_CLSS_02"] = mm["NODE_CLSS_02"].astype(str).str.strip()

    b = b.copy()
    if "Year" in b.columns:
        b["Year"] = pd.to_numeric(b["Year"], errors="coerce")
    if "NODE_CLSS_02" in b.columns:
        b["NODE_CLSS_02"] = b["NODE_CLSS_02"].astype(str).str.strip()

    # year/class 필터(가능할 때만)
    if "Year" in mm.columns and "NODE_CLSS_02" in mm.columns and "Year" in b.columns and "NODE_CLSS_02" in b.columns:
        y = int(b["Year"].dropna().iloc[0])
        c = str(b["NODE_CLSS_02"].dropna().iloc[0]).strip()
        mm = mm[(mm["Year"] == y) & (mm["NODE_CLSS_02"] == c)].copy()

    if mm.empty:
        if debug:
            print(f"[WARN] mergemap filtered to 0 rows: {mergemap_path}")
        return b, {"merge_used": False, "match_num": 0, "match_den": len(b), "merged_groups": 0, "merged_variants": 0}

    # 정규화 키 생성
    mm["variant"] = mm["variant"].astype(str).str.strip()
    mm["canonical"] = mm["canonical"].astype(str).str.strip()
    mm["variant_norm"] = mm["variant"].map(_norm)

    b["keyword"] = b["keyword"].astype(str).str.strip()
    b["keyword_norm"] = b["keyword"].map(_norm)

    # variant_norm -> canonical 매핑
    mm = mm[mm["variant_norm"].notna() & (mm["variant_norm"] != "")]
    mm = mm.drop_duplicates(subset=["variant_norm"], keep="first")
    mp = dict(zip(mm["variant_norm"], mm["canonical"]))

    # 매칭률 계산
    match_num = int(b["keyword_norm"].isin(set(mp.keys())).sum())
    match_den = int(len(b))
    if debug:
        print(f"[DBG] match_rate={match_num}/{match_den} for {os.path.basename(mergemap_path)}")

    # canonical 적용
    b["canonical"] = b["keyword_norm"].map(lambda x: mp.get(x, None))
    b["canonical"] = b["canonical"].fillna(b["keyword"])

    # merge stats
    group_sizes = b.groupby("canonical")["keyword"].nunique()
    merged_groups = int((group_sizes >= 2).sum())
    merged_variants = int(group_sizes.sum() - group_sizes.size)

    # canonical 기준 재집계
    agg_ops = {}
    if "freq" in b.columns:        agg_ops["freq"] = "sum"
    if "paper_count" in b.columns: agg_ops["paper_count"] = "max"
    if "share_prev" in b.columns:  agg_ops["share_prev"] = "sum"

    keys = ["Year", "NODE_CLSS_02", "canonical"] if "Year" in b.columns and "NODE_CLSS_02" in b.columns else ["canonical"]
    out = b.groupby(keys, as_index=False).agg(agg_ops)

    # share/burst 재계산
    if "freq" in out.columns and "paper_count" in out.columns:
        out["share"] = out["freq"] / out["paper_count"].replace(0, pd.NA)
        out["share"] = out["share"].fillna(0.0)

    if "share_prev" in out.columns and "share" in out.columns:
        out["burst"] = out["share"] - out["share_prev"]

    out = out.rename(columns={"canonical": "keyword"})

    return out, {
        "merge_used": True,
        "match_num": match_num,
        "match_den": match_den,
        "merged_groups": merged_groups,
        "merged_variants": merged_variants
    }

# =========================================================
# 메인
# =========================================================
def run(input_dir: str, out_dir: str, years: list[int], mode: str,
        sleep_sec: float, max_output_tokens: int, preview: bool,
        make_mergemap: bool, apply_merge: bool, merge_map_dir: str | None,
        debug_merge: bool):
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY 환경변수가 없습니다. export OPENAI_API_KEY=... 로 설정하세요.")

    model = os.getenv("OPENAI_MODEL", "gpt-5-nano")
    client = OpenAI()

    os.makedirs(out_dir, exist_ok=True)

    if merge_map_dir is None:
        merge_map_dir = os.path.join(out_dir, "_llm_merge_maps")
    os.makedirs(merge_map_dir, exist_ok=True)

    merged_inputs_dir = os.path.join(out_dir, "_merged_inputs")
    os.makedirs(merged_inputs_dir, exist_ok=True)

    for year in years:
        freq_path  = os.path.join(input_dir, f"{year}_freq_top100_min5.csv")
        burst_path = os.path.join(input_dir, f"{year}_trend_mix_top100_min5.csv")

        if not os.path.exists(burst_path):
            print(f"[SKIP] {burst_path} 없음")
            continue
        if mode == "compare" and not os.path.exists(freq_path):
            print(f"[SKIP] {freq_path} 없음 (compare 모드)")
            continue

        burst_df = pd.read_csv(burst_path)
        freq_df = pd.read_csv(freq_path) if mode == "compare" else None

        if "NODE_CLSS_02" not in burst_df.columns:
            print(f"[SKIP] {burst_path} 에 NODE_CLSS_02 컬럼이 없음")
            continue

        classes = sorted(burst_df["NODE_CLSS_02"].dropna().unique().tolist())
        year_md = [f"# {year} 공학 트렌드 리포트 ({mode})\n\n"]

        for cls in classes:
            b = burst_df[burst_df["NODE_CLSS_02"] == cls].copy()
            mm_path = os.path.join(merge_map_dir, f"{year}_{safe_filename(str(cls))}_merge_map.csv")

            # 1) mergemap 생성(없을 때만)
            if make_mergemap and (not os.path.exists(mm_path)):
                cols = [c for c in ["Year","NODE_CLSS_02","keyword","freq","paper_count","share","share_prev","burst"] if c in b.columns]
                csv_for_merge = df_to_csv_text(b[cols] if cols else b, limit_rows=160)
                prompt_merge = PROMPT_MERGE_MAP.format(csv_text=csv_for_merge)

                merge_text = ""
                for attempt in range(3):
                    try:
                        merge_text = call_llm(client, model, prompt_merge, max_output_tokens=1200)
                        break
                    except Exception as e:
                        print(f"[ERR] mergemap year={year}, class={cls}, attempt={attempt+1}: {e}")
                        time.sleep(2 * (attempt + 1))

                # 코드펜스 제거 후 저장(안전)
                merge_text = strip_code_fences(merge_text)
                if merge_text.strip():
                    with open(mm_path, "w", encoding="utf-8") as fw:
                        fw.write(merge_text.strip() + "\n")
                    print(f"[OK] mergemap saved: {mm_path}")
                else:
                    print(f"[WARN] mergemap empty: year={year}, class={cls}")

                time.sleep(sleep_sec)

            # 2) mergemap 적용(합산)
            mstats = {"match_num": 0, "match_den": len(b), "merged_groups": 0, "merged_variants": 0}
            if apply_merge and os.path.exists(mm_path):
                b, mstats = apply_mergemap_to_df(b, mm_path, debug=debug_merge)
                merged_out_path = os.path.join(merged_inputs_dir, f"{year}_{safe_filename(str(cls))}_burst_merged.csv")
                b.to_csv(merged_out_path, index=False, encoding="utf-8-sig")
                print(f"[OK] merged burst saved: {merged_out_path} (groups={mstats['merged_groups']}, variants={mstats['merged_variants']})")

            # 3) 리포트 생성
            if mode == "trend":
                cols = [c for c in ["Year","NODE_CLSS_02","keyword","freq","paper_count","share","share_prev","burst"] if c in b.columns]
                csv_text = df_to_csv_text(b[cols] if cols else b, limit_rows=160)

                burst_only = 0
                if "burst" in b.columns:
                    burst_only = int((pd.to_numeric(b["burst"], errors="coerce").fillna(0.0) > 0).sum())
                total_rows = int(len(b))

                prompt = PROMPT_TREND_TOPICS.format(class_name=cls, csv_text=csv_text + (
                    f"\n# (stats) burst_only={burst_only}, total_rows={total_rows}, "
                    f"match_rate={mstats.get('match_num',0)}/{mstats.get('match_den',0)}, "
                    f"merged_groups={mstats.get('merged_groups',0)}, merged_variants={mstats.get('merged_variants',0)}\n"
                ))
            else:
                f = freq_df[freq_df["NODE_CLSS_02"] == cls].copy()
                cols_f = [c for c in ["Year","NODE_CLSS_02","keyword","freq","paper_count","share"] if c in f.columns]
                cols_b = [c for c in ["Year","NODE_CLSS_02","keyword","freq","paper_count","share","share_prev","burst"] if c in b.columns]
                csv_freq = df_to_csv_text(f[cols_f] if cols_f else f, limit_rows=160)
                csv_burst = df_to_csv_text(b[cols_b] if cols_b else b, limit_rows=160)
                prompt = PROMPT_COMPARE_FREQ_BURST.format(class_name=cls, csv_freq=csv_freq, csv_burst=csv_burst)

            text = ""
            for attempt in range(3):
                try:
                    text = call_llm(client, model, prompt, max_output_tokens=max_output_tokens)
                    break
                except Exception as e:
                    print(f"[ERR] report year={year}, class={cls}, attempt={attempt+1}: {e}")
                    time.sleep(2 * (attempt + 1))

            if preview:
                print(f"\n----- PREVIEW year={year} class={cls} -----")
                print(text[:600] if text else "(EMPTY)")
                print("----- END PREVIEW -----\n")

            if not text.strip():
                print(f"[WARN] Empty output: year={year}, class={cls}. (skip write)")
                continue

            cls_file = safe_filename(str(cls))
            out_path = os.path.join(out_dir, f"{year}_{cls_file}_{mode}.md")
            with open(out_path, "w", encoding="utf-8") as fw:
                fw.write(text.strip() + "\n")
            print(f"[OK] {out_path}")

            year_md.append(text.strip() + "\n\n")
            time.sleep(sleep_sec)

        year_all_path = os.path.join(out_dir, f"{year}_ALL_{mode}.md")
        combined = "".join(year_md).strip()
        if combined:
            with open(year_all_path, "w", encoding="utf-8") as fw:
                fw.write(combined + "\n")
            print(f"[OK] {year_all_path} (연도 통합)")
        else:
            print(f"[WARN] {year} year_all empty. (no write)")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", default=".", help="CSV가 있는 폴더")
    p.add_argument("--out_dir", default="./llm_reports", help="결과 md 저장 폴더")
    p.add_argument("--years", default="2021,2022,2023,2024,2025", help="쉼표로 구분")
    p.add_argument("--mode", choices=["trend", "compare"], default="trend",
                   help="trend=토픽 생성 / compare=빈도 vs burst 비교")
    p.add_argument("--sleep", type=float, default=0.3, help="요청 사이 쉬는 시간(초)")
    p.add_argument("--max_tokens", type=int, default=1200, help="최대 출력 토큰")
    p.add_argument("--preview", action="store_true", help="응답 미리보기 출력")

    # mergemap 생성/적용
    p.add_argument("--make_mergemap", action="store_true", help="merge_map을 LLM으로 생성(없을 때만)")
    p.add_argument("--apply_merge", action="store_true", help="merge_map 기반으로 keyword를 합산 후 LLM 입력")
    p.add_argument("--merge_map_dir", default=None, help="merge_map CSV들이 있는 폴더(기본: out_dir/_llm_merge_maps)")
    p.add_argument("--debug_merge", action="store_true", help="merge 매칭률 로그 출력")

    args = p.parse_args()
    years = [int(x.strip()) for x in args.years.split(",") if x.strip()]

    run(
        input_dir=args.input_dir,
        out_dir=args.out_dir,
        years=years,
        mode=args.mode,
        sleep_sec=args.sleep,
        max_output_tokens=args.max_tokens,
        preview=args.preview,
        make_mergemap=args.make_mergemap,
        apply_merge=args.apply_merge,
        merge_map_dir=args.merge_map_dir,
        debug_merge=args.debug_merge
    )
