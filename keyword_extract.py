import json
import pandas as pd
from collections import defaultdict
import os
import re
import io
import time

# =========================
# OpenAI LLM for aggressive merge
# =========================
from openai import OpenAI

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

# =========================================================
# [설정]
# =========================================================
input_file_path = '/Users/idonghyeon/Desktop/datathon/SSU_Datathon2025_공학분야_62199.json'
output_dir = '/Users/idonghyeon/Desktop/dathon_result'

YEAR_MIN, YEAR_MAX = 2021, 2025
TOP_N = 100
MIN_FREQ = 5
ONLY_ENGINEERING = True

# ✅ 최소 확보 목표(품질 유지 + 부족하면 그냥 부족하게 저장)
MIN_KEEP = 50

# ✅ fallback min_freq (절대 2 밑으로 안 감)
FREQ_LEVELS = [5, 4, 3, 2]

# ✅ (핵심) 키워드 부족 감지 기준: (연도×중분류)에서 살아남는 키워드 수
AGGRESSIVE_THRESHOLD = 20  # (year,class)에서 freq>=5 키워드 수가 20 미만이면 aggressive 대상

# ✅ (핵심) LLM에 넣을 키워드 수
AGGRESSIVE_MAX_KW = 800

# ✅ canonical 목표 범위 (너가 원하던 "2개만 남는거" 방지)
TARGET_CANONICAL_MIN_DEFAULT = 25
TARGET_CANONICAL_MAX_DEFAULT = 80
TARGET_CANONICAL_MIN_CHEM = 40
TARGET_CANONICAL_MAX_CHEM = 120

# ✅ LLM 모델
AGGRESSIVE_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ✅ LLM 요청 간 sleep (rate limit)
AGGRESSIVE_SLEEP = 0.2


# =========================================================
# 키워드 전처리
# =========================================================
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

    return tokens


# =========================================================
# Aggressive merge_map (LLM)
# =========================================================
PROMPT_AGGRESSIVE_MERGE_TSV = """너는 학술 키워드 "대주제 압축" 담당자다.
아래 목록은 특정 공학 중분류의 키워드와 빈도(freq)다.

목표:
- 희소 키워드를 "중간~넓은 대주제"로 합쳐서 분석 가능한 키워드 풀이 되게 만든다.
- 단, canonical(대주제) 수가 너무 적어지면(예: 1~2개) 분석이 불가능해지므로 이를 금지한다.
- 가능한 한 많은 canonical이 sum_freq >= {min_freq}가 되도록 압축하라.

압축 규칙(중요):
1) 표기/번역/약어/오탈자/유사표현은 적극 병합한다.
2) 세부 기술/세부 재료/세부 공정은 상위 대주제(예: polymer, catalysis, process optimization 등)로 합칠 수 있다.
3) 하지만 "모든 것을 1~2개로 뭉치기"는 금지한다.
   - canonical(고유 대주제) 수는 대략 {target_min}~{target_max} 사이가 되게 유지하라.
   - 가능하면 {target_min}개 이상을 만들되, 너무 잘게 쪼개지도 말라.
4) canonical은 짧고 일반적인 영어 표현을 우선하라.
   (필요하면 입력에 없는 새 canonical 생성 가능)
5) variant는 반드시 입력 목록에 있는 키워드만 사용하라.
6) 애매하면 "miscellaneous" 같은 완충 canonical을 만들어 흡수해도 된다.

출력 형식:
- 설명/코드블록 없이 TSV 본문만 출력
- 헤더는 반드시 아래 그대로:
variant\tcanonical\treason\tconfidence
- reason 값은 아래 중 하나:
broad|synonym|acronym|typo|same
- confidence는 0~1 사이 문자열

입력 목록(키워드\\tfreq):
{kw_list}
"""

ALLOWED_REASON = {"broad", "synonym", "acronym", "typo", "same"}


def call_llm_text(client: OpenAI, model: str, prompt: str, max_output_tokens: int = 1800) -> str:
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


def _is_float(x: str) -> bool:
    try:
        float(str(x).strip())
        return True
    except Exception:
        return False


def read_tsv_robust(text: str) -> pd.DataFrame:
    """
    LLM TSV 출력이 5컬럼 이상으로 깨져도 최대한 복구
    """
    cleaned = strip_code_fences(text or "")
    cleaned = cleaned.replace("\ufeff", "").strip()
    if not cleaned:
        return pd.DataFrame()

    lines = [ln for ln in cleaned.splitlines() if str(ln).strip()]
    if not lines:
        return pd.DataFrame()

    # 헤더 유무
    if lines[0].lower().startswith("variant"):
        lines = lines[1:]

    recs = []
    for ln in lines:
        ln = str(ln).rstrip("\t").strip()
        parts = ln.split("\t")
        while parts and parts[-1] == "":
            parts = parts[:-1]
        if len(parts) < 2:
            continue

        variant = parts[0].strip()
        canonical = parts[1].strip()
        reason = "same"
        conf = "0.7"

        if len(parts) >= 3 and parts[2].strip() in ALLOWED_REASON:
            reason = parts[2].strip()

        # confidence는 뒤에서 숫자 찾기
        for p in reversed(parts[2:]):
            if _is_float(p):
                conf = str(float(p))
                break

        # clamp
        try:
            cf = float(conf)
            if cf < 0:
                conf = "0.0"
            elif cf > 1:
                conf = "1.0"
            else:
                conf = str(cf)
        except Exception:
            conf = "0.7"

        recs.append({"variant": variant, "canonical": canonical, "reason": reason, "confidence": conf})

    df = pd.DataFrame(recs)
    return df


def safe_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in " _-." else "_" for c in str(s)).strip()


def build_aggressive_merge_map(
    client: OpenAI,
    model: str,
    cls: str,
    kw_freq_df: pd.DataFrame,
    cache_dir: str,
    min_freq: int,
    target_min: int,
    target_max: int,
    max_kw: int = 800,
    sleep_sec: float = 0.2,
) -> pd.DataFrame:
    """
    kw_freq_df columns: keyword, freq (aggregated across years)
    returns DataFrame with columns: variant, canonical, reason, confidence
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"aggressive_{safe_filename(cls)}.tsv")

    # cache reuse
    if os.path.exists(cache_path):
        mm = pd.read_csv(cache_path, sep="\t", dtype=str, engine="python")
        if {"variant", "canonical"}.issubset(set(mm.columns)):
            return mm

    # Top-K for LLM
    top = kw_freq_df.sort_values("freq", ascending=False).head(max_kw).copy()

    lines = []
    for _, r in top.iterrows():
        lines.append(f"{r['keyword']}\t{int(r['freq'])}")
    kw_list = "\n".join(lines)

    prompt = PROMPT_AGGRESSIVE_MERGE_TSV.format(
        kw_list=kw_list,
        min_freq=min_freq,
        target_min=target_min,
        target_max=target_max
    )

    raw = call_llm_text(client, model, prompt, max_output_tokens=1800)
    mm = read_tsv_robust(raw)

    # 실패하면 identity
    if mm.empty or not {"variant", "canonical"}.issubset(set(mm.columns)):
        mm = pd.DataFrame({
            "variant": top["keyword"].astype(str).tolist(),
            "canonical": top["keyword"].astype(str).tolist(),
            "reason": ["same"] * len(top),
            "confidence": ["1.0"] * len(top),
        })

    mm["variant"] = mm["variant"].astype(str).str.strip()
    mm["canonical"] = mm["canonical"].astype(str).str.strip()
    mm = mm.drop_duplicates(subset=["variant"], keep="first").copy()

    # save
    mm.to_csv(cache_path, index=False, sep="\t", encoding="utf-8-sig")
    time.sleep(sleep_sec)
    return mm


def apply_aggressive_map_to_agg(agg: pd.DataFrame, cls: str, mm: pd.DataFrame) -> pd.DataFrame:
    """
    agg: columns [Year, NODE_CLSS_02, keyword, freq, paper_count, share]
    mm: variant->canonical
    """
    sub = agg[agg["NODE_CLSS_02"] == cls].copy()
    if sub.empty:
        return agg

    mp = dict(zip(mm["variant"].astype(str), mm["canonical"].astype(str)))

    sub["canonical"] = sub["keyword"].astype(str).map(lambda x: mp.get(x, x))

    merged_variants = (
        sub.groupby(["Year", "NODE_CLSS_02", "canonical"])["keyword"]
           .apply(lambda s: ";".join(sorted(set([str(x) for x in s if str(x).strip()]))))
           .reset_index()
           .rename(columns={"keyword": "merged_variants"})
    )

    sub["freq"] = pd.to_numeric(sub["freq"], errors="coerce").fillna(0).astype(int)
    sub["paper_count"] = pd.to_numeric(sub["paper_count"], errors="coerce").fillna(0).astype(int)

    merged = (
        sub.groupby(["Year", "NODE_CLSS_02", "canonical"], as_index=False)
           .agg(freq=("freq", "sum"), paper_count=("paper_count", "max"))
           .rename(columns={"canonical": "keyword"})
    )

    merged = merged.merge(
        merged_variants.rename(columns={"canonical": "keyword"}),
        on=["Year", "NODE_CLSS_02", "keyword"],
        how="left"
    )

    merged["share"] = merged["freq"] / merged["paper_count"].replace(0, pd.NA)
    merged["share"] = merged["share"].fillna(0.0)

    out = pd.concat([agg[agg["NODE_CLSS_02"] != cls], merged], ignore_index=True)
    return out


# =========================================================
# 1) 데이터 로드
# =========================================================
if not os.path.exists(input_file_path):
    raise FileNotFoundError(f"오류: '{input_file_path}' 파일을 찾을 수 없습니다.")

with open(input_file_path, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

if isinstance(raw_data, dict) and "NODE_LIST" in raw_data:
    data_list = raw_data["NODE_LIST"]
elif isinstance(raw_data, list):
    data_list = raw_data
else:
    data_list = [raw_data]

df = pd.DataFrame(data_list)

required_columns = ['PBSH', 'NODE_CLSS_02', 'KYWD']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"오류: 필수 컬럼 '{col}' 이(가) 누락되었습니다.")

# 공학만 필터(컬럼이 있을 때만)
if ONLY_ENGINEERING and 'NODE_CLSS_01' in df.columns:
    df = df[df['NODE_CLSS_01'].fillna('') == '공학'].copy()

# =========================================================
# 2) 전처리
# =========================================================
df['Year'] = df['PBSH'].fillna('').astype(str).str[:4]
df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')
df['NODE_CLSS_02'] = df['NODE_CLSS_02'].fillna('미분류')
df['KYWD'] = df['KYWD'].fillna('')

df = df[(df['Year'].notna()) & (df['Year'] >= YEAR_MIN) & (df['Year'] <= YEAR_MAX)].copy()
df['Year'] = df['Year'].astype(int)

os.makedirs(output_dir, exist_ok=True)

# =========================================================
# 3) 집계 테이블 만들기 (원본 agg)
# - freq = "키워드가 등장한 논문 수" (논문 내 중복 방지)
# =========================================================
paper_count = defaultdict(int)  # (year, class) -> 논문 수
kw_freq = defaultdict(int)      # (year, class, keyword) -> 등장 논문 수

for _, row in df.iterrows():
    year = row['Year']
    clss = row['NODE_CLSS_02']
    paper_count[(year, clss)] += 1

    kws = set(extract_keywords(row['KYWD']))
    for k in kws:
        kw_freq[(year, clss, k)] += 1

rows = []
for (year, clss, k), freq in kw_freq.items():
    pc = paper_count[(year, clss)]
    rows.append({
        "Year": year,
        "NODE_CLSS_02": clss,
        "keyword": k,
        "freq": int(freq),
        "paper_count": int(pc),
        "share": float(freq) / float(pc) if pc else 0.0
    })

agg = pd.DataFrame(rows)

# 전체 집계 저장
agg_all_path = os.path.join(output_dir, f"keywords_agg_all_{YEAR_MIN}_{YEAR_MAX}.csv")
agg.to_csv(agg_all_path, index=False, encoding="utf-8-sig")
print(f"[OK] 전체 집계 저장: {agg_all_path}")

# =========================================================
# 3.5) 키워드 부족 중분류 감지 -> aggressive 선압축
# =========================================================
need_classes = set()
for (year, cls), g in agg.groupby(["Year", "NODE_CLSS_02"]):
    alive = g[g["freq"] >= MIN_FREQ]["keyword"].nunique()
    if alive < AGGRESSIVE_THRESHOLD:
        need_classes.add(cls)

print(f"[INFO] aggressive 압축 대상 중분류 수 = {len(need_classes)}")
if need_classes:
    if load_dotenv:
        load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY 환경변수가 없습니다. aggressive 압축을 위해 필요합니다.")

    client = OpenAI()
    cache_dir = os.path.join(output_dir, "_aggressive_merge_maps")
    os.makedirs(cache_dir, exist_ok=True)

    for cls in sorted(list(need_classes)):
        cls_df = agg[agg["NODE_CLSS_02"] == cls].copy()
        if cls_df.empty:
            continue

        cls_sum = (
            cls_df.groupby("keyword", as_index=False)
                  .agg(freq=("freq", "sum"))
                  .sort_values("freq", ascending=False)
        )

        if "화학" in str(cls):
            tmin, tmax = TARGET_CANONICAL_MIN_CHEM, TARGET_CANONICAL_MAX_CHEM
        else:
            tmin, tmax = TARGET_CANONICAL_MIN_DEFAULT, TARGET_CANONICAL_MAX_DEFAULT

        before_alive = []
        for (y, _), gg in cls_df.groupby(["Year", "NODE_CLSS_02"]):
            before_alive.append((y, gg[gg["freq"] >= MIN_FREQ]["keyword"].nunique()))
        before_min = min([x[1] for x in before_alive]) if before_alive else -1

        mm = build_aggressive_merge_map(
            client=client,
            model=AGGRESSIVE_MODEL,
            cls=cls,
            kw_freq_df=cls_sum,
            cache_dir=cache_dir,
            min_freq=MIN_FREQ,
            target_min=tmin,
            target_max=tmax,
            max_kw=AGGRESSIVE_MAX_KW,
            sleep_sec=AGGRESSIVE_SLEEP,
        )

        agg = apply_aggressive_map_to_agg(agg, cls, mm)

        after_df = agg[agg["NODE_CLSS_02"] == cls].copy()
        after_alive = []
        for (y, _), gg in after_df.groupby(["Year", "NODE_CLSS_02"]):
            after_alive.append((y, gg[gg["freq"] >= MIN_FREQ]["keyword"].nunique()))
        after_min = min([x[1] for x in after_alive]) if after_alive else -1

        print(f"[OK] aggressive merge applied: {cls} | alive(min) {before_min} -> {after_min}")

    agg_merged_path = os.path.join(output_dir, f"keywords_agg_all_{YEAR_MIN}_{YEAR_MAX}_aggressive_merged.csv")
    agg.to_csv(agg_merged_path, index=False, encoding="utf-8-sig")
    print(f"[OK] aggressive merged agg 저장: {agg_merged_path}")

# =========================================================
# 4) 빈도 TopN 저장 (연도별)  ✅ 5→4→3→2 fallback + MIN_KEEP=50 목표
# =========================================================
for year in sorted(agg['Year'].unique()):
    year_df = agg[agg['Year'] == year].copy()

    out_rows = []
    for cls, g in year_df.groupby("NODE_CLSS_02"):
        g = g.copy()

        picked = None
        used_mf = None

        # ✅ MIN_KEEP 충족 가능한 mf를 우선 선택
        for mf in FREQ_LEVELS:
            cand = g[g["freq"] >= mf].sort_values("freq", ascending=False)
            if len(cand) >= MIN_KEEP:
                picked = cand.head(TOP_N).copy()
                used_mf = mf
                break

        # ✅ 그래도 MIN_KEEP가 안되면: mf=2 기준으로 최대한 확보(2 밑으로 내려가지 않음)
        if picked is None:
            cand = g[g["freq"] >= 2].sort_values("freq", ascending=False)
            picked = cand.head(TOP_N).copy()
            used_mf = 2
            if len(picked) < MIN_KEEP:
                print(f"[WARN] {year}/{cls} freq_top: freq>=2로도 {MIN_KEEP}개 미만({len(picked)}개)")

        picked["min_freq_used"] = used_mf
        out_rows.append(picked)

    freq_top = pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame()
    if freq_top.empty:
        print(f"[WARN] {year} freq_top empty")
        continue

    freq_top["rank_freq"] = (freq_top.groupby("NODE_CLSS_02")["freq"]
                             .rank(method="first", ascending=False).astype(int))

    out_path = os.path.join(output_dir, f"{year}_freq_top{TOP_N}_min{MIN_FREQ}.csv")
    freq_top.sort_values(["NODE_CLSS_02", "rank_freq"]).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[OK] {year} 빈도 Top{TOP_N} 저장: {out_path}")

# =========================================================
# 5) 트렌드(burst) mix TopN 저장 (연도별)
# - burst = share(Y) - share(Y-1)
# - ✅ 5→4→3→2 fallback, 절대 2 미만으로 안감
# - ✅ burst>0 부족하면 freq_fill도 2까지로만 채움 (100 못 채우면 그냥 부족하게 저장)
# =========================================================
prev = agg[["Year", "NODE_CLSS_02", "keyword", "share"]].copy()
prev["Year"] = prev["Year"] + 1
prev = prev.rename(columns={"share": "share_prev"})

burst_df = agg.merge(prev, on=["Year", "NODE_CLSS_02", "keyword"], how="left")
burst_df["share_prev"] = burst_df["share_prev"].fillna(0.0)
burst_df["burst"] = burst_df["share"] - burst_df["share_prev"]

for year in sorted(burst_df["Year"].unique()):
    year_df = burst_df[burst_df["Year"] == year].copy()

    out_rows = []
    for cls, g in year_df.groupby("NODE_CLSS_02"):
        g = g.copy()

        # ✅ 1) burst>0 우선 (MIN_KEEP 만족하려고 mf 완화)
        trend = None
        used_mf = None

        for mf in FREQ_LEVELS:
            cand = g[(g["freq"] >= mf) & (g["burst"] > 0)].sort_values("burst", ascending=False)
            if len(cand) >= MIN_KEEP:
                trend = cand.head(TOP_N).copy()
                used_mf = mf
                break

        # ✅ 그래도 부족하면: freq>=2 & burst>0만 최대한
        if trend is None:
            cand = g[(g["freq"] >= 2) & (g["burst"] > 0)].sort_values("burst", ascending=False)
            trend = cand.head(TOP_N).copy()
            used_mf = 2
            if len(trend) == 0:
                # burst>0 자체가 없음 -> freq 기반으로만 시작
                trend = g[g["freq"] >= 2].sort_values("freq", ascending=False).head(min(TOP_N, MIN_KEEP)).copy()
                used_mf = 2
                trend["source"] = "no_positive_burst_freq_seed"
            elif len(trend) < MIN_KEEP:
                print(f"[WARN] {year}/{cls} trend_mix: burst>0 & freq>=2로도 {MIN_KEEP}개 미만({len(trend)}개)")

        if "source" not in trend.columns:
            trend["source"] = "burst"
        trend["min_freq_used"] = used_mf

        picked = set(trend["keyword"].tolist())

        # ✅ 2) 부족하면 freq로 채우기(절대 2 미만 안감)
        if len(trend) < TOP_N:
            need = TOP_N - len(trend)

            filler = g[(g["freq"] >= 2) & (~g["keyword"].isin(picked))] \
                .sort_values("freq", ascending=False) \
                .head(need) \
                .copy()

            filler["source"] = "freq_fill"
            filler["min_freq_used"] = 2

            trend = pd.concat([trend, filler], ignore_index=True)

            # 그래도 TOP_N 못 채우면 그대로(품질 유지)
            if len(trend) < TOP_N:
                print(f"[WARN] {year}/{cls} trend_mix: freq>=2로 TOP_N={TOP_N} 채우기 실패({len(trend)}개)")

        out_rows.append(trend)

    trend_mix = pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame()
    if trend_mix.empty:
        print(f"[WARN] {year} trend_mix empty")
        continue

    # rank는 burst 기준(내림차순)
    trend_mix["rank"] = (trend_mix.groupby("NODE_CLSS_02")["burst"]
                         .rank(method="first", ascending=False).astype(int))

    out_path = os.path.join(output_dir, f"{year}_trend_mix_top{TOP_N}_min{MIN_FREQ}.csv")
    trend_mix.sort_values(["NODE_CLSS_02", "rank"]).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[OK] {year} 트렌드 mix Top{TOP_N} 저장: {out_path}")

print("\nDONE ✅ aggressive 선압축(대주제 canonical 범위 유지) -> freq fallback(5→4→3→2) -> trend mix fallback(5→4→3→2) 완료")
