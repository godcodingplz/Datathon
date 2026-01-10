import json
import pandas as pd
from collections import defaultdict
import os
import re

# =========================================================
# [설정]
# =========================================================
input_file_path = '/Users/idonghyeon/Downloads/top_30_percent_papers.json'
output_dir = '/Users/idonghyeon/Desktop/dathon_result'

YEAR_MIN, YEAR_MAX = 2021, 2025
TOP_N = 100          # 빈도 Top100 / 트렌드(burst) Top100
MIN_FREQ = 5         # 노이즈 컷(현재 연도에서 최소 등장 논문 수)
ONLY_ENGINEERING = True  # NODE_CLSS_01이 있으면 공학만

# =========================================================
# 키워드 전처리
# - 괄호 안 약어(CNN 등)는 살리고
# - 괄호 내용은 제거
# - , ; / 로 split
# - 소문자, 공백/기호 정리
# =========================================================
_acronym_pat = re.compile(r'\(([A-Za-z0-9\-]{2,10})\)')  # (CNN), (LSTM) 등

def normalize_token(t: str) -> str:
    t = (t or "").strip().lower()
    if not t:
        return ""
    # 기호 통일(필요하면 더 추가)
    t = t.replace("·", " ").replace("_", " ").replace("-", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t

def extract_keywords(kw_str: str) -> list[str]:
    text = str(kw_str or "")

    # 1) 괄호 안 약어 추출
    acronyms = _acronym_pat.findall(text)
    acronyms = [normalize_token(a) for a in acronyms if a]

    # 2) 괄호 내용 제거
    text = re.sub(r"\([^)]*\)", "", text)

    # 3) split
    tokens = re.split(r"[,;/]", text)
    tokens = [normalize_token(t) for t in tokens if normalize_token(t)]

    # 4) 약어도 추가
    tokens.extend([a for a in acronyms if a])

    # 5) 너무 짧은 토큰 제거(원하면 기준 조절)
    tokens = [t for t in tokens if len(t) > 1]

    return tokens

# =========================================================
# 1. 데이터 로드
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

# 필수 컬럼 확인
required_columns = ['PBSH', 'NODE_CLSS_02', 'KYWD']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"오류: 필수 컬럼 '{col}' 이(가) 누락되었습니다.")

# 공학만 필터(컬럼이 존재할 때만)
if ONLY_ENGINEERING and 'NODE_CLSS_01' in df.columns:
    df = df[df['NODE_CLSS_01'].fillna('') == '공학'].copy()

# =========================================================
# 2. 전처리
# =========================================================
df['Year'] = df['PBSH'].fillna('').astype(str).str[:4]
df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')
df['NODE_CLSS_02'] = df['NODE_CLSS_02'].fillna('미분류')
df['KYWD'] = df['KYWD'].fillna('')

df = df[(df['Year'].notna()) & (df['Year'] >= YEAR_MIN) & (df['Year'] <= YEAR_MAX)].copy()
df['Year'] = df['Year'].astype(int)

os.makedirs(output_dir, exist_ok=True)

# =========================================================
# 3. 집계 테이블 만들기
# - freq는 "키워드가 등장한 논문 수"로 집계(논문당 중복 방지)
# =========================================================
paper_count = defaultdict(int)               # (year, class) -> 논문 수
kw_freq = defaultdict(int)                   # (year, class, keyword) -> 등장 논문 수

for _, row in df.iterrows():
    year = row['Year']
    clss = row['NODE_CLSS_02']

    paper_count[(year, clss)] += 1

    # 한 논문 안에서 동일 키워드 중복 카운트 방지
    kws = set(extract_keywords(row['KYWD']))
    for k in kws:
        kw_freq[(year, clss, k)] += 1

# agg dataframe
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

# 전체 집계 저장(나중에 다시 뽑을 수 있게)
agg_all_path = os.path.join(output_dir, f"keywords_agg_all_{YEAR_MIN}_{YEAR_MAX}.csv")
agg.to_csv(agg_all_path, index=False, encoding="utf-8-sig")
print(f"[OK] 전체 집계 저장: {agg_all_path}")

# =========================================================
# 4. 빈도 Top100 저장 (연도별 파일)
# =========================================================
for year in sorted(agg['Year'].unique()):
    year_df = agg[agg['Year'] == year].copy()

    # (연도, 중분류)별로 freq 내림차순 Top100
    freq_top = (year_df[year_df["freq"] >= MIN_FREQ]
                .sort_values(["NODE_CLSS_02", "freq"], ascending=[True, False])
                .groupby("NODE_CLSS_02", as_index=False, group_keys=False)
                .head(TOP_N)
               )

    # rank 부여(중분류별)
    freq_top["rank_freq"] = (freq_top.groupby("NODE_CLSS_02")["freq"]
                             .rank(method="first", ascending=False).astype(int))

    out_path = os.path.join(output_dir, f"{year}_freq_top{TOP_N}_min{MIN_FREQ}.csv")
    freq_top.sort_values(["NODE_CLSS_02", "rank_freq"]).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[OK] {year} 빈도 Top{TOP_N} 저장: {out_path}")

# =========================================================
# 5. 트렌드(burst) Top100 저장 (연도별 파일)
# - burst = share(Y) - share(Y-1)
# - 전년도 값이 없으면 0으로 처리
# - burst 후보가 부족하면 freq로 채워서 TOP_N 맞춤
# =========================================================

# (1) 전년도 share 붙이기 위한 prev 만들기
prev = agg[["Year", "NODE_CLSS_02", "keyword", "share"]].copy()
prev["Year"] = prev["Year"] + 1
prev = prev.rename(columns={"share": "share_prev"})

# (2) burst_df 만들기 (여기 필수!)
burst_df = agg.merge(prev, on=["Year", "NODE_CLSS_02", "keyword"], how="left")
burst_df["share_prev"] = burst_df["share_prev"].fillna(0.0)
burst_df["burst"] = burst_df["share"] - burst_df["share_prev"]

FREQ_LEVELS = [5, 3, 2, 1]  # 필요하면 조절

for year in sorted(burst_df["Year"].unique()):
    year_df = burst_df[burst_df["Year"] == year].copy()

    out_rows = []
    for cls, g in year_df.groupby("NODE_CLSS_02"):
        g = g.copy()

        # 1) burst>0 우선 (자동 완화로 하나라도 잡기)
        trend = None
        used_mf = None
        for mf in FREQ_LEVELS:
            cand = g[(g["freq"] >= mf) & (g["burst"] > 0)].sort_values("burst", ascending=False)
            if len(cand) > 0:
                trend = cand.head(TOP_N).copy()
                used_mf = mf
                break

        if trend is None:
            trend = g[g["burst"] > 0].sort_values("burst", ascending=False).head(TOP_N).copy()
            used_mf = 0

        trend["source"] = "burst"
        trend["min_freq_used"] = used_mf

        picked = set(trend["keyword"].tolist())

        # 2) 부족하면 freq로 채우기
        if len(trend) < TOP_N:
            need = TOP_N - len(trend)
            mf_fill = used_mf if used_mf and used_mf > 0 else MIN_FREQ

            filler = g[(g["freq"] >= mf_fill) & (~g["keyword"].isin(picked))] \
                       .sort_values("freq", ascending=False) \
                       .head(need) \
                       .copy()

            filler["source"] = "freq_fill"
            filler["min_freq_used"] = mf_fill

            trend = pd.concat([trend, filler], ignore_index=True)

        out_rows.append(trend)

    trend_mix = pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame()
    if trend_mix.empty:
        print(f"[WARN] {year} trend_mix empty")
        continue

    # rank (중분류 내에서 burst 우선이므로 burst 기준으로)
    trend_mix["rank"] = (trend_mix.groupby("NODE_CLSS_02")["burst"]
                         .rank(method="first", ascending=False).astype(int))

    out_path = os.path.join(output_dir, f"{year}_trend_mix_top{TOP_N}_min{MIN_FREQ}.csv")
    trend_mix.sort_values(["NODE_CLSS_02", "rank"]).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[OK] {year} 트렌드 mix Top{TOP_N} 저장: {out_path}")
