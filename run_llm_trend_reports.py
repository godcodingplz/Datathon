import os
import time
import argparse
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# =========================================================
# Prompt 템플릿
# =========================================================
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
3) 마지막에 아래 2줄을 반드시 포함하라:
   - Candidate stats: burst_only=<정수>, total_rows=<정수>
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
# OpenAI 호출 + 응답 텍스트 추출(빈 파일 문제 방지 핵심)
# =========================================================
def call_llm(client: OpenAI, model: str, prompt: str, max_output_tokens: int = 1200) -> str:
    resp = client.responses.create(
        model=model,
        input=prompt,
        max_output_tokens=max_output_tokens,
    )

    # 1) 우선 output_text 시도
    if getattr(resp, "output_text", None):
        return (resp.output_text or "").strip()

    # 2) fallback: output 배열에서 text 합치기
    text_parts = []
    for item in getattr(resp, "output", []) or []:
        for c in getattr(item, "content", []) or []:
            if getattr(c, "type", "") in ("output_text", "text"):
                text_parts.append(getattr(c, "text", "") or "")
    return "\n".join([t for t in text_parts if t]).strip()

# =========================================================
# 유틸
# =========================================================
def df_to_csv_text(df: pd.DataFrame, limit_rows: int = 120) -> str:
    # 너무 길어질 때 안전장치 (Top100이지만 혹시 컬럼 많으면)
    if len(df) > limit_rows:
        df = df.head(limit_rows)
    return df.to_csv(index=False)

def safe_filename(s: str) -> str:
    # 한글 파일명도 OS에 따라 문제될 수 있어 최소한의 치환
    return "".join(c if c.isalnum() or c in " _-." else "_" for c in s).strip()

# =========================================================
# 메인 로직
# =========================================================
def run(input_dir: str, out_dir: str, years: list[int], mode: str,
        sleep_sec: float, max_output_tokens: int, preview: bool):
    load_dotenv()  # .env 있으면 읽음(없어도 OK)

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY 환경변수가 없습니다. export/setx로 설정하세요.")

    model = os.getenv("OPENAI_MODEL", "gpt-5-nano")  # 싼 모델 기본값(원하면 env로 바꾸기)
    client = OpenAI()

    os.makedirs(out_dir, exist_ok=True)

    for year in years:
        freq_path = os.path.join(input_dir, f"{year}_freq_top100_min5.csv")
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

            # 프롬프트에 넣을 CSV 텍스트
            if mode == "trend":
                cols = [c for c in ["Year","NODE_CLSS_02","keyword","freq","paper_count","share","share_prev","burst"] if c in b.columns]
                csv_text = df_to_csv_text(b[cols] if cols else b)
                prompt = PROMPT_TREND_TOPICS.format(class_name=cls, csv_text=csv_text)
            else:
                f = freq_df[freq_df["NODE_CLSS_02"] == cls].copy()
                cols_f = [c for c in ["Year","NODE_CLSS_02","keyword","freq","paper_count","share"] if c in f.columns]
                cols_b = [c for c in ["Year","NODE_CLSS_02","keyword","freq","paper_count","share","share_prev","burst"] if c in b.columns]
                csv_freq = df_to_csv_text(f[cols_f] if cols_f else f)
                csv_burst = df_to_csv_text(b[cols_b] if cols_b else b)
                prompt = PROMPT_COMPARE_FREQ_BURST.format(class_name=cls, csv_freq=csv_freq, csv_burst=csv_burst)

            # 호출(재시도)
            text = ""
            for attempt in range(3):
                try:
                    text = call_llm(client, model, prompt, max_output_tokens=max_output_tokens)
                    break
                except Exception as e:
                    print(f"[ERR] year={year}, class={cls}, attempt={attempt+1}: {e}")
                    time.sleep(2 * (attempt + 1))

            # 미리보기 출력
            if preview:
                print(f"\n----- PREVIEW year={year} class={cls} -----")
                print(text[:400] if text else "(EMPTY)")
                print("----- END PREVIEW -----\n")

            # 빈 응답이면 파일 생성 안 함(0B 방지)
            if not text.strip():
                print(f"[WARN] Empty output: year={year}, class={cls}. (skip write)")
                continue

            # 개별 파일 저장
            cls_file = safe_filename(str(cls))
            out_path = os.path.join(out_dir, f"{year}_{cls_file}_{mode}.md")
            with open(out_path, "w", encoding="utf-8") as fw:
                fw.write(text.strip() + "\n")
            print(f"[OK] {out_path}")

            # 연도 통합본에도 추가
            year_md.append(text.strip() + "\n\n")
            time.sleep(sleep_sec)

        # 연도 통합본 저장(내용이 하나라도 있을 때만)
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
                   help="trend=burst로 토픽 5개 / compare=빈도 vs burst 비교")
    p.add_argument("--sleep", type=float, default=0.3, help="요청 사이 쉬는 시간(초)")
    p.add_argument("--max_tokens", type=int, default=1200, help="최대 출력 토큰")
    p.add_argument("--preview", action="store_true", help="응답 미리보기 출력")
    args = p.parse_args()

    years = [int(x.strip()) for x in args.years.split(",") if x.strip()]
    run(args.input_dir, args.out_dir, years, args.mode, args.sleep, args.max_tokens, args.preview)
