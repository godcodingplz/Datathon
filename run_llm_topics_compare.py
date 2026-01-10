import os
import time
import argparse
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


# =========================================================
# Prompt 템플릿
# - keyword는 반드시 CSV에 있는 것만 사용하도록 강제
# - 토픽은 최대 5개(부족하면 줄이기)
# =========================================================
PROMPT_TREND_TOPICS = """너는 학술 트렌드 분석가다. 아래 CSV는 공학 분야의 특정 연도(year) 및 중분류(NODE_CLSS_02)별 키워드 후보 목록이다.
반드시 CSV에 있는 keyword만 근거로 사용하고, CSV에 없는 키워드/정보를 새로 만들어내지 마라.

중요 규칙:
- 토픽은 '최대 5개'까지만 만든다.
- 후보 키워드가 부족하면(의미적으로 묶기 어려우면) 토픽 수를 1~4개로 줄여도 된다.
- 억지로 5개를 만들지 마라. 부족하면 부족하다고 명시하라.
- Keywords에는 반드시 CSV의 keyword만 포함하라(merged_variants에 있는 원문도 쓰지 말고 keyword만 써라).

작업:
1) 키워드들을 의미적으로 묶어 “트렌드 토픽”을 만든다. (최대 5개)
2) 각 토픽에 대해:
   - 토픽명(한국어 1개 + 영어 1개)
   - 대표 키워드 6~10개(반드시 CSV의 keyword에서만)
   - Evidence(burst): 해당 토픽에 속한 키워드 중 burst 상위 3개와 burst 수치
3) 마지막에 아래 2줄을 반드시 포함하라:
   - Candidate stats: total_rows=<정수>, unique_keywords=<정수>
   - Note: 후보 부족 여부와 그 이유(1문장)

출력 형식(그대로 지켜):
[중분류] {class_name}
- Topic 1: <토픽명KR> / <TopicNameEN>
  - Keywords: k1, k2, ...
  - Evidence(burst): kA(burst=...), kB(...), kC(...)
  - Summary: ...
...
- One-line: ...
- Candidate stats: total_rows=..., unique_keywords=...
- Note: ...

CSV:
{csv_text}
"""


PROMPT_COMPARE_FREQ_BURST = """너는 공학 논문 트렌드 분석가다. 아래에 두 개의 CSV가 있다.
A는 연도×중분류별 ‘빈도 TopN’, B는 같은 조건의 ‘트렌드 TopN(혼합: burst 우선 + freq 보충)’이다.
반드시 CSV에 있는 keyword만 사용하라. 외부지식/새 키워드 생성 금지.

작업:
1) “기본 관심사(빈도 상위)”와 “신규 트렌드(트렌드 상위)”를 비교해 차이를 3줄로 요약한다.
2) 아래를 뽑아라:
   - Stable keywords: A에는 있고 B에는 없는 키워드 중 상위 10개(빈도 기준)
   - Emerging keywords: B에는 있고 A에는 없는 키워드 중 상위 10개(가능하면 burst>0 우선, 없으면 트렌드순)
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

CSV B (trend mix):
{csv_trend}
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


def df_to_csv_text(df: pd.DataFrame, limit_rows: int = 140) -> str:
    # 프롬프트 너무 길어지는 것 방지(Top100이면 대체로 OK)
    if len(df) > limit_rows:
        df = df.head(limit_rows)
    return df.to_csv(index=False)


def safe_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in " _-." else "_" for c in s).strip()


# =========================================================
# 메인
# =========================================================
def run(input_dir: str, out_dir: str, years: list[int], mode: str,
        sleep_sec: float, max_output_tokens: int, preview: bool):
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY 환경변수가 없습니다. export OPENAI_API_KEY=... 로 설정하세요.")

    model = os.getenv("OPENAI_MODEL", "gpt-5-nano")  # 싼 모델
    client = OpenAI()

    os.makedirs(out_dir, exist_ok=True)

    for year in years:
        # ✅ 너가 만든 merged 산출물 기준으로 읽음
        freq_path = os.path.join(input_dir, f"{year}_freq_merged_top100_min5.csv")
        trend_path = os.path.join(input_dir, f"{year}_trend_mix_merged_top100_min5.csv")

        if mode == "trend":
            if not os.path.exists(trend_path):
                print(f"[SKIP] trend 파일 없음: {trend_path}")
                continue
            trend_df = pd.read_csv(trend_path)
            if "NODE_CLSS_02" not in trend_df.columns:
                print(f"[SKIP] NODE_CLSS_02 컬럼 없음: {trend_path}")
                continue

            classes = sorted(trend_df["NODE_CLSS_02"].dropna().unique().tolist())
            year_md = [f"# {year} 공학 트렌드 토픽 리포트 (merged)\n\n"]

            for cls in classes:
                sub = trend_df[trend_df["NODE_CLSS_02"] == cls].copy()

                cols = [c for c in ["Year","NODE_CLSS_02","keyword","freq","paper_count","share","share_prev","burst","merged_variants"]
                        if c in sub.columns]
                # ✅ LLM에는 keyword만 사용하라고 했으니, merged_variants는 참고용 컬럼으로만 넣고(있어도 ok)
                csv_text = df_to_csv_text(sub[cols] if cols else sub)

                prompt = PROMPT_TREND_TOPICS.format(class_name=cls, csv_text=csv_text)

                text = ""
                for attempt in range(3):
                    try:
                        text = call_llm(client, model, prompt, max_output_tokens=max_output_tokens)
                        break
                    except Exception as e:
                        print(f"[ERR] year={year}, class={cls}, attempt={attempt+1}: {e}")
                        time.sleep(2 * (attempt + 1))

                if preview:
                    print(f"\n----- PREVIEW trend year={year} class={cls} -----")
                    print(text[:500] if text else "(EMPTY)")
                    print("----- END PREVIEW -----\n")

                if not text.strip():
                    print(f"[WARN] Empty output: trend year={year}, class={cls} (skip)")
                    continue

                out_path = os.path.join(out_dir, f"{year}_{safe_filename(str(cls))}_trend_topics.md")
                with open(out_path, "w", encoding="utf-8") as fw:
                    fw.write(text.strip() + "\n")
                print(f"[OK] {out_path}")

                year_md.append(text.strip() + "\n\n")
                time.sleep(sleep_sec)

            year_all_path = os.path.join(out_dir, f"{year}_ALL_trend_topics.md")
            combined = "".join(year_md).strip()
            if combined:
                with open(year_all_path, "w", encoding="utf-8") as fw:
                    fw.write(combined + "\n")
                print(f"[OK] {year_all_path}")

        elif mode == "compare":
            if not os.path.exists(freq_path):
                print(f"[SKIP] freq 파일 없음: {freq_path}")
                continue
            if not os.path.exists(trend_path):
                print(f"[SKIP] trend 파일 없음: {trend_path}")
                continue

            freq_df = pd.read_csv(freq_path)
            trend_df = pd.read_csv(trend_path)

            if "NODE_CLSS_02" not in freq_df.columns or "NODE_CLSS_02" not in trend_df.columns:
                print(f"[SKIP] NODE_CLSS_02 컬럼 없음: {year}")
                continue

            classes = sorted(set(freq_df["NODE_CLSS_02"].dropna().unique().tolist())
                             | set(trend_df["NODE_CLSS_02"].dropna().unique().tolist()))

            year_md = [f"# {year} 공학 비교 리포트 (freq vs trend-mix, merged)\n\n"]

            for cls in classes:
                f = freq_df[freq_df["NODE_CLSS_02"] == cls].copy()
                t = trend_df[trend_df["NODE_CLSS_02"] == cls].copy()
                if f.empty and t.empty:
                    continue

                cols_f = [c for c in ["Year","NODE_CLSS_02","keyword","freq","paper_count","share","merged_variants"] if c in f.columns]
                cols_t = [c for c in ["Year","NODE_CLSS_02","keyword","freq","paper_count","share","share_prev","burst","merged_variants"] if c in t.columns]

                csv_freq = df_to_csv_text(f[cols_f] if cols_f else f)
                csv_trend = df_to_csv_text(t[cols_t] if cols_t else t)

                prompt = PROMPT_COMPARE_FREQ_BURST.format(class_name=cls, csv_freq=csv_freq, csv_trend=csv_trend)

                text = ""
                for attempt in range(3):
                    try:
                        text = call_llm(client, model, prompt, max_output_tokens=max_output_tokens)
                        break
                    except Exception as e:
                        print(f"[ERR] year={year}, class={cls}, attempt={attempt+1}: {e}")
                        time.sleep(2 * (attempt + 1))

                if preview:
                    print(f"\n----- PREVIEW compare year={year} class={cls} -----")
                    print(text[:500] if text else "(EMPTY)")
                    print("----- END PREVIEW -----\n")

                if not text.strip():
                    print(f"[WARN] Empty output: compare year={year}, class={cls} (skip)")
                    continue

                out_path = os.path.join(out_dir, f"{year}_{safe_filename(str(cls))}_compare.md")
                with open(out_path, "w", encoding="utf-8") as fw:
                    fw.write(text.strip() + "\n")
                print(f"[OK] {out_path}")

                year_md.append(text.strip() + "\n\n")
                time.sleep(sleep_sec)

            year_all_path = os.path.join(out_dir, f"{year}_ALL_compare.md")
            combined = "".join(year_md).strip()
            if combined:
                with open(year_all_path, "w", encoding="utf-8") as fw:
                    fw.write(combined + "\n")
                print(f"[OK] {year_all_path}")

        else:
            raise ValueError("mode는 trend 또는 compare만 가능")

    print("\nDONE")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True, help="merged 산출물 폴더(YYYY_freq_merged_top100_min5.csv 등이 있는 곳)")
    p.add_argument("--out_dir", default="./llm_reports_merged", help="결과 md 저장 폴더")
    p.add_argument("--years", default="2021,2022,2023,2024,2025", help="쉼표로 구분")
    p.add_argument("--mode", choices=["trend", "compare"], default="trend",
                   help="trend=토픽 생성 / compare=빈도 vs 트렌드 비교")
    p.add_argument("--sleep", type=float, default=0.3, help="요청 사이 쉬는 시간(초)")
    p.add_argument("--max_tokens", type=int, default=1200, help="최대 출력 토큰")
    p.add_argument("--preview", action="store_true", help="응답 미리보기 출력")
    args = p.parse_args()

    years = [int(x.strip()) for x in args.years.split(",") if x.strip()]
    run(args.input_dir, args.out_dir, years, args.mode, args.sleep, args.max_tokens, args.preview)
