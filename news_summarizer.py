# news_summarizer_ko.py
"""
Korean News Summarizer (콘솔 버전)

- 한국어 뉴스 기사 텍스트를 입력받아 요약을 생성
- KoBART 기반 한국어 요약 모델 사용 (Hugging Face)
- 긴 텍스트는 여러 조각으로 나누어 순차적으로 요약
- 요약 결과를 로그로 저장하고 간단한 평가 지표를 제공
"""

import os
import csv
from datetime import datetime
from typing import List, Dict

import torch
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration

# -----------------------------
# 1. 모델 로딩
# -----------------------------
print("🛠 한국어 요약 모델 로딩 중입니다... (처음 한 번은 조금 걸릴 수 있습니다)")
tokenizer = PreTrainedTokenizerFast.from_pretrained("gogamza/kobart-summarization")
model = BartForConditionalGeneration.from_pretrained("gogamza/kobart-summarization")
print("✅ 모델 준비 완료!\n")

# -----------------------------
# 2. 요약 모드 설정
# -----------------------------
SUMMARY_MODES: Dict[str, dict] = {
    "1": {"name": "짧은 요약", "max_len": 128, "min_len": 32},
    "2": {"name": "기본 요약", "max_len": 256, "min_len": 64},
    "3": {"name": "긴 요약", "max_len": 384, "min_len": 96},
}

# -----------------------------
# 3. 로그 설정
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_PATH = os.path.join(LOG_DIR, "summary_log.csv")


def ensure_log_file() -> None:
    """로그 디렉토리와 CSV 파일이 없으면 생성."""
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "mode",
                    "input_length_chars",
                    "input_length_words",
                    "summary_length_chars",
                    "summary_length_words",
                    "compression_ratio",
                    "overlap_ratio",
                    "summary_preview",
                ]
            )


def append_log(
    mode_name: str,
    original: str,
    summary: str,
    compression_ratio: float,
    overlap_ratio: float,
) -> None:
    """요약 결과를 로그 파일에 한 줄 추가."""
    ensure_log_file()
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                mode_name,
                len(original),
                len(original.split()),
                len(summary),
                len(summary.split()),
                f"{compression_ratio:.3f}",
                f"{overlap_ratio:.3f}",
                summary[:80].replace("\n", " ") + ("..." if len(summary) > 80 else ""),
            ]
        )


# -----------------------------
# 4. 요약 관련 함수들
# -----------------------------
def split_into_chunks(text: str, max_chars: int = 800) -> List[str]:
    """
    긴 텍스트를 max_chars 길이 이하의 여러 조각으로 나눈다.
    문장 경계(줄바꿈, 마침표)를 우선적으로 기준으로 자르려고 시도.
    """
    text = text.strip()
    chunks: List[str] = []

    while len(text) > max_chars:
        chunk = text[:max_chars]

        cut_pos = max(chunk.rfind("\n"), chunk.rfind("."))
        if cut_pos == -1 or cut_pos < max_chars * 0.4:
            cut_pos = max_chars

        chunks.append(text[:cut_pos].strip())
        text = text[cut_pos:].strip()

    if text:
        chunks.append(text.strip())

    return chunks


def generate_summary_once(text: str, max_len: int, min_len: int) -> str:
    """
    KoBART를 사용해 한 번 요약을 수행하는 함수.
    (한 덩어리 텍스트에 대해 한 번 요약)
    """
    raw_input_ids = tokenizer.encode(text)
    input_ids = [tokenizer.bos_token_id] + raw_input_ids + [tokenizer.eos_token_id]
    input_ids = torch.tensor([input_ids])

    summary_ids = model.generate(
        input_ids,
        max_length=max_len,
        min_length=min_len,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        num_beams=4,
        no_repeat_ngram_size=3,
    )
    summary = tokenizer.decode(summary_ids[0].tolist(), skip_special_tokens=True)
    return summary.strip()


def summarize_korean_news(
    text: str,
    max_len: int = 256,
    min_len: int = 64,
) -> str:
    """
    한국어 뉴스 텍스트를 요약하는 함수.
    텍스트가 길면 여러 조각으로 나눠서 요약 후 다시 합침.
    """
    text = text.strip()

    if not text:
        return "⚠️ 입력된 텍스트가 없습니다."

    if len(text) < 150:
        return "⚠️ 요약할 만큼 충분히 긴 한국어 뉴스 본문을 입력해 주세요. (최소 150자 이상 권장)"

    chunks = split_into_chunks(text, max_chars=800)

    summaries: List[str] = []
    for i, chunk in enumerate(chunks, start=1):
        try:
            chunk_summary = generate_summary_once(chunk, max_len, min_len)
        except Exception as e:
            chunk_summary = f"[⚠️ {i}번째 조각 요약 중 오류 발생: {e}]"
        summaries.append(chunk_summary)

    combined_summary = "\n".join(summaries).strip()

    if len(chunks) > 1 and len(combined_summary) > 400:
        try:
            final_summary = generate_summary_once(
                combined_summary,
                max_len=max_len,
                min_len=min_len,
            )
            return final_summary
        except Exception:
            return combined_summary
    else:
        return combined_summary


# -----------------------------
# 5. 간단 평가 함수
# -----------------------------
def evaluate_summary(original: str, summary: str) -> Dict[str, float]:
    """
    요약 품질을 아주 간단히 평가하는 함수.
    - compression_ratio: 요약 길이 / 원문 길이 (단어 기준)
    - overlap_ratio: 원문 단어 집합과 요약 단어 집합의 겹치는 비율
    """
    orig_words = [w for w in original.split() if w.strip()]
    summ_words = [w for w in summary.split() if w.strip()]

    if not orig_words or not summ_words:
        return {"compression_ratio": 0.0, "overlap_ratio": 0.0}

    compression_ratio = len(summ_words) / len(orig_words)

    orig_set = set(orig_words)
    summ_set = set(summ_words)
    overlap = len(orig_set & summ_set)
    overlap_ratio = overlap / len(orig_set) if orig_set else 0.0

    return {
        "compression_ratio": compression_ratio,
        "overlap_ratio": overlap_ratio,
    }


# -----------------------------
# 6. 입력 관련 함수
# -----------------------------
def input_article_from_console() -> str:
    """
    콘솔에서 여러 줄로 한국어 뉴스 기사 텍스트를 입력받는 함수.
    빈 줄만 입력하면 종료.
    """
    print("=== 한국어 뉴스 기사 텍스트를 입력하세요. ===")
    print("입력을 마치려면 빈 줄에서 Enter를 누르세요.\n")

    lines: List[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "":
            break
        lines.append(line)
    return "\n".join(lines)


def input_article_from_file() -> str:
    """
    .txt 파일 경로를 입력받아서 한국어 뉴스 텍스트를 읽어오는 함수.
    """
    path = input("불러올 .txt 파일 경로를 입력하세요: ").strip()
    if not path:
        print("⚠️ 경로가 비어 있습니다.\n")
        return ""

    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        print(f"✅ 파일을 성공적으로 불러왔습니다. (길이: {len(text)} 글자)\n")
        return text
    except FileNotFoundError:
        print("⚠️ 파일을 찾을 수 없습니다. 경로를 다시 확인하세요.\n")
        return ""
    except Exception as e:
        print(f"⚠️ 파일을 여는 중 오류가 발생했습니다: {e}\n")
        return ""


# -----------------------------
# 7. 하나의 요약 흐름
# -----------------------------
def choose_summary_mode() -> tuple[str, int, int]:
    """
    요약 모드를 선택하거나, 직접 max_len/min_len을 입력할 수 있도록 한다.
    """
    print("\n=== 요약 모드 선택 ===")
    for key, info in SUMMARY_MODES.items():
        print(f"{key}. {info['name']} (max_len={info['max_len']}, min_len={info['min_len']})")
    print("4. 직접 길이 입력 (고급 설정)")

    choice = input("모드를 선택하세요 (기본 2번): ").strip() or "2"

    if choice in SUMMARY_MODES:
        info = SUMMARY_MODES[choice]
        return info["name"], int(info["max_len"]), int(info["min_len"])
    else:
        print("직접 길이를 입력합니다.")
        try:
            max_len = int(input("max_len (기본 256): ").strip() or "256")
            min_len = int(input("min_len (기본 64): ").strip() or "64")
        except ValueError:
            print("⚠️ 숫자가 아니어서 기본값(256/64)로 진행합니다.")
            max_len, min_len = 256, 64
        return "사용자 지정", max_len, min_len


def summarize_flow(text: str):
    """
    텍스트를 전달받아 요약 모드 선택, 요약, 평가, 로그 저장까지 수행.
    """
    if not text:
        print("⚠️ 텍스트가 비어 있어서 요약을 진행할 수 없습니다.\n")
        return

    mode_name, max_len, min_len = choose_summary_mode()

    print("\n📰 한국어 뉴스 요약을 생성하는 중입니다...\n")
    summary = summarize_korean_news(text, max_len=max_len, min_len=min_len)

    print("=== 요약 결과 ===")
    print(summary)
    print("\n=== 통계 ===")
    orig_words = len(text.split())
    summ_words = len(summary.split())
    print(f"- 원문 단어 수(공백 기준): {orig_words}")
    print(f"- 요약 단어 수(공백 기준): {summ_words}")

    if not summary.startswith("⚠️"):
        metrics = evaluate_summary(text, summary)
        compression_ratio = metrics["compression_ratio"]
        overlap_ratio = metrics["overlap_ratio"]
        print(f"- 압축 비율: {compression_ratio*100:.1f}% (요약/원문)")
        print(f"- 단어 겹침 비율: {overlap_ratio*100:.1f}% (원문 단어 집합 대비)")

        append_log(mode_name, text, summary, compression_ratio, overlap_ratio)

    print()  # 빈 줄


# -----------------------------
# 8. 메인 메뉴 (3번 로그 보기 제거)
# -----------------------------
def main():
    """
    콘솔 메뉴를 통해 한국어 뉴스 요약을 실행하는 메인 함수.
    """
    while True:
        print("===== Korean News Summarizer =====")
        print("1. 콘솔에 직접 한국어 뉴스 본문 입력 후 요약")
        print("2. .txt 파일에서 한국어 뉴스 불러와 요약")
        print("3. 종료")
        choice = input("메뉴 번호를 선택하세요: ").strip()

        if choice == "1":
            article = input_article_from_console()
            summarize_flow(article)

        elif choice == "2":
            article = input_article_from_file()
            summarize_flow(article)

        elif choice == "3":
            print("프로그램을 종료합니다.")
            break

        else:
            print("⚠️ 잘못된 입력입니다. 1~3 중에서 선택하세요.\n")


if __name__ == "__main__":
    main()
