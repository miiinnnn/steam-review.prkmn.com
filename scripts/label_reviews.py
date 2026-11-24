"""
Step 1: Select 5,000 Korean Steam reviews from data/reviews_raw.csv and
perform binary labeling:
  1 = 버그 및 최적화 이슈 리뷰 (Bug & Optimization Issue)
  0 = 그 외 리뷰 (Other)
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data" / "reviews_raw.csv"
OUT_CSV = ROOT / "data" / "reviews_labeled.csv"

CLAUSE_SPLIT_RE = re.compile(
    r"[\n.!?。]+|\s+(?:근데|그런데|하지만|다만|그러나|그래도)\s+"
)

# Strong bug / crash / stability / technical issue patterns
BUG_PATTERNS = [
    r"버그",
    r"오류",
    r"에러",
    r"크래[시쉬]",
    r"[튕팅퉁](?:김|겨|기|겼|긴|길|겼|구)",
    r"강종|강제\s*종료",
    r"프리징",
    r"먹통",
    r"블랙\s*스크린|블루\s*스크린|흰\s*화면|검은\s*화면",
    r"무한\s*로딩",
    r"응답\s*없음",
    r"실행\s*(?:불가|안\s*됨|안\s*돼|안\s*되|오류|실패|문제|이\s*안)",
    r"(?:게임|겜)이?\s*(?:안\s*켜|켜지\s*지\s*않|켜지질|안\s*돌아|꺼져|꺼지|꺼짐)",
    r"(?:갑자기|자꾸|계속|자주)\s*(?:꺼져|꺼지|꺼짐|멈춰|멈춤|멈추|튕|팅)",
    r"(?:화면|게임|겜|로딩|컴|컴퓨터|피시|pc)[^\n.!?]{0,12}(?:멈춤|멈춰|멈추|얼어|얼음|굳어|굳음)",
    r"멈춤\s*현상|멈춰\s*버|멈춰서",
    r"(?:세이브|저장|파일)[^\n.!?]{0,15}(?:날아|초기화|깨[지졌짐]|손상|파손|덮어씌|안\s*됨|안\s*돼|안\s*되|오류|사라|증발|실패|버그)",
    r"(?:컴퓨터|pc|게임|모드|엔진|드라이버|안티치트)\s*(?:랑|와|과)?\s*충돌|충돌\s*(?:나|발생|오류|현상)",
    r"(?:접속|로그인|연결)\s*(?:불가|안\s*됨|안\s*돼|안\s*되|오류|실패|끊[김겨기]|장애|불안|지연)",
    r"서버\s*(?:터[짐져지]|불안|렉|랙|끊[김겨기]|오류|장애|문제|다운|먹통|상태\s*가?\s*(?:개판|별로|안\s*좋|최악|구림|구려))",
    r"진행\s*(?:불가|막힘|안\s*됨|안\s*돼|버그)|퀘스트\s*(?:버그|진행\s*불가|안\s*깨)",
    r"(?:맵|지형|벽|바닥)\s*(?:뚫|끼임|끼어|빠짐)|글리치",
    r"(?:소리|사운드|오디오|음성)\s*(?:안\s*나|끊[김겨기]|깨[짐져지]|버그|오류|밀림)",
    r"(?:한글|폰트|글자|텍스트|화면|그래픽|텍스처)\s*(?:깨[짐져지]|뭉개|깜빡|깜박|안\s*보|출력\s*오류)",
    r"(?:패드|컨트롤러|키보드|마우스)\s*(?:인식\s*(?:불가|안\s*됨|안\s*돼|안\s*되|오류|문제)|먹통|씹힘|입력\s*지연)",
    r"패킷\s*로스|핑\s*(?:튀|높|심|문제|터|개판)|네트워크\s*(?:오류|끊|불안|지연|문제)",
    r"\b(?:bug|bugs|buggy|crash|crashes|crashing|freeze|freezing|glitch|error\s*code)\b",
]

# Strong optimization / performance / frame / lag / spec patterns
OPT_PATTERNS = [
    r"발적화",
    r"최적화",
    r"(?:프레임|fps|frame)\s*(?:드랍|드롭|떨어|저하|하락|방어|고정|제한|끊|낮|안\s*나|출렁|요동|박살|개판|문제|이슈|심하|불안|유지|60|30)",
    r"(?:30|60)\s*(?:프레임|fps)(?:\s*(?:고정|제한|락|방어|인\s*거|밖에|뿐))?",
    r"(?<![가-힣])(?:렉|랙)(?:이|도|은|는|을|를|걸|심|때문|땜|때매|있|많|쩔|장난|지리|나[서고는면]|생기|발생|\s|$|[,.!?~])",
    r"(?:서버|게임|겜|화면|프레임|핑|심한|개|존나|미친)\s*(?:렉|랙)",
    r"버벅(?:임|거|대|이)",
    r"(?:화면|게임|겜|프레임|영상|모션|움직임|실행|플레이)\s*(?:이|가)?\s*끊[김겨기]",
    r"끊김\s*(?:현상|심|이|도|때문|많)|자꾸\s*끊[겨기]|계속\s*끊[겨기]|심하게\s*끊[겨기]",
    r"스터터링|stutter(?:ing)?",
    r"(?:로딩|loading)\s*(?:이|가|시간|속도)?\s*(?:너무|진짜|존나|개|겁나|엄청|되게|좀)?\s*(?:길|느리|느려|오래|심|답답|지연|무한|멈)",
    r"메모리\s*(?:누수|부족|점유|터짐|과다|폭발)|램\s*(?:누수|부족|점유|터짐|과다)",
    r"(?:cpu|gpu|글카|그래픽\s*카드)\s*(?:점유|발열|과부하|병목|터지|비명|온도|팬\s*소음|로드율|갈굼|혹사|만?\s*잡아\s*먹)",
    r"[셰쉐]이더\s*(?:컴파일|로딩|문제|오류|스터터)",
    r"(?:고사양|사양)\s*(?:을|이)?\s*(?:너무|많이|심하게|과하게|타|요구|높)|권장\s*사양|최소\s*사양",
    r"(?:해상도|울트라\s*와이드|21:9|창\s*모드|전체\s*화면|수직\s*동기화|v-?sync|dlss|fsr)\s*(?:미지원|지원\s*안|오류|문제|버그|깨|안\s*됨|안\s*돼|안\s*되)",
    r"성능\s*(?:저하|문제|이슈|개선\s*필요)",
    r"\b(?:lag|laggy|lagging|fps\s*drop|low\s*fps|optimi[sz]ation|poorly\s*optimi[sz]ed|memory\s*leak)\b",
]

COMBINED_ISSUE_RE = re.compile(
    "|".join(f"(?:{p})" for p in BUG_PATTERNS + OPT_PATTERNS), re.IGNORECASE
)

# Patterns that indicate the clause is ONLY praising optimization/stability (no issue)
POSITIVE_ONLY_RE = re.compile(
    r"(?:최적화|프레임|로딩|성능)\s*(?:이|가|도|은|는)?\s*(?:진짜|정말|매우|너무|아주|완전|엄청|꽤|잘|좋|훌륭|갓|최고|대박|미쳤|괜찮|안정적|쾌적|빠르|빨라|부드럽|부드러)"
    r"|(?:버그|오류|렉|랙|끊김|튕김|팅김|프레임\s*드랍|프리징|크래[시쉬])\s*(?:이|가|도|은|는)?\s*(?:없|안\s*걸|안\s*생|전혀\s*없|거의\s*없|적고|적음|못\s*봤|못\s*느꼈)"
    r"|(?:저사양|똥컴|노트북|내장\s*그래픽|구형\s*컴)\s*(?:에서도|으로도|도|으로)\s*(?:잘\s*돌아|쾌적|부드럽|원활|가능|무난)",
    re.IGNORECASE,
)

# Negative/complaint indicators that override POSITIVE_ONLY_RE if present in the same clause
STRONG_COMPLAINT_RE = re.compile(
    r"발적화|개판|최악|쓰레기|답\s*없|고쳐|고치|패치\s*좀|수정\s*좀|심하|심해|심함|불편|짜증|빡치|환불|포기|못\s*해먹|할\s*수가\s*없|"
    r"튕|팅|강종|프리징|크래[시쉬]|먹통|안\s*됨|안\s*돼|안\s*되|불가|날아|초기화|증발|터짐|터져|끊김|끊겨|버벅|스터터링|누수|드랍|드롭|떨어|저하|고정|제한",
    re.IGNORECASE,
)

# False positive guards for specific ambiguous terms
FALSE_STOP_RE = re.compile(
    r"(?:손이|시간이|플레이|중독|웃음|눈물|재미|열정|생각|발걸음|심장)\s*(?:이|가|을|를)?\s*멈추|멈출\s*수\s*없|멈추지\s*않",
    re.IGNORECASE,
)


def is_bug_or_opt_clause(clause: str) -> bool:
    c = clause.strip()
    if not c:
        return False
    # Remove false "can't stop playing" idioms before matching
    cleaned = FALSE_STOP_RE.sub("", c)
    if not COMBINED_ISSUE_RE.search(cleaned):
        return False
    # If it matches positive-only praise without any strong complaint in the same clause, exclude it
    if POSITIVE_ONLY_RE.search(cleaned) and not STRONG_COMPLAINT_RE.search(cleaned):
        return False
    return True


def classify_review(text: str) -> int:
    for clause in CLAUSE_SPLIT_RE.split(text):
        if is_bug_or_opt_clause(clause):
            return 1
    return 0


def main() -> None:
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_CSV}")

    with RAW_CSV.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    # Deduplicate and keep Korean (Hangul-containing) non-empty reviews
    seen: set[str] = set()
    hangul_rows: list[dict[str, str]] = []
    for r in rows:
        rid = r.get("review_id", "").strip()
        txt = (r.get("review_text") or "").strip()
        if rid and rid not in seen and txt and re.search(r"[가-힣]", txt):
            seen.add(rid)
            hangul_rows.append(r)

    selected = hangul_rows

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pos_count = 0
    neg_count = 0

    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "review_id",
                "app_id",
                "game_name",
                "recommended",
                "playtime_at_review_hours",
                "review_text",
                "label",
                "label_name",
            ],
        )
        writer.writeheader()
        for r in selected:
            label = classify_review(r["review_text"])
            if label == 1:
                pos_count += 1
            else:
                neg_count += 1
            writer.writerow(
                {
                    "review_id": r["review_id"],
                    "app_id": r["app_id"],
                    "game_name": r["game_name"],
                    "recommended": r.get("recommended", ""),
                    "playtime_at_review_hours": r.get("playtime_at_review_hours", ""),
                    "review_text": r["review_text"],
                    "label": label,
                    "label_name": "버그·최적화 이슈" if label == 1 else "그 외 리뷰",
                }
            )

    print(f"Labeled {len(selected)} reviews -> {OUT_CSV}")
    print(f"  1 (버그·최적화 이슈): {pos_count} ({pos_count / len(selected) * 100:.2f}%)")
    print(f"  0 (그 외 리뷰):       {neg_count} ({neg_count / len(selected) * 100:.2f}%)")


if __name__ == "__main__":
    main()
