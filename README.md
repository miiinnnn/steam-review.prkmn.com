# 스팀 버그·최적화 리뷰 분석기 (Web)

Steam 게임 이름 일부만 입력해 자동완성으로 게임을 선택하면, 최신 한국어 리뷰 1,000개씩 불러와 **버그 및 최적화 이슈 리뷰만** 골라서 보여주는 웹 애플리케이션입니다.

## 프로젝트 구조

```text
.
├── vercel.json                     # Vercel 배포 설정 (Serverless 함수 타임아웃 및 모델 포함)
├── public/
│   └── index.html                  # 웹 UI (자동완성 검색, 버그·최적화 이슈 리뷰 뷰어)
├── api/
│   ├── model-info.mjs              # Vercel API: 모델 성능 정보
│   ├── search.mjs                  # Vercel API: Steam 게임 자동완성 검색
│   └── reviews.mjs                 # Vercel API: Steam 리뷰 수집 및 실시간 이진 분류
├── web/
│   ├── package.json
│   ├── lib.mjs                     # 공통 분류기 및 Steam API 모듈
│   └── server.mjs                  # 로컬 실행용 Node.js 웹 서버
├── data/
│   ├── reviews_raw.csv             # 원본 Steam 한국어 리뷰 데이터 (127,615개)
│   └── reviews_labeled.csv         # 이진 라벨링된 학습용 리뷰 (1: 버그·최적화, 0: 그 외)
├── scripts/
│   ├── collect_reviews.py          # 0단계: Steam 한국어 리뷰 대량 수집
│   ├── label_reviews.py            # 1단계: 리뷰 이진 라벨링
│   └── train_model.py              # 2단계: 전체 리뷰 학습 및 경량 JSON 모델 추출
└── model/
    ├── bug_opt_classifier.json     # 학습된 TF-IDF + LogisticRegression 이진 분류기
    └── metrics.json                # 검증/테스트 성능 지표
```

## 로컬에서 실행

Node.js 20 이상에서 별도 패키지 설치 없이 바로 실행할 수 있습니다.

```powershell
node web/server.mjs
```

브라우저에서 `http://127.0.0.1:3000` 에 접속하면 됩니다. (포트 변경: `$env:PORT=8080; node web/server.mjs`)

## Vercel 배포

1. 이 저장소를 GitHub에 푸시합니다.
2. [Vercel](https://vercel.com)에서 **Add New Project → Import Git Repository**로 저장소를 선택합니다.
3. 기본 설정(`Root Directory: ./`, `Framework Preset: Other`) 그대로 **Deploy**를 누르면 자동으로 배포됩니다.

## 데이터 라벨링 및 모델 재학습

```powershell
python -m pip install -r requirements.txt
python scripts/label_reviews.py
python scripts/train_model.py
```
