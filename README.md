# COCO MVP 모델 평가

COCO 사전학습 탐지 모델로 `computer`, `book`, `other`를 평가합니다.
`computer`는 COCO의 `laptop`과 `tv`를 통합합니다.

## RTMDet-tiny top-k 비교 결과

기존 전체 평가 292장의 저장된 탐지 결과를 재집계했습니다. 학습이나 추가 추론 없이
동일한 탐지 결과에서 후보를 하나와 최대 두 개 허용했을 때를 비교했습니다.
추가로 저장된 [새 실행 결과](docs/rtmdet-topk/comparison-new-run/README.md)에서도
전체·클래스별 정답 포함률과 추가 정답 수가 동일했습니다.

| 정답 그룹 | 이미지 수 | Top-1 정답률 | Top-2 정답 포함률 | 증가 (%p) | 추가 정답 수 |
|---|---:|---:|---:|---:|---:|
| 전체 | 292 | 90.07% | 97.95% | 7.88 | 23 |
| computer | 200 | 89.00% | 97.50% | 8.50 | 17 |
| book | 19 | 63.16% | 94.74% | 31.58 | 6 |
| other | 73 | 100.00% | 100.00% | 0.00 | 0 |

![RTMDet-tiny top-1과 top-2 비교](docs/rtmdet-topk/results/topk-comparison.png)

Top-1에서 놓친 29장 중 23장은 두 번째 후보에 정답이 있었고, 6장은 top-2에서도
놓쳤습니다. 특히 book은 12/19장에서 18/19장으로 늘었지만 표본이 19장으로 작습니다.
전체 데이터의 200장이 computer이므로 전체 수치와 클래스별 수치를 함께 봐야 합니다.

**97.95%는 후보 두 개 안에 정답이 들어가는 비율입니다.** 단일 예측 정확도나
운영 수락·거절 성능이 97.95%가 됐다는 뜻은 아닙니다. 후보가 두 개인 이미지는
269/292장이므로, 실제 사용에는 후보 선택 또는 확인 절차가 필요합니다.
other의 100%도 other가 후보에 포함됐다는 의미이며 잘못된 대상 후보가 없다는 뜻은 아닙니다.

평가 조건은 CPU, RTMDet-tiny COCO 사전학습 가중치, 그룹별 임계값 0.05,
탐지 confidence 0.01, NMS IoU 0.6입니다. Top-k는 `computer/book/other` 그룹 순위이며
원본 COCO 클래스 순위나 탐지 박스 개수가 아닙니다.

RTMDet-tiny의 **top-1 / top-2 후보 비교**, 실행 명령과 Git용 그래프는
[비교 README](docs/rtmdet-topk/README.md)를 참고하세요. top-k는 추론 후 후보 수이며 재학습 없이 비교합니다.
집계 근거는 [CSV](docs/rtmdet-topk/results/comparison.csv)와
[설정·원본 탐지 해시](docs/rtmdet-topk/results/comparison.json)에 기록했습니다.

## 실행

```powershell
.\.venv\Scripts\python.exe scripts/run_inference.py
```

기본 설정의 7개 모델(YOLOv8n/s, RTMDet-tiny, NanoDet-Plus-m-320,
PicoDet-s-320, YOLOX-nano/tiny)을 실행합니다. 모델별 환경은
`config/inference.yaml`의 `python`에서 지정합니다. 기본 런타임은 CPU입니다.
가중치는 로컬 파일을 사용하며 모델 환경과 의존성이 설치되어 있어야 합니다.

```powershell
# 선택한 모델만 전체 데이터 평가
.\.venv\Scripts\python.exe scripts/run_inference.py --models yolov8n yolov8s --output-name mvp-evaluation
# Drive 경로 직접 지정
.\.venv\Scripts\python.exe scripts/run_inference.py --source "G:/내 드라이브/side_project/sample"
# 출처 카테고리마다 2장으로 실행 검증 (정식 성능 평가가 아님)
.\.venv\Scripts\python.exe scripts/run_inference.py --limit-per-category 2
# 저장한 탐지 결과로 리포트 재생성 (당시 설정 사용, 재추론 없음)
.\.venv\Scripts\python.exe scripts/run_inference.py --report-only outputs/coco_model_suite/RUN_NAME
```

각 실행은 새 폴더를 생성합니다. 동일한 `--output-name`은 덮어쓰지 않습니다.
모델 실패 시 다른 모델을 계속 실행하고, 리포트에 실패를 표시하며 종료 코드 1을 반환합니다.

## 입력 데이터와 정답

```text
sample/
├─ laptop/...
├─ tv/...       # 선택 사항; 실제 폴더가 있을 때 포함
├─ computer/...
├─ book/...
├─ bicycle/...  # bike, bycicle도 bicycle로 인식
├─ guitar/...
└─ 기타카테고리/...
```

최상위 카테고리 폴더명을 주 객체 정답으로 사용합니다.
`laptop`/`tv`/`computer`는 `computer`, `book`은 `book`, 나머지는 `other`입니다.
정규화된 출처 라벨과 원본 파일 경로는 별도로 보존합니다. COCO에 없는 `guitar`도 `other`의 음성 샘플로
사용할 수 있지만, 모델이 COCO 출력으로 `guitar`를 예측할 수는 없습니다.
하위 `open_image`, `public` 등의 이름은 정답에 영향을 주지 않습니다.

파일 해시로 중복을 제거하고 손상된 이미지를 제외합니다. 같은 이미지에 서로 다른
정답 폴더가 지정되어 있으면 오류로 중단합니다. 카테고리 폴더 밖의 이미지는
정답을 알 수 없어 오류로 처리합니다. Drive 원본은 읽기만 합니다.

## 판정과 지표

80개 COCO 클래스 전체를 탐지한 뒤, 각 그룹의 **최대 탐지 confidence**를 사용합니다.
`computer`는 laptop/tv 탐지 중 최대값입니다. 탐지 점수는 상호 배타적 확률이
아니므로 더하거나 재정규화하지 않습니다. 그룹별 임계값을 통과한 그룹 중 점수가
가장 높은 하나를 최종 예측으로 선택합니다. 통과한 탐지가 없으면 `other`로 거절합니다.
따라서 강한 bicycle 탐지는 약한 book 탐지보다 우선할 수 있습니다.
동점이면 설정 순서(computer, book, other)를 따릅니다.

- 전체 정확도: computer/book/other 정답과 최종 예측이 같은 이미지 비율
- 클래스별 precision = TP/(TP+FP), recall = TP/(TP+FN), F1 = 2TP/(2TP+FP+FN)
- 클래스별 FPR = FP/(FP+TN), FNR = FN/(TP+FN)
- 이미지 수락 오류율: 정답이 other인 이미지 중 computer/book으로 수락한 비율
- 이미지 거절 오류율: 정답이 computer/book인 이미지 중 other로 거절한 비율
- computer↔book 오판은 전체 정확도 및 클래스별 FPR/FNR에 반영
- confusion matrix: 행은 정답, 열은 예측. 원시 건수와 행 정규화 비율을 모두 저장
- macro 지표: 3개 클래스의 단순 평균. 분모가 0인 지표 및 이를 포함한 평균은 N/A
- 속도: 디스크 읽기를 제외한 전처리+추론+후처리, batch=1, warm-up 후 반복 측정

| 요구 기준 | 리포트 처리 |
|---|---|
| 목표 전체 판별 정확도 90% | 전체 이미지 정확도 ≥90% 판정 |
| 최소 허용 정확도 80% | 전체 이미지 정확도 ≥80% 판정 |
| 세션 기준 수락 오류 5% 이하 | 세션 정보가 없어 N/A |
| 세션 기준 거절 오류 20% 이하 | 세션 정보가 없어 N/A |

이미지 오류율로 세션 오류 기준의 통과 여부를 대체하지 않습니다.
샘플 실행은 `SMOKE_ONLY`로 표시하며 정확도 기준을 판정하지 않습니다.

이 지표는 **폴더 정답에 근거한 이미지 분류 평가**이며 바운딩 박스 탐지 mAP가 아닙니다.
비대상 사진에 실제 book/computer가 함께 있으면 오류가 과대 집계될 수 있습니다.
현재 폴더 데이터는 단일 주 객체 정답이라는 가정으로 평가하며, 실제 복수 객체나
세션 운영 성능을 측정하려면 별도 정답이 필요합니다.

모델마다 기본 입력 크기가 다릅니다. NanoDet와 PicoDet는 원래 head/export의 NMS를
사용하며 최소 점수는 각각 0.05, 0.025입니다. 그보다 낮은 판정 임계값은 허용하지 않습니다.
설정·입력 크기·후처리 조건·런타임을 결과에 남겨 비교 조건을 확인할 수 있게 합니다.

## 출력

```text
outputs/coco_model_suite/RUN_NAME/
├─ report.html                  # 브라우저에서 모델 비교, 지표, 혼동행렬 확인
├─ report.md
├─ summary.csv / summary.json
├─ run.json                     # 실행 설정, 데이터 목록/수, 샘플 실행 여부
└─ MODEL/
   ├─ report.md
   ├─ metrics.json
   ├─ predictions.csv           # 정답, 최종 예측, 그룹 점수, 원본 COCO 탐지
   ├─ per_class.csv             # TP/FP/FN/TN, precision/recall/F1/FPR/FNR
   ├─ confusion_matrix.csv
   ├─ confusion_matrix_normalized.csv
   ├─ confusion_matrix.png
   ├─ criteria.csv
   ├─ by_source.csv             # bicycle/guitar 등 출처별 예측 분포
   ├─ raw_coco_outcomes.csv      # 출처별 원본 COCO top-1 분포
   ├─ detections.json           # 재추론 없이 리포트를 만들기 위한 원본 결과
   ├─ job.json
   └─ inference.log
```

## 검증

```powershell
.\.venv\Scripts\python.exe tests/test_reports.py
```

수작업 계산 혼동행렬, 오류율 분모, 클래스 통합, 임계값, 음성 샘플 로딩,
지표의 N/A 처리와 리포트 파일 생성을 확인합니다.

어댑터 구현 참고: [Ultralytics predict](https://docs.ultralytics.com/modes/predict/),
[MMDetection](https://github.com/open-mmlab/mmdetection/blob/main/docs/en/get_started.md),
[PicoDet](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.9/configs/picodet).
YOLOX/NanoDet는 보관된 `third_party/`의 공식 추론 코드를 따릅니다.
