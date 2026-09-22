# RTMDet-tiny top-k 1·2 실험

추가된 Google Drive 이미지를 COCO 사전학습 RTMDet-tiny로 추론하고 동일한 탐지 결과에서 top-1과 top-2를 비교합니다.
이 브랜치는 추론 재평가용이며 가중치를 갱신하는 fine-tuning은 수행하지 않습니다.
현재 데이터는 폴더별 이미지 정답이며, 탐지 모델 학습에 필요한 박스 주석과 학습/검증/테스트 분할은 없습니다.

## 데이터와 판정

[Google Drive sample](https://drive.google.com/drive/folders/1H-wfZyZIEnoOGA3Mr6yhfgsb0a2cdCXR)을 마운트 경로에서 직접 읽습니다. 원본은 수정하지 않습니다.

- computer: laptop/tv 탐지, book: book 탐지, 나머지: other.
- 정답은 최상위 폴더입니다. bed/table/bicycle/guitar 폴더는 other로 평가합니다.
- SHA-256으로 동일 이미지 중복을 제거합니다. 서로 다른 라벨의 중복은 오류 처리합니다.
- 그룹별 최대 confidence를 사용하고 0.05 이상 그룹을 점수순으로 선택합니다.
- Top-1은 최대 1개, top-2는 최대 2개 후보입니다. 후보가 없으면 other입니다.
- Top-2 정답 포함률은 단일 분류 정확도와 다릅니다. 등록 카테고리별 Precision/Recall/FPR/FNR도 함께 봅니다.

## 실행

기존 `.venv`는 데이터 검증·보고서용, `.model_envs/rtmdet`은 RTMDet 추론용입니다.
보고서 의존성은 `requirements.txt`, 모델 환경 버전은 `requirements-rtmdet.txt`에 기록했습니다.
모델 환경은 Python 3.11과 PyTorch에 맞는 MMCV 바이너리가 필요합니다.
가중치 경로는 `models/pretrained/rtmdet_tiny.pth`입니다.

```powershell
.venv/Scripts/python.exe scripts/run_inference.py --top-k 2 --output-name drive-20260922
.venv/Scripts/python.exe scripts/compare_topk.py --run outputs/rtmdet/drive-20260922 --output docs/rtmdet-topk/drive-20260922
.venv/Scripts/python.exe -m unittest discover -s tests -p "test_*.py"
```

재실행할 때는 새 output-name과 output 경로를 사용합니다. `--source`로 데이터 위치를 지정할 수 있습니다.
기본 CPU, 입력 640, 탐지 confidence 0.01, NMS IoU 0.6을 사용합니다.
Top-1·2는 같은 탐지 결과를 재사용합니다. 속도는 워밍업 5회 후 이미지당 3회 측정하며 top-k 자체의 속도 비교가 아닙니다.

`outputs/rtmdet/<실행명>/run.json`에는 이미지 해시와 설정,
`rtmdet-tiny/detections.json`에는 원본 탐지 결과가 저장됩니다.
공유 보고서에는 원본 이미지 경로 대신 집계 결과와 탐지 파일 해시를 기록합니다.
`--limit-per-category` 실행은 SMOKE_ONLY로 표시되어 전체 평가와 구분됩니다.

YOLO/PicoDet/NanoDet 어댑터·설정·Git 참조, 이전 다중 모델 보고서·예제 생성 코드는 제거했습니다.
로컬의 Git 제외 모델 환경·가중치·다운로드는 보존합니다.

## 데이터 갱신 내역 (2026-09-22)

파일 451개에서 중복 79개를 제외한 372장을 평가합니다. 손상된 이미지는 없었습니다.
기존 292장은 모두 유지되며, 추가된 고유 이미지는 book 52장·bed 13장·table 15장입니다.
평가 구성은 computer 200장, book 71장, other 101장입니다.
[데이터 점검 JSON](docs/rtmdet-topk/data-audit.json)에 집계 내역을 기록했습니다.

## 비교 결과

[전체 비교 보고서](docs/rtmdet-topk/drive-20260922/README.md) · [설정·집계·가중치 해시](docs/rtmdet-topk/drive-20260922/comparison.json)

| 정답 그룹 | 이미지 수 | Top-1 정확도 | Top-2 정답 포함률 |
|---|---:|---:|---:|
| 전체 | 372 | 87.63% (326/372) | 98.39% (366/372) |
| computer | 200 | 89.00% | 97.50% |
| book | 71 | 67.61% | 98.59% |
| other | 101 | 99.01% | 100.00% |

![Top-k 비교](docs/rtmdet-topk/drive-20260922/topk-comparison.png)

Top-2는 정답 40장을 추가로 포함하지만 잘못된 후보도 늘립니다.
등록 카테고리가 후보에 있으면 수락하는 정책에서는 computer FPR이 1.74%에서 11.63%로,
book FPR이 0.33%에서 23.92%로 증가합니다. Book precision은 97.96%에서 49.30%로 감소합니다.
따라서 top-2 포함률을 자동 승인 정확도로 해석하지 않습니다.
사진에 여러 객체가 함께 있을 수 있으므로 폴더의 단일 정답 라벨에 따른 평가라는 한계도 있습니다.

위 결과는 추가 데이터에 대한 **사전학습 모델의 추론 재평가**입니다.

## 탐지 특징 학습 결과

박스 주석이 없으므로 RTMDet 가중치는 고정했습니다. 대신 RTMDet가 출력한 80개 COCO 클래스의
최대 confidence와 탐지 수를 특징으로 computer/book/other 후처리 분류기를 학습했습니다.
성능은 각 이미지를 해당 fold의 학습에서 제외한 nested 5-fold OOF 예측으로 계산했습니다.

| 모델 | Top-1 | Top-2 정답 포함률 |
|---|---:|---:|
| 기존 규칙 | 87.63% | 98.39% |
| 학습 후처리기 (OOF) | 94.35% | 98.92% |

[학습 보고서](docs/rtmdet-topk/trained-postprocessor-20260922/README.md) ·
[학습 지표와 해시](docs/rtmdet-topk/trained-postprocessor-20260922/metrics.json)

```powershell
.venv/Scripts/python.exe scripts/train_topk_postprocessor.py `
  --run outputs/rtmdet/drive-20260922 `
  --output docs/rtmdet-topk/trained-postprocessor-20260922 `
  --model-output models/trained/rtmdet_topk_postprocessor.joblib

.venv/Scripts/python.exe scripts/apply_topk_postprocessor.py `
  --model models/trained/rtmdet_topk_postprocessor.joblib `
  --detections outputs/rtmdet/drive-20260922/rtmdet-tiny/detections.json `
  --output outputs/rtmdet/trained-predictions.json --top-k 1
```

최종 후처리기는 372장 전체로 학습했습니다. 보고된 수치는 별도 외부 테스트셋 성능이 아니며,
새 촬영 환경에 배포하기 전 분리된 테스트 데이터로 재검증해야 합니다.
