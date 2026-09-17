# RTMDet-tiny top-1 / top-2 비교

기존 전체 평가 292장에서 top-1 정답률은 **90.07% (263/292)**,
top-2 정답 포함률은 **97.95% (286/292)**였습니다. 후보를 최대 두 개로 늘리면
23장의 정답을 추가로 포함하며, 증가 폭은 **7.88%p**입니다.

| 정답 그룹 | 이미지 수 | Top-1 | Top-2 | 추가 정답 수 |
|---|---:|---:|---:|---:|
| computer | 200 | 89.00% (178/200) | 97.50% (195/200) | 17 |
| book | 19 | 63.16% (12/19) | 94.74% (18/19) | 6 |
| other | 73 | 100.00% (73/73) | 100.00% (73/73) | 0 |

![RTMDet-tiny top-k 비교](results/topk-comparison.png)

## 결과 해석

Top-1 오답 29장 중 23장은 두 번째 후보로 복구되고 6장은 여전히 정답을 포함하지
못했습니다. book의 증가 폭은 31.58%p로 가장 크지만 표본이 19장뿐이므로
추가 데이터에서도 확인해야 합니다. computer는 8.50%p 증가했습니다.

후보가 두 개인 이미지는 269장, 하나인 이미지는 23장입니다. 세 그룹 중 두 그룹을
허용한 결과이므로 포함률 상승만으로 모델이나 자동 판별 성능이 개선됐다고
판단할 수 없습니다. 실제 적용 시 두 후보 중 선택하는 절차를 별도로 평가해야 합니다.
other 포함률 100% 역시 잘못된 computer/book 후보가 함께 나오지 않았다는 뜻은 아닙니다.

이번 비교는 `mvp-full-evaluation`의 동일 탐지 결과를 사용했습니다.
그룹별 임계값은 모두 0.05, 탐지 confidence는 0.01, NMS IoU는 0.6, 장치는 CPU입니다.
새 학습·추론은 실행하지 않았으며, 속도는 top-k별로 새로 측정하지 않았습니다.
[집계 CSV](results/comparison.csv)와 [설정·원본 탐지 SHA-256](results/comparison.json)으로
수치와 비교 조건을 확인할 수 있습니다.

저장소에 추가된 [새 실행 결과](comparison-new-run/README.md)에서도 동일한 292장에 대해
전체·클래스별 포함률과 후보 개수 집계가 같았습니다. 기존 실행의 평균 지연은
323.93ms, 새 실행은 651.21ms였지만, 각각 별도 실행에서 측정된 모델 추론 시간이며
top-k 후보 선택만의 비용을 분리한 실험이 아닙니다. 이 차이를 top-2의 속도 저하로
해석하지 않습니다. 이번 문서 작성 과정에서는 학습·추론을 추가 실행하지 않았습니다.

## 실험 범위

현재 프로젝트는 COCO 사전학습 가중치로 추론하는 평가 프로젝트입니다.
여기서 top-k는 학습 파라미터가 아니라 **이미지별 MVP 그룹 후보 수**입니다.
따라서 재학습 없이 같은 탐지 결과로 비교합니다. 실제 fine-tuning을 하려면
별도 학습 데이터·정답·학습 설정이 필요하며, 아래 명령은 학습 명령이 아닙니다.

## 판정 정의

- laptop/tv를 computer로 통합하고 computer/book/other 각각의 최대 confidence를 사용합니다.
- 그룹별 기존 임계값을 통과한 그룹을 점수 내림차순으로 정렬합니다.
- Top-1은 첫 그룹, top-2는 최대 두 그룹을 후보로 반환합니다.
- laptop과 tv 또는 같은 클래스의 여러 박스가 두 후보 자리를 차지하지 않습니다.
- 동점은 computer → book → other 순서입니다.
- 통과한 그룹이 없으면 후보는 `[other]`, 하나면 그 하나만 사용합니다. 후보를 억지로 채우지 않습니다.
- 정답이 후보에 있으면 hit입니다. other도 후보로 경쟁합니다.

Top-2 포함률은 단일 예측 정확도와 다릅니다. 후보가 늘면 포함률이 상승할 수 있으므로
모델 자체가 개선됐다는 뜻은 아닙니다. 기존 confusion matrix, precision/recall/F1,
수락·거절 오류율과 정확도 기준 판정은 계속 top-1 기준입니다.
두 후보 중 어떤 것을 실제 수락할지는 별도의 운영 정책이 필요합니다.

## 실행 명령 (저장소 루트, PowerShell)

기존 전체 평가 결과로 비교만 생성합니다. 학습·추론 없이 CPU에서 집계·그래프만 생성합니다.
출력 폴더가 이미 있으면 덮어쓰지 않으므로 새 이름을 지정하세요.

```powershell
.\.venv\Scripts\python.exe scripts/compare_topk.py --run outputs/coco_model_suite/mvp-full-evaluation --output docs/rtmdet-topk/comparison-new
```

RTMDet-tiny만 새 데이터로 추론하고 최대 두 후보를 저장하려면:

```powershell
.\.venv\Scripts\python.exe scripts/run_inference.py --models rtmdet-tiny --top-k 2 --output-name rtmdet-top2
.\.venv\Scripts\python.exe scripts/compare_topk.py --run outputs/coco_model_suite/rtmdet-top2 --output docs/rtmdet-topk/comparison-new-run
```

데이터 경로는 `--source "G:/My Drive/side_project/sample"`로 지정할 수 있습니다.
모델 환경·가중치는 기존 `config/inference.yaml`을 사용합니다. top-1을 따로 추론할
필요는 없습니다. 비교 스크립트는 같은 이미지, 가중치, 탐지, 임계값으로 두 값을 계산합니다.
`--limit-per-category`로 실행한 결과는 SMOKE_ONLY로 표시됩니다.

## Git에서 결과 보기

[저장된 전체 평가의 비교 결과](results/README.md)

각 비교 폴더에는 `README.md`, `topk-comparison.png`, `comparison.csv`,
`comparison.json`이 생성됩니다. Markdown에서 PNG를 상대 경로로 표시하므로 GitHub에서
바로 볼 수 있습니다. 원본 이미지 경로 대신 집계와 탐지 파일 SHA-256을 기록합니다.
`outputs/`는 Git 제외 대상이고, `docs/`의 비교 결과는 Git에 추가할 수 있습니다.

```powershell
git add README.md config/inference.yaml scripts/run_inference.py scripts/compare_topk.py src/yolo_benchmark/reports.py tests/test_reports.py docs/rtmdet-topk
```

검증 (모델 실행 없음):

```powershell
.\.venv\Scripts\python.exe tests/test_reports.py
```
