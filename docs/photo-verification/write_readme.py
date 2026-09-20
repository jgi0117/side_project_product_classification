"""Render the root presentation README from published aggregate metrics."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
data = json.loads((HERE / "metrics.json").read_text(encoding="utf-8"))
pct = lambda x: f"{x:.2%}"
model_rows = []
for m in data["top1"]:
    b = [b for b in data["category_verification"] if b["model"] == m["model"] and b["top_k"] == 1]
    model_rows.append(f"| {m['model']} | {pct(m['accuracy'])} | {pct(m['macro']['recall'])} | {pct(m['macro']['precision'])} | {pct(b[0]['fpr'])} | {pct(b[1]['fpr'])} |")
rt_rows = []
for label in ["computer", "book"]:
    for b in data["category_verification"]:
        if b["model"] == "rtmdet-tiny" and b["class"] == label:
            rt_rows.append(f"| {label} | {b['top_k']} | {pct(b['accuracy'])} | {pct(b['recall'])} | {pct(b['precision'])} | {pct(b['fpr'])} ({b['fp']}/{b['fp']+b['tn']}) | {pct(b['fnr'])} |")
text = r"""# F-03. 작업환경 행동 요구 — 사진 인증 AI 평가

## 1. 목적과 AI 엔지니어 담당 범위

사진 인증은 사용자가 **등록한 작업도구가 있는 환경으로 이동하도록 준비 행동을 요구하는 개입**입니다.
**실제 작업 시작을 증명하는 기능이 아닙니다.** 사진에서 도구가 보이는지 판정할 수 있어도,
사용자의 이동 여부·작업 시작·지속 여부까지 사진 한 장으로 확인하지는 못합니다.

AI 담당 범위는 제출된 사진과 등록 카테고리의 일치 여부를 평가하고, 판정 기준과 오류 특성을
서비스 팀에 제공하는 것입니다. 현재 저장소는 이를 위한 **오프라인 사전학습 모델 평가 코드**이며,
카메라 UI·인증 상태 저장·재시도 서비스가 구현됐다는 의미는 아닙니다.

| 사용자 흐름 | 서비스 동작 / AI 연동 요구 |
|---|---|
| 카메라 권한 허용 → 등록 작업도구 촬영 | 권한·촬영 UI는 앱 담당 |
| 제출 전 재촬영 → 사진 제출 | 앱/서버가 사진과 등록 카테고리를 전달 |
| 판정 결과 확인 | 성공 / 판정거절 / 기술오류를 구분해 표시 |
| 판정거절 후 재촬영·재제출 | 인증 가능 시간이 남아 있으면 허용 |
| 나중에 인증 화면에 재진입 | 인증 가능 시간이 남아 있으면 재시도 허용 |
| 성공 | 서버가 인증 완료 상태 기록 |

AI 연동에서 등록 카테고리에 맞는 도구로 판정하면 성공, 정상 처리됐지만 조건을 충족하지
못하면 판정거절로 다룹니다. 디코딩 실패·추론 실패 같은 기술오류는 판정거절 및 사용자
미인증과 분리해야 합니다. 인증 가능 종료 시점은 서비스 운영 정책에서 정합니다.

MVP에서는 목적 이해, 실제 도구 환경으로의 이동, 인증 부담, 인증 후 작업 시작,
인증 후 이탈, 판정 오류로 인한 시작 방해를 확인해야 합니다. 아래 AI 평가는 이 중
**판정 오류 위험**을 다루며, 나머지는 사용자 관찰과 서비스 로그로 확인해야 합니다.

## 2. 평가 조건과 MVP 목표

COCO 사전학습 경량 탐지 모델 7개를 top-1으로 비교한 뒤 RTMDet-tiny의 top-2를 평가했습니다.
모두 이름이 tiny인 모델은 아니며 nano/small 모델도 포함합니다. 별도 fine-tuning은 없었습니다.
이번 문서는 저장된 탐지 결과를 재집계했으며 학습·추론을 새로 실행하지 않았습니다.

| 항목 | 조건 |
|---|---|
| 평가 데이터 | 292장: computer 200, book 19, other 73 (bicycle 31 + guitar 42) |
| 정답 | 최상위 폴더의 단일 주 객체 라벨 |
| 등록 카테고리 | computer, book; other는 비대상 평가 그룹 |
| 그룹 매핑 | COCO laptop/tv → computer, book → book, 나머지 → other |
| 후보 점수 | 그룹 내 최대 confidence; 합산·재정규화하지 않음 |
| 임계값 | 그룹별 0.05, 탐지 confidence 0.01, 설정 NMS IoU 0.6 |
| 실행 환경 | CPU, 모델별 기본 입력 크기·후처리 사용 |
| 평가 범위 | 이미지 분류·등록 카테고리 판정; 바운딩 박스 mAP 아님 |

NanoDet/PicoDet는 native head/export 후처리를 사용하며 최소 점수는 각각 0.05/0.025입니다.
모델별 입력 크기와 후처리가 달라 동일 구조·동일 연산량의 비교는 아닙니다.

아래 수치는 저장소의 `report_criteria`에 있는 MVP 목표입니다. F-03 개요 자체에는 수치 목표가 없습니다.

| 기준 | 목표 | 현재 판단 범위 |
|---|---:|---|
| 전체 단일 예측 Accuracy | 목표 ≥90%, 최소 ≥80% | Top-1 이미지 평가로 확인 |
| 세션 수락 오류율 | ≤5% | 세션 정답·재시도 정보가 없어 N/A |
| 세션 거절 오류율 | ≤20% | 세션 정답·재시도 정보가 없어 N/A |
| Precision / Recall | 별도 목표 미정 | 클래스별 오류와 선택 기준으로 보고 |

**이미지 FPR과 세션 수락 오류율은 다릅니다.** 아래 이미지 지표만으로 세션 목표 달성을 선언하지 않습니다.

### 지표 읽는 법

| 지표 | 쉽게 말하면 | 사진 인증에서 쓰는 이유 |
|---|---|---|
| Accuracy (Acc) | 전체 판단 중 맞은 비율 | 전반적인 판별 성능과 MVP 최소/목표 수준 확인 |
| Recall | 맞는 도구 사진을 얼마나 통과시키는가 | 불필요한 재촬영과 시작 행동 방해를 줄이기 위해 |
| Precision | 통과한 사진을 얼마나 믿을 수 있는가 | 엉뚱한 도구 사진으로 인증되는 위험 확인 |
| FPR (오수락률) | 다른 클래스 사진이 얼마나 잘못 통과하는가 | 등록 작업도구와 무관한 사진의 수락 위험 확인 |
| FNR (오거절률) | 맞는 도구 사진이 얼마나 거절되는가 | 정상 사용자의 인증 실패 부담 확인 |
| Macro 평균 / F1 | 클래스 동등 평균 / precision·recall의 조화평균 | 데이터가 많은 computer가 book의 약점을 가리지 않도록 비교 |
| 혼동행렬 | 어떤 정답을 어떤 예측으로 착각했는가 | 오류 원인과 개선할 카테고리를 구체적으로 확인 |

TP는 맞는 도구 수락, FP는 다른 도구 수락, FN은 맞는 도구 거절, TN은 다른 도구 거절입니다.

- **Top-1 Acc**: 3개 그룹 중 단일 예측이 정답인 비율.
- **Recall** = TP/(TP+FN): 해당 등록 클래스 사진을 수락한 비율.
- **Precision** = TP/(TP+FP): 해당 클래스로 수락한 사진 중 실제 그 클래스인 비율.
- **오수락률 FPR** = FP/(FP+TN): **다른 클래스 사진인데 등록 클래스가 맞다고 판정한 비율**.
  computer 등록의 음성은 book+other 92장, book 등록의 음성은 computer+other 273장입니다.
- **오거절률 FNR** = FN/(TP+FN) = 1−Recall.
- **등록 카테고리 Acc** = (TP+TN)/292: 해당 카테고리에 대한 수락/거절 이진 정확도.
- **F1** = 2TP/(2TP+FP+FN): precision과 recall을 함께 보는 보조 지표.

FPR은 1−Precision과 분모가 다릅니다. 모델 비교 표의 Recall/Precision은
computer/book/other 3개 클래스의 **macro 평균**이며, 등록 카테고리별 수치는 별도 표로 제시합니다.

## 3. 1차 평가 — 사전학습 모델 Top-1 비교

| 모델 | Acc ↑ | Macro Recall ↑ | Macro Precision ↑ | computer 등록 FPR ↓ | book 등록 FPR ↓ |
|---|---:|---:|---:|---:|---:|
MODEL_ROWS

![모델별 Top-1 성능과 오수락률](docs/photo-verification/top1-models.png)

**정확도 1위는 PicoDet-s-320(91.78%)입니다.** RTMDet-tiny(90.07%)와 함께 목표 90%를
넘었으며 YOLOv8n/s는 최소 80%에 미달했습니다. 따라서 RTMDet를 “정확도 최고 모델”이라고
설명하는 것은 맞지 않습니다.

RTMDet-tiny는 **3개 클래스 macro recall 84.05%, F1 84.52%로 1위**입니다.
Macro precision은 RTMDet 88.31%이며, 1위는 YOLOX-tiny(89.12%)입니다.
전체의 68.49%가 computer인 데이터에서 클래스 균형 지표가 강점인 후속 평가 대상입니다.
Top-2는 RTMDet-tiny에 대해서만 수행했으므로 다른 모델의 top-2보다 우수한지는 알 수 없습니다.

### Top-1 혼동행렬 — 모델 7개

행은 실제 클래스, 열은 단일 예측입니다. 칸마다 **건수와 실제 클래스 내 비율**을 표시했습니다.
대각선은 정답, 비대각선은 오판입니다. 예를 들어 RTMDet의 book → other 오판은
book 촬영 후 인증이 거절될 수 있는 위험을 보여줍니다.

![7개 모델 Top-1 혼동행렬](docs/photo-verification/top1-confusion.png)

## 4. 2차 평가 — RTMDet-tiny Top-1 → Top-2

동일한 RTMDet 탐지 결과에서 임계값을 통과한 **그룹**을 점수순으로 최대 두 개 허용했습니다.
동점은 computer → book → other 순서이며, 후보가 하나면 하나만 유지합니다.
통과한 탐지가 없으면 `[other]`입니다. 이는 학습 설정 변경이 아닌 후처리 정책 비교입니다.

| 정답 그룹 | 이미지 수 | Top-1 정답률 | Top-2 정답 포함률 | 추가 정답 |
|---|---:|---:|---:|---:|
| 전체 | 292 | 90.07% (263/292) | 97.95% (286/292) | 23 |
| computer | 200 | 89.00% | 97.50% | 17 |
| book | 19 | 63.16% | 94.74% | 6 |
| other | 73 | 100.00% | 100.00% | 0 |

![RTMDet 후보 정답 포함률](docs/rtmdet-topk/results/topk-comparison.png)

**97.95%는 단일 판정 정확도가 아니라 정답 포함률입니다.** 후보가 두 개인 사진은 269장입니다.
정답이 포함돼도 잘못된 클래스가 함께 포함될 수 있어 이 수치만으로 인증 성능을 판단하면 안 됩니다.

### 등록 카테고리가 후보에 있으면 수락할 경우

각 사진을 computer 등록과 book 등록에 대해 각각 검사하는 **오프라인 정책 시뮬레이션**입니다.
실제 사용자 등록 카테고리·시도 로그를 측정한 결과는 아닙니다. 같은 수락 규칙을 top-1과 top-2에
적용해 비교했으며, 실제 서비스에 이 정책을 배포한 것은 아닙니다.

| 등록 카테고리 | k | 수락/거절 Acc ↑ | Recall ↑ | Precision ↑ | FPR ↓ (FP/음성 수) | FNR ↓ |
|---|---:|---:|---:|---:|---:|---:|
RT_ROWS

![RTMDet 등록 카테고리별 성능 변화](docs/photo-verification/rtmdet-tradeoff.png)

Computer는 정답 수락이 178→195장으로 증가하지만 오수락도 2→16장으로 늘었습니다.
Book은 정답 수락이 12→18장으로 늘어난 반면 오수락이 1→55장으로 증가했습니다.
특히 **book precision은 92.31%→24.66%, FPR은 0.37%→20.15%**로 악화됩니다.
Top-2는 올바른 도구 사진의 거절을 줄이지만, 다른 도구 사진의 수락을 크게 늘립니다.

### 등록 카테고리별 혼동행렬

Top-2에는 단일 예측이 없으므로 3×3 분류 혼동행렬을 만들지 않습니다.
아래 2×2 행렬에서 행은 등록 클래스 일치/불일치, 열은 수락/거절입니다.
왼쪽 아래가 **다른 클래스인데 수락한 FP**, 오른쪽 위가 **올바른 클래스인데 거절한 FN**입니다.

![RTMDet Top-1 및 Top-2 수락 거절 혼동행렬](docs/photo-verification/rtmdet-confusion.png)

## 5. MVP 판단과 다음 검증

| 질문 | 이번 결과로 말할 수 있는 것 |
|---|---|
| Top-1이 목표 Acc 90%를 만족하는가? | 현재 데이터에서 PicoDet와 RTMDet가 만족 |
| Top-2가 올바른 사진의 거절을 줄이는가? | RTMDet computer/book recall은 모두 상승 |
| Top-2를 그대로 자동 수락에 적용해도 되는가? | 등록 카테고리 FPR 증가가 커서 현재 결과로 채택을 권하기 어려움 |
| 세션 오류 목표를 만족하는가? | N/A — 사용자 등록·반복 시도·기술오류를 포함한 세션 평가 필요 |
| 사진 인증이 실제 작업 시작을 유도하는가? | 모델 평가로 알 수 없음 — 행동 지표와 사용자 조사 필요 |

후속 검증은 카테고리별 임계값/2순위 수락 조건을 검증용 데이터에서 조정하고,
분리된 테스트 데이터로 recall과 FPR을 함께 확인하는 순서가 적절합니다.
Book은 19장뿐이므로 음성 사진과 함께 데이터를 늘려야 합니다. 폴더 정답은 사진 속 모든 객체를
표시한 것이 아니어서 실제 book/computer가 함께 있는 사진은 오수락으로 과대 집계될 수 있습니다.
복수 객체 주석과 실제 등록 카테고리·세션 데이터로 최종 판단해야 합니다.

## 6. 모델 라이선스와 상업 이용

공식 저장소·공급자 문서를 2026-09-20에 확인했습니다. 아래는 **사용한 구현의 라이선스**입니다.

| 평가 모델 | 공식 구현 라이선스 | 상업 서비스 사용 조건 / 출처 |
|---|---|---|
| YOLOv8n | AGPL-3.0 / Enterprise | AGPL 의무 준수 시 상업 이용 가능. 비공개 제품 통합은 공급자의 Enterprise 계약 경로 확인. [공식 안내](https://www.ultralytics.com/license) |
| YOLOv8s | AGPL-3.0 / Enterprise | YOLOv8n과 동일. 공급자는 학습 코드와 생성 모델에도 적용한다고 안내. [공식 안내](https://www.ultralytics.com/license) |
| RTMDet-tiny (MMDetection) | Apache-2.0 | 상업 이용·비공개 제품 통합 가능, 재배포 시 고지 등 준수. [LICENSE](https://github.com/open-mmlab/mmdetection/blob/main/LICENSE) |
| NanoDet-Plus-m-320 | Apache-2.0 | 상업 이용 가능, 재배포 조건 준수. [LICENSE](https://github.com/RangiLyu/nanodet/blob/main/LICENSE) |
| PicoDet-s-320 (PaddleDetection) | Apache-2.0 | 상업 이용 가능, 재배포 조건 준수. [LICENSE](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/LICENSE) |
| YOLOX-nano | Apache-2.0 | 상업 이용 가능, 재배포 조건 준수. [LICENSE](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/LICENSE) |
| YOLOX-tiny | Apache-2.0 | 상업 이용 가능, 재배포 조건 준수. [LICENSE](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/LICENSE) |

Apache-2.0 재배포 시 라이선스 사본·관련 저작권/NOTICE 고지 보존, 수정 파일 표시 등의 조건을
따릅니다. 비공개 상업 서비스의 우선 검토 대상은 RTMDet/PicoDet/NanoDet/YOLOX입니다.
AGPL은 “상업 이용 금지”가 아니지만 소스 제공 등 의무가 있어 비공개 서비스 요구와 함께 검토해야 합니다.

**코드 라이선스와 가중치·학습/평가 이미지 권리는 구분합니다.** 현재 로컬 가중치의 다운로드
출처·버전·개별 약관을 모두 증빙한 상태는 아니므로 위 표를 전체 배포물의 권리 검토 완료로
해석하지 않습니다. 출시 시 선택 체크포인트의 출처·조건과 의존성 고지를 고정해야 합니다.
평가용 Drive 이미지 역시 모델의 Apache 라이선스로 재사용 권리가 부여되는 것은 아닙니다.

## 7. 실제 추론 이미지 예시

EXAMPLE_SECTION

## 8. 재현과 결과 파일

- [모델별 Top-1 CSV](docs/photo-verification/top1-models.csv)
- [등록 카테고리별 TP/FP/FN/TN·지표 CSV](docs/photo-verification/category-verification.csv)
- [혼동행렬 원시 수치·설정 목표·원본 탐지 SHA-256](docs/photo-verification/metrics.json)
- [Top-k 실행 안내](docs/rtmdet-topk/README.md)

저장된 전체 평가를 재집계하고 그래프를 생성합니다. 학습·추론을 실행하지 않습니다.

```powershell
.\.venv\Scripts\python.exe scripts/build_photo_report.py
.\.venv\Scripts\python.exe docs/photo-verification/write_readme.py
```

`outputs/coco_model_suite/mvp-full-evaluation`의 원본 결과가 필요합니다. 원본 이미지·가중치·
탐지 캐시는 Git 제외 대상이며, Git에는 집계 JSON/CSV와 그래프를 포함합니다.
그래프와 수치 생성 스크립트는 지정한 출력 폴더의 동명 파일을 갱신합니다.
다른 데이터로 재실행하면 본문의 수작업 해석·목표 판정도 다시 검토해야 합니다.

RTMDet만 새로 추론하려면 (직접 실행):

```powershell
.\.venv\Scripts\python.exe scripts/run_inference.py --models rtmdet-tiny --top-k 2 --output-name rtmdet-top2-new
```
"""
(HERE.parents[1] / "README.md").write_text(text.replace("MODEL_ROWS", "\n".join(model_rows)).replace("RT_ROWS", "\n".join(rt_rows)).replace("EXAMPLE_SECTION", (HERE / "examples.md").read_text(encoding="utf-8")), encoding="utf-8")
