# RTMDet 모델 크기 비교 실행

`--top-k 2`로 추론을 한 번 실행하면 비교 스크립트가 동일한 탐지 결과에서 **k=1과 k=2를 모두 계산**합니다. 결과 표와 CSV에는 모델마다 두 행이 생성됩니다. k=1 정답 포함률은 단일 예측 정확도이고, k=2 정답 포함률은 두 후보 중 정답이 있는 비율입니다.

COCO 사전학습 RTMDet tiny/s/m/l/x를 같은 이미지 목록에서 추론합니다. 각 모델은 해당 크기의 MMDetection 설정과 체크포인트를 사용합니다. `run_inference.py`가 모델별 `metrics.json` 및 전체 `summary.csv`를 만들고, `compare_model_sizes.py`가 top-1 정확도, top-2 정답 포함률, 클래스별 재현율, 이미지 단위 오류율, 속도, 체크포인트 크기를 한 표로 모읍니다.

## 1. 체크포인트 준비

아래 PowerShell 명령은 설치된 MMDetection 3.3.0의 RTMDet `metafile.yml`에 등록된 공식 체크포인트 네 개를 받습니다. 기존 tiny 체크포인트는 그대로 사용합니다. 다운로드가 끝난 파일만 `.pth`로 옮깁니다.

```powershell
$weights = @{
  s = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_s_8xb32-300e_coco/rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth'
  m = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_m_8xb32-300e_coco/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth'
  l = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_l_8xb32-300e_coco/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth'
  x = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_x_8xb32-300e_coco/rtmdet_x_8xb32-300e_coco_20220715_230555-cc79b9ae.pth'
}
$hashPrefixes = @{ s = '387A891E'; m = '229F527C'; l = '5A0BE7C4'; x = 'CC79B9AE' }
foreach ($size in 's','m','l','x') {
  $target = "models/pretrained/rtmdet_$size.pth"
  if (Test-Path $target) { continue }
  $partial = "$target.part"
  curl.exe --fail --location --retry 3 --output $partial $weights[$size]
  if ($LASTEXITCODE -ne 0) { throw "다운로드 실패: rtmdet_$size" }
  $hash = (Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash
  if (!$hash.StartsWith($hashPrefixes[$size])) { throw "체크포인트 해시 불일치: rtmdet_$size" }
  Move-Item -LiteralPath $partial -Destination $target
}
```

기존 `.venv`와 `.model_envs/rtmdet` 환경을 사용합니다. 새 환경에서는 `requirements.txt`와 `requirements-rtmdet.txt`를 각각 설치하고 PyTorch 버전에 맞는 MMCV wheel을 준비해야 합니다.

## 2. 실행

저장소 루트에서 실행합니다. Google Drive가 다른 위치에 마운트되어 있으면 `--source`의 값을 바꾸세요. `--device cuda:0`은 CUDA가 준비된 경우에만 사용하세요. 기본값은 CPU입니다.

```powershell
.venv/Scripts/python.exe scripts/run_inference.py --source 'G:/내 드라이브/side_project/sample' --top-k 2 --output-name model-size-20260927
.venv/Scripts/python.exe scripts/compare_model_sizes.py --run outputs/rtmdet/model-size-20260927
```

먼저 작은 샘플로 동작을 확인하려면 별도 실행 이름을 사용하세요.

```powershell
.venv/Scripts/python.exe scripts/run_inference.py --source 'G:/내 드라이브/side_project/sample' --models rtmdet-tiny rtmdet-s --top-k 2 --limit-per-category 1 --output-name model-size-smoke
.venv/Scripts/python.exe scripts/compare_model_sizes.py --run outputs/rtmdet/model-size-smoke
```

결과는 `outputs/rtmdet/<실행 이름>/model-size-comparison/README.md`, `comparison.csv`, `comparison.json`에 저장됩니다. 실행 폴더 이름은 매번 새로 지정하세요. `--limit-per-category` 결과는 `SMOKE_ONLY`로 표시됩니다. 체크포인트가 빠진 모델은 추론 시작 전에 오류가 나며, `--models`로 준비된 크기만 선택할 수 있습니다.

이는 폴더당 하나의 정답 라벨을 사용하는 이미지 분류 비교입니다. 박스 mAP나 세션 오류율을 측정하지 않습니다. top-2는 정답이 두 후보 안에 있는 비율이며 top-1 정확도와 구분해 해석해야 합니다.
