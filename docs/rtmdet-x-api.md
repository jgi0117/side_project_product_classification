# RTMDet-x Top-1 FastAPI

이번 스프린트의 추론 모델은 **COCO 사전학습 RTMDet-x**이며, 결과는 `computer`, `book`, `other` 중 하나입니다. 실험에서 사용한 입력 640, 탐지 confidence 0.01, NMS IoU 0.6, 그룹 임계값 0.05와 동일한 판정 로직을 사용합니다. 모델은 서버 시작 시 한 번만 로드하고 요청마다 재사용합니다.

## 준비와 실행

저장소 루트에서 Python 3.11 RTMDet 환경을 사용합니다. 기존 `.model_envs/rtmdet` 환경에는 PyTorch, MMCV, MMDetection이 설치돼 있습니다. 새 환경을 구성한다면 [RTMDet 의존성](../requirements-rtmdet.txt)을 참고하고, PyTorch 버전에 맞는 MMCV wheel을 설치하세요. API 패키지는 다음 명령으로 설치합니다.

```powershell
uv pip install --python .model_envs/rtmdet/Scripts/python.exe -r requirements-api.txt
```

체크포인트는 Git에 포함되지 않습니다. `models/pretrained/rtmdet_x.pth`가 없다면 공식 파일을 받으세요. 서버는 시작할 때 이 파일의 SHA-256을 확인합니다.

```powershell
New-Item -ItemType Directory -Force models/pretrained | Out-Null
curl.exe --fail --location --retry 3 --output models/pretrained/rtmdet_x.pth https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_x_8xb32-300e_coco/rtmdet_x_8xb32-300e_coco_20220715_230555-cc79b9ae.pth
$env:RTMDET_DEVICE = 'cpu'
$env:RTMDET_API_KEY = 'replace-with-a-shared-secret'
.model_envs/rtmdet/Scripts/python.exe -m uvicorn yolo_benchmark.rtmdet_api:app --app-dir src --host 0.0.0.0 --port 8000 --workers 1
```

`RTMDET_DEVICE`는 CUDA 사용 시 `cuda:0`으로 지정할 수 있습니다. `RTMDET_X_WEIGHTS`로 체크포인트 경로를 바꿀 수 있지만 동일한 RTMDet-x 체크포인트 SHA-256이어야 합니다. `RTMDET_API_KEY`를 설정하면 `/predict`에 `X-API-Key` 헤더가 필요합니다. 다른 머신의 백엔드는 이 서버의 접근 가능한 IP/도메인과 포트로 호출해야 합니다. `0.0.0.0` 바인딩은 네트워크 수신을 허용하지만 방화벽이나 배포 환경의 라우팅까지 설정하지는 않습니다.

## 요청과 응답

`POST /predict`는 `multipart/form-data`의 `file` 필드로 이미지 한 장을 받습니다. 디코딩 가능한 이미지 최대 10 MiB를 허용하며, 추론 후 원본 이미지를 영구 저장하지 않습니다.

```powershell
curl.exe -X POST http://SERVER_IP:8000/predict -H "X-API-Key: replace-with-a-shared-secret" -F "file=@C:/path/to/image.jpg"
```

응답 JSON에는 `model: "rtmdet-x"`, `top_k: 1`, `prediction`, `confidence`, `group_scores`, `decision_reason`이 있습니다. `prediction`은 `computer`·`book`·`other` 중 하나입니다. `confidence`는 선택된 그룹의 최고 탐지 점수이며 보정된 분류 확률은 아닙니다. 임계값을 통과한 탐지가 없으면 `other`로 반환하고 `decision_reason`은 `no_passing_detection`입니다.

백엔드 Python 호출 예시:

```python
import httpx

with open("image.jpg", "rb") as image:
    response = httpx.post(
        "http://SERVER_IP:8000/predict",
        headers={"X-API-Key": "replace-with-a-shared-secret"},
        files={"file": ("image.jpg", image, "image/jpeg")},
        timeout=30.0,
    )
response.raise_for_status()
prediction = response.json()["prediction"]
```

`GET /health`는 모델 로드가 끝난 서버에서 `{"status":"ok","model":"rtmdet-x","top_k":1}`을 반환합니다. 자동 생성 API 문서는 `/docs`에서 볼 수 있습니다. 잘못된 이미지에는 400, 키 오류에는 401, 10 MiB 초과에는 413, 파일 필드 누락에는 422를 반환합니다.
