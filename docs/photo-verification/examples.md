[평가 이미지 Google Drive 폴더](https://drive.google.com/drive/folders/1H-wfZyZIEnoOGA3Mr6yhfgsb0a2cdCXR)의 실제 추론 이미지입니다.
Drive 동기화 파일의 SHA-256을 평가 당시 목록과 대조해 동일 파일임을 확인하고 원본 그대로 복사했습니다.
성공·복구·오수락 위험을 설명하기 위해 선택한 예시이며 무작위 표본이나 전체 성능의 근거가 아닙니다.

| 예시 | 실제 폴더 정답 | Top-1 | Top-2 후보 |
|---|---|---|---|
| computer 정답<br><img src="docs/photo-verification/examples/computer-correct.jpg" width="240" alt="computer 정답"> | computer (computer) | computer | computer, other |
| book: 2순위로 정답 복구<br><img src="docs/photo-verification/examples/book-recovered.jpg" width="240" alt="book: 2순위로 정답 복구"> | book (book) | other | other, book |
| 다른 클래스: book 오수락 위험<br><img src="docs/photo-verification/examples/book-false-accept.jpg" width="240" alt="다른 클래스: book 오수락 위험"> | other (bicycle) | other | other, book |
| other 정답<br><img src="docs/photo-verification/examples/other-correct.jpg" width="240" alt="other 정답"> | other (bicycle) | other | other |

사진 속 모든 객체가 주석 처리된 데이터는 아닙니다. 다른 폴더의 사진에 book이 실제로 함께 있다면
모델 탐지와 폴더 정답이 다를 수 있으므로, 아래 오수락은 폴더 정답 기준으로 해석합니다.
이미지 출처 파일명·해시·그룹 점수는 [예시 기록](docs/photo-verification/examples.json)에 보존했습니다.