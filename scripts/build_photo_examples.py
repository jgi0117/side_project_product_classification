"""Copy selected evaluated Drive images unchanged, with cached predictions."""
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from yolo_benchmark.reports import decide

run = ROOT / "outputs/coco_model_suite/mvp-full-evaluation"
snapshot = json.loads((run / "run.json").read_text(encoding="utf-8"))
raw = json.loads((run / "rtmdet-tiny/detections.json").read_text(encoding="utf-8"))
expected = {r["path"]: r["sha256"] for r in snapshot["images"]}
cases = [
    ("computer-correct", "computer 정답", lambda r, d: r["source_label"] == "computer" and d["prediction"] == "computer"),
    ("book-recovered", "book: 2순위로 정답 복구", lambda r, d: r["source_label"] == "book" and d["prediction"] != "book" and "book" in d["candidates"]),
    ("book-false-accept", "다른 클래스: book 오수락 위험", lambda r, d: r["source_label"] not in ["book", "computer"] and d["prediction"] != "book" and "book" in d["candidates"]),
    ("other-correct", "other 정답", lambda r, d: r["source_label"] not in ["book", "computer"] and d["candidates"] == ["other"]),
]
out = ROOT / "docs/photo-verification"
(out / "examples").mkdir(exist_ok=True)
records = []
lines = ["[평가 이미지 Google Drive 폴더](https://drive.google.com/drive/folders/1H-wfZyZIEnoOGA3Mr6yhfgsb0a2cdCXR)의 실제 추론 이미지입니다.",
         "Drive 동기화 파일의 SHA-256을 평가 당시 목록과 대조해 동일 파일임을 확인하고 원본 그대로 복사했습니다.",
         "성공·복구·오수락 위험을 설명하기 위해 선택한 예시이며 무작위 표본이나 전체 성능의 근거가 아닙니다.", "",
         "| 예시 | 실제 폴더 정답 | Top-1 | Top-2 후보 |",
         "|---|---|---|---|"]
for name, title, predicate in cases:
    for item in raw["images"]:
        d = decide(item["detections"], {**snapshot["config"], "top_k": 2})
        if not predicate(item, d):
            continue
        source = Path(item["path"])
        content = source.read_bytes()
        sha = hashlib.sha256(content).hexdigest()
        if sha != expected[item["path"]]:
            raise ValueError(f"Changed image: {source.name}")
        target = out / "examples" / (name + source.suffix.lower())
        shutil.copyfile(source, target)
        truth = item["source_label"] if item["source_label"] in ["computer", "book"] else "other"
        rel = target.relative_to(ROOT).as_posix()
        lines.append(f"| {title}<br><img src=\"{rel}\" width=\"240\" alt=\"{title}\"> | {truth} ({item['source_label']}) | {d['prediction']} | {', '.join(d['candidates'])} |")
        records.append({"example": name, "source_filename": source.name, "source_label": item["source_label"],
                        "sha256": sha, "top1": d["prediction"], "top2": d["candidates"], "group_scores": d["scores"]})
        break
    else:
        raise ValueError(f"No matching example: {name}")
lines += ["", "사진 속 모든 객체가 주석 처리된 데이터는 아닙니다. 다른 폴더의 사진에 book이 실제로 함께 있다면",
          "모델 탐지와 폴더 정답이 다를 수 있으므로, 아래 오수락은 폴더 정답 기준으로 해석합니다.",
          "이미지 출처 파일명·해시·그룹 점수는 [예시 기록](docs/photo-verification/examples.json)에 보존했습니다."]
(out / "examples.md").write_text("\n".join(lines), encoding="utf-8")
(out / "examples.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(records, ensure_ascii=False, indent=2))
