"""Build a gallery of evaluated images, without prediction annotations."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
run = ROOT / 'outputs/coco_model_suite/mvp-full-evaluation'
snapshot = json.loads((run / 'run.json').read_text(encoding='utf-8'))
out = ROOT / 'docs/photo-verification'
(out / 'examples').mkdir(exist_ok=True)
records = []
lines = ['[Google Drive 평가 이미지](https://drive.google.com/drive/folders/1H-wfZyZIEnoOGA3Mr6yhfgsb0a2cdCXR) 중 실제 추론에 사용한 사진을 클래스별 10장씩 소개합니다.', '']
for label in ('book', 'computer', 'other'):
    if label == 'other':
        selected = []
        for source_label in ('bicycle', 'guitar'):
            pool = sorted((r for r in snapshot['images'] if r['source_label'] == source_label), key=lambda r: r['sha256'])
            if len(pool) < 5:
                raise ValueError(f'Need five images for {source_label}')
            selected.extend(pool[:5])
    else:
        selected = sorted((r for r in snapshot['images'] if r['source_label'] == label), key=lambda r: r['sha256'])[:10]
    if len(selected) != 10:
        raise ValueError(f'Need ten images for {label}')
    lines += [f'### {label} — 10장', '', '| 1 | 2 | 3 | 4 | 5 |', '|---|---|---|---|---|']
    cells = []
    for index, item in enumerate(selected, 1):
        source = Path(item['path'])
        sha = hashlib.sha256(source.read_bytes()).hexdigest()
        if sha != item['sha256']:
            raise ValueError(f'Changed image: {source.name}')
        target = out / 'examples' / f'{label}-{index:02d}{source.suffix.lower()}'
        shutil.copyfile(source, target)
        rel = target.relative_to(ROOT).as_posix()
        cells.append(f'<img src="{rel}" width="180" alt="{label} 예시 {index}">')
        records.append({'category': label, 'example': index, 'file': target.name,
                        'source_filename': source.name, 'source_label': item['source_label'], 'sha256': sha})
    for start in (0, 5):
        lines.append('| ' + ' | '.join(cells[start:start + 5]) + ' |')
    lines.append('')
(out / 'examples.md').write_text('\n'.join(lines), encoding='utf-8')
(out / 'examples.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
print('Gallery created: book 10, computer 10, other 10 (bicycle 5 + guitar 5).')
