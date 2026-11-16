import sys, os, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tmp/review_deps'))
PROJECT = ROOT / 'Dipu_Bhai_Thesis'
OUT = ROOT / 'tmp/prof_review'
OUT.mkdir(parents=True, exist_ok=True)

if sys.argv[1] == 'pdf':
    import pymupdf
    doc = pymupdf.open(PROJECT / 'prof/RAFT_presentation-1.pdf')
    texts = []
    for i, page in enumerate(doc):
        text = f'PAGE {i+1}\n{page.get_text()}'
        texts.append(text)
        page.get_pixmap(matrix=pymupdf.Matrix(1.2, 1.2)).save(OUT / f'slide_{i+1:02}.png')
    (OUT / 'presentation.txt').write_text('\n\n'.join(texts), encoding='utf-8')
    print('\n\n'.join(texts))
elif sys.argv[1] == 'audio':
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
    from faster_whisper import WhisperModel
    import av
    source = next((PROJECT / 'prof').glob('*.mp4'))
    with av.open(str(source)) as c:
        print('DURATION_SECONDS', c.duration / av.time_base, flush=True)
    model = WhisperModel('small', device='cpu', compute_type='int8', cpu_threads=6, download_root=str(OUT / 'models'))
    segments, info = model.transcribe(str(source), beam_size=5, vad_filter=True)
    print('TRANSCRIBING', info.duration, info.language, info.language_probability, flush=True)
    with (OUT / 'transcript.jsonl').open('w', encoding='utf-8') as f:
        for seg in segments:
            item = {'start':seg.start, 'end':seg.end, 'text':seg.text}
            f.write(json.dumps(item) + '\n')
            f.flush()
            print(f'[{seg.start/60:.2f}m] {seg.text}', flush=True)
