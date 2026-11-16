from pathlib import Path
import csv
import hashlib
import json
import nbformat

root = Path(__file__).resolve().parents[1]
project = root / 'professor-review'
nb_path = project / 'code/main_5.ipynb'
notebook = nbformat.read(nb_path, as_version=4)
notebook.cells[0].source = '''# Radiative cooling: acquisition checks to Random Forest validation

Predict recorded net heat flux from five atmospheric inputs for the Sample 2 RAFT
campaign. This executed notebook includes acquisition checks, three-minute alignment
without gap interpolation, a basic Random Forest, two validation protocols, and
20 embedded figures. Run all cells to reproduce the analysis; no separate figure
files are created.

See the [repository README](../README.md) for setup and results,
the [data dictionary](../processed/README.md) for inputs, and the
[methodology and references](../docs/methodology.md) for processing decisions and
the supervisor's instructions. Private recordings are not redistributed or required
to execute this notebook.
'''
notebook.cells[1].source = '''## 1. Configuration and provenance

The default `full_campaign` mode uses the April–July merged table. Separate instrument
exports cover only April–May; `native_exports` restricts modelling to those timestamps.
Full-campaign results depend on an earlier merge whose later source timestamps and
matching tolerance are unavailable. Both modes use the existing explicit logger-unit
metadata and inherited calibration. No sensor polarity change is guessed.

The latest guidance permits three-minute alignment and asks for a basic Random Forest.
The older presentation's five-minute MLP example is not the modelling specification
for this baseline. Input hashes and actual package versions are printed below.
'''
nbformat.validate(notebook)
nbformat.write(notebook, nb_path)

periods = {
    'merged_dataset_104957.csv': ('2026-04-17 14:53:44', '2026-07-03 07:59:16'),
    'merged_raft.csv': ('2026-04-17 14:53:44', '2026-05-15 12:02:32'),
    'merged_meteo.csv': ('2026-04-17 14:54:24', '2026-05-15 12:03:46'),
    'merged_ir.csv': ('2026-04-20 09:36:08.788', '2026-05-15 12:03:23'),
}
files = []
for filename, (first, last) in periods.items():
    path = project / 'processed' / filename
    content = path.read_bytes()
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.reader(handle)
        columns = next(reader)
        rows = sum(1 for _ in reader)
    files.append({'path': f'processed/{filename}', 'rows': rows, 'columns': columns,
                  'first_recorded_timestamp': first, 'last_recorded_timestamp': last,
                  'timezone': 'not supplied', 'bytes': len(content),
                  'sha256': hashlib.sha256(content).hexdigest()})
(project / 'data_manifest.json').write_text(json.dumps({'schema_version': 1, 'files': files}, indent=2) + '\n', encoding='utf-8')
print('Prepared publication notebook and input manifest.')
