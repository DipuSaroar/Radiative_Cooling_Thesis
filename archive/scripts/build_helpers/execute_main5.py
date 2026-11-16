from pathlib import Path
import os
import sys
import json
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
work = root / 'tmp/main5_execution'
for folder in ['jupyter/kernels/python3', 'runtime', 'ipython', 'matplotlib']:
    (work / folder).mkdir(parents=True, exist_ok=True)
os.environ['JUPYTER_PATH'] = str(work / 'jupyter')
os.environ['JUPYTER_RUNTIME_DIR'] = str(work / 'runtime')
os.environ['IPYTHONDIR'] = str(work / 'ipython')
os.environ['MPLCONFIGDIR'] = str(work / 'matplotlib')
(work / 'jupyter/kernels/python3/kernel.json').write_text(json.dumps({
    'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
    'display_name': 'Python 3', 'language': 'python',
}), encoding='utf-8')
path = root / 'Dipu_Bhai_Thesis/code/main_5.ipynb'
nb = nbformat.read(path, as_version=4)
output_path = path
if len(sys.argv) > 1:
    assert sys.argv[1] == 'native_exports'
    for cell in nb.cells:
        if cell.cell_type == 'code':
            cell.source = cell.source.replace("DATA_SCOPE = 'full_campaign'", "DATA_SCOPE = 'native_exports'")
            cell.outputs = []
            cell.execution_count = None
    output_path = work / 'native_scope_check.ipynb'
def progress(cell, cell_index):
    if cell.cell_type == 'code':
        print(f'Executing cell {cell_index}: {cell.source.splitlines()[0][:100]}', flush=True)
client = NotebookClient(nb, timeout=600, kernel_name='python3',
                        resources={'metadata': {'path': str(path.parent)}},
                        allow_errors=False, on_cell_start=progress)
try:
    client.execute()
finally:
    nbformat.write(nb, output_path)
nbformat.validate(nb)
images = sum('image/png' in output.get('data', {}) for cell in nb.cells for output in cell.get('outputs', []))
errors = [output for cell in nb.cells for output in cell.get('outputs', []) if output.get('output_type') == 'error']
assert not errors
assert images >= 10, images
print(f'Execution complete: {images} inline figures; {output_path.stat().st_size:,} bytes', flush=True)
for cell in nb.cells:
    for output in cell.get('outputs', []):
        if output.get('output_type') == 'display_data' and 'text/markdown' in output.get('data', {}):
            text = output['data']['text/markdown']
            if 'Executed baseline result' in text:
                print(text)
