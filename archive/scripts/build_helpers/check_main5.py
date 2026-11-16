from pathlib import Path
import base64
import nbformat

root = Path(__file__).resolve().parents[1]
path = root / 'Dipu_Bhai_Thesis/code/main_5.ipynb'
nb = nbformat.read(path, as_version=4)
nb.cells[1].source = nb.cells[1].source.replace(
    '$Q_{net}=V/(60\times\n10^{-6})$', '`Q_net = voltage_V / 60e-6`')
nbformat.validate(nb)
nbformat.write(nb, path)
image_dir = root / 'tmp/main5_execution/qa'
image_dir.mkdir(parents=True, exist_ok=True)
image_number = 0
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code':
        assert cell.execution_count is not None, f'Unexecuted code cell {i}'
        assert 'savefig(' not in cell.source and 'to_csv(' not in cell.source
    else:
        assert '\t' not in cell.source, f'Tab in markdown {i}'
    for output in cell.get('outputs', []):
        assert output.output_type != 'error'
        if 'image/png' in output.get('data', {}):
            image_number += 1
            (image_dir / f'figure_{image_number:02}.png').write_bytes(base64.b64decode(output.data['image/png']))
        if 'text/html' in output.get('data', {}):
            if i in [10, 19, 23]:
                print(f'CELL {i}\n{output.data.get("text/plain", "")}\n')
print('Notebook valid; every code cell executed; no execution errors;', image_number, 'embedded PNGs.')
print('Notebook bytes:', path.stat().st_size)
