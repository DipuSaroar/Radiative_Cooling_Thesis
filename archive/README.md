# Archive: earlier exploratory work

Kept for reference only. The final, validated analysis is `code/main.ipynb` at the
repository root; nothing here is used by it.

| Folder | Contents |
|---|---|
| `notebooks/` | Earlier notebooks: `main_3`, `main_clean`, `main_4` (5-min grid, interpolation, MLP), `main_5`, and the `data_clean` branch version of `main.ipynb` |
| `data/` | Earlier intermediate datasets (`merged_all_*`, 5-min `model_input_*`) |
| `figures/` | Figure PDFs/PNGs exported by the earlier notebooks |
| `scripts/` | Earlier merge/figure scripts; `build_helpers/` holds the helper scripts used to build and check `main_5` and the submission |
| `requirements_main4_main5.txt` | Dependencies used by the earlier notebooks |

These earlier versions differ from the final method (3-min bins, 20-sample moving
average, no interpolation, Random Forest), so their results are not comparable.
