# Alloy databases (processed .npz), bundled with the `datasets` package

These `*.npz` files were copied from the sibling `Alloys-SR` repository
(`processed_data/`) so the alloy datasets are bundled and usable offline.
Schema (per file): `targets`, `formulas`, `target_name`, `source`.

They are loaded by `datasets.alloy.load_alloy_dataset` (the default
`data_dir`), i.e. `get_dataset("alloy_density")` needs no path.

Source datasets retain their original licensing from the `Alloys-SR`/literature
sources; see `Alloys-SR/raw_datas/` and `Alloys-SR/refs/` for provenance.
