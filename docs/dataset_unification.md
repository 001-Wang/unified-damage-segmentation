# Crack Dataset Unification Pipeline

Recommended order:

```bash
python scripts/00_inventory.py
python scripts/01_build_crack_manifest.py --include-dacl --csb patch_70_30_corrected
python scripts/02_materialize_crack_dataset.py --limit 50
python scripts/03_check_unified_crack_dataset.py --sample-limit 50
```

If the 50-sample smoke test looks good, build the full dataset:

```bash
python scripts/02_materialize_crack_dataset.py
python scripts/03_check_unified_crack_dataset.py
```

Default choices are conservative:

- LCW uses `512x512`, avoiding duplicates from `original` and dilated variants.
- CSB uses `512_512/70_30/train_corrected` and `test_corrected`, avoiding duplicate patch variants.
- DACL10K is included only when `--include-dacl` is passed, and only `Crack` plus `ACrack` polygons are converted to binary crack masks.
- Masks are normalized to single-channel PNG files with values `0` for background and `1` for crack.

For a first crack segmentation model, train on `unified_crack/manifest.csv`.
For later multi-class damage segmentation, create a separate DACL10K converter instead of mixing binary crack masks with the 19-class DACL10K labels.

