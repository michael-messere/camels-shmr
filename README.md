# camels-shmr

Stellar–halo mass relation (SHMR) study on the **CAMELS IllustrisTNG `L50n512` SB35**
simulation suite (1024 boxes spanning cosmological + astrophysical parameter space).

The pipeline walks each box's SubLink merger trees to build per-snapshot **formation-time
catalogs**, then trains emulators that predict a halo's stellar mass (`log10 M★`) from its
halo mass, formation history, and the box's cosmo/astro parameters.

> **Note on paths:** the scripts and notebooks use absolute `/home/jovyan/...` paths because
> they were written to run on the **SDSC binder** environment where the CAMELS data lives.
> Adjust the `*_ROOT` / `OUT_DIR` config constants near the top of each file to run elsewhere.

---

## Pipeline at a glance

```
SubLink merger trees                                    (raw CAMELS data)
        │
        ▼  camels_merger_history_parallel.py            (or *_extended.py)
per-box formation-time catalogs  (HDF5, one file per box)
        │  └─ camels_add_extended_fields.py  (optionally adds extra columns in place)
        ▼
shmr_emulator.ipynb  /  shmr_npe_emulator*.ipynb        (train emulators)
        │
        ▼  model/mstar_emulator_SB35.joblib
run_emulator.ipynb                                      (load model + predict)
```

---

## Data

The generated catalogs and training tables are hosted on Google Drive (they're too large
for the repo and are excluded by [`.gitignore`](.gitignore)):

**📂 [CAMELS-SHMR data on Google Drive](https://drive.google.com/drive/folders/1Z2LaEqqnqsJxTcM6aoaNZhM2P7cBoqLF?usp=sharing)**

It contains two things:

1. **`emulator_training_table_snapshot_90.parquet`** — the flat, ready-to-train table for the
   **z=0 (snapshot 90)** emulators. One row per selected halo, written from
   [`shmr_emulator.ipynb`](shmr_emulator.ipynb) via
   `df.to_parquet(..., engine='fastparquet')`. Read it back with:
   ```python
   import pandas as pd
   df = pd.read_parquet('emulator_training_table_snapshot_90.parquet', engine='fastparquet')
   ```
2. **A tarball of the per-snapshot HDF5 catalogs** — the full set of `SB35_<i>_dm_mass_history.hdf5`
   files (one per box, *all* snapshots, not just z=0), produced by
   [`camels_merger_history_parallel.py`](camels_merger_history_parallel.py). Unpack with
   `tar -xzf <name>.tar.gz`.

### Format of the snapshot-90 training table (`.parquet`)

Each row is one z=0 central halo (above the `log10 M200c > 11` mass cut). Columns:

| Column | Description |
|--------|-------------|
| `M200c` | FoF M200c at snapshot 90 [physical Msun] |
| `t_form_10`, `t_form_25`, `t_form_50`, `t_form_90` | cosmic age [Gyr] at which the main-progenitor branch first reached 10/25/50/90% of its z=0 M200c |
| *(35 cosmo/astro params)* | the box's parameters, taken straight from the `CosmoAstroSeed_*.txt` column names (the `seed` column dropped) — e.g. `Omega_m`, `sigma_8`, `A_SN1`, `A_SN2`, `A_AGN1`, `A_AGN2`, … These are the `PARAM_KEYS` used as model features. |
| `box` | integer box index (`i` in `SB35_i`), used for the by-box train/test split |
| `Mstar` | subhalo stellar mass [physical Msun] — the prediction **target** |
| `logM200c` | `log10(M200c)` (feature) |
| `logMstar` | `log10(Mstar)` (target the model is trained on) |

The emulator features are `FEATURES = ['logM200c'] + ['t_form_10','t_form_25','t_form_50','t_form_90'] + PARAM_KEYS`,
and the target is `logMstar`. (Rows with `Mstar <= 0`, `M200c <= 0`, or any `NaN` feature are
dropped before saving.)

### Format of the per-snapshot HDF5 catalogs (in the tarball)

One file per box, `SB35_<i>_dm_mass_history.hdf5`, written by
[`camels_merger_history_parallel.py`](camels_merger_history_parallel.py). Structure:

- **Root attributes** — run metadata and the box's cosmo/astro parameters:
  `title`, `simulation`, `basePath`, `z0_snapshot` (90), `mass_unit`,
  `mass_cut_log10Msun` (11.0), `formation_percents` (`[10,25,50,90]`), `baryon_fraction`,
  plus every cosmo/astro parameter (`Omega_m`, `Omega_b`, `Omega_L`, `hubble_h`, `sigma_8`, …).
- **One group per snapshot**, named `snap_000` … `snap_090`. Each group has attrs
  `snapshot`, `redshift`, `age_Gyr`, `n_halos`, and the following equal-length datasets
  (one entry per halo present at that snapshot), each carrying a `description` attribute:

  | Dataset | Description |
  |---------|-------------|
  | `GroupNumber` | FoF group index (`SubhaloGrNr`) at this snapshot |
  | `GroupFirstSub` | main-progenitor subhalo index (`SubfindID`) at this snapshot |
  | `M200c_snap` | FoF M200c at this snapshot [physical Msun] |
  | `M200c_z0` | FoF M200c of the z=0 root halo [physical Msun] |
  | `Mstar` | subhalo stellar mass [physical Msun] |
  | `ISM_mass` | ISM gas mass within 2·r_half [physical Msun] |
  | `CGM_mass` | total subhalo gas − ISM [physical Msun] |
  | `fcgm` | `CGM_mass / M200c / baryon_fraction` |
  | `SFR` | subhalo star-formation rate [Msun/yr] |
  | `SplashbackFlag` | 1 if any progenitor was ever >10× the z=0 mass, else 0 |
  | `t_form_10/25/50/90` | interpolated cosmic age [Gyr] at 10/25/50/90% of the z=0 M200c |

The "extended" catalogs (from `*_extended.py` / `camels_add_extended_fields.py`) add
`M500c_snap`, `SubhaloVelDisp`, `SubhaloVmax`, and `SubhaloSpin` to each snapshot group.

Minimal read example:

```python
import h5py
with h5py.File('SB35_0_dm_mass_history.hdf5', 'r') as f:
    print(dict(f.attrs))                 # box params + run metadata
    g = f['snap_090']                    # z=0
    mstar = g['Mstar'][:]                # one value per halo
    m200  = g['M200c_snap'][:]
```

---

## Files

### Catalog-building scripts (run these first)

| File | What it does |
|------|--------------|
| [`camels_merger_history_parallel.py`](camels_merger_history_parallel.py) | **Main catalog builder.** For a range of boxes (default `SB35_0..SB35_1023`), runs the merger-tree analysis in parallel across N cores. Selects z=0 centrals above a mass cut (`log10 M200c > 11`), walks each main-progenitor branch once, computes cubic-interpolated **formation times** (10/25/50/90% of z=0 M200c), and records per-snapshot `Mstar / ISM / CGM / fcgm / SFR`. Writes one HDF5 file per box with the cosmo/astro parameters in the header. |
| [`camels_merger_history_parallel_extended.py`](camels_merger_history_parallel_extended.py) | Same as above, **plus** extra per-snapshot subhalo/group fields: `M500c_snap`, `SubhaloVelDisp`, `SubhaloVmax`, `SubhaloSpin` (all in physical/proper units, little-`h` divided out). |
| [`camels_add_extended_fields.py`](camels_add_extended_fields.py) | **Augments existing catalogs in place** with the extra columns above, *without* re-walking the trees — it re-reads each snapshot's group/subhalo catalog and indexes by the IDs already stored. Cheap way to add fields to catalogs already built by the non-extended script. |

All three accept `--start`, `--end`, and `--overwrite` (the parallel scripts also take `--nproc`). See each file's docstring for exact usage.

### Notebooks

| File | What it does |
|------|--------------|
| [`merger_tree.ipynb`](merger_tree.ipynb) | **Single-box interactive prototype** of the merger-tree / formation-time analysis. The exploratory version that the parallel `camels_merger_history_parallel.py` script was distilled from — useful for inspecting one box (`box = 'SB35_0'`) step by step. |
| [`shmr_emulator.ipynb`](shmr_emulator.ipynb) | **Point-estimate emulator.** Trains a `HistGradientBoostingRegressor` to predict `log10 M★` from halo mass, redshift, formation times, and the 35 cosmo/astro parameters. Train/test split is **by box**, so the score reflects generalization to unseen parameter sets. Produces the feature-importance and SHMR figures, and saves `model/mstar_emulator_SB35.joblib`. |
| [`shmr_npe_emulator.ipynb`](shmr_npe_emulator.ipynb) | **Density emulator (HaloFlow-style).** Learns the full distribution `p(log M★ \| features)` with a Mixture Density Network (NN outputting a mixture of Gaussians, trained on negative log-likelihood). Uses the **z=0 snapshot only**. Split **by box** (out-of-distribution test). |
| [`shmr_npe_emulator_randomsplit.ipynb`](shmr_npe_emulator_randomsplit.ipynb) | Same MDN as above but split over **random halos** instead of whole boxes — an **in-distribution calibration** test (train and test share the same parameter sets). Run to diagnose the over/under-confidence seen in the by-box version. |
| [`run_emulator.ipynb`](run_emulator.ipynb) | **Inference demo.** Loads the saved `model/mstar_emulator_SB35.joblib` bundle and calls `predict_mstar(logM200c, t_forms, params)` to predict stellar mass for new inputs. |

### Directories

| Path | Contents |
|------|----------|
| [`model/`](model/) | Saved trained emulator bundle(s), e.g. `mstar_emulator_SB35.joblib` (model + feature list + param keys + percentiles). |
| [`figures/`](figures/) | Output figures: `emulated_shmr.pdf`, `emulated_shmr_t50.pdf`, `feature_importance.pdf`. |
| [`old_scripts/`](old_scripts/) | Earlier/superseded notebook versions, kept for reference. |

### Other

| File | Purpose |
|------|---------|
| [`.gitignore`](.gitignore) | Ignores large generated artifacts (`*.parquet`, `*.joblib`). |
| [`push.sh`](push.sh) | Convenience script to stage, commit, and push all changes (see below). |

---

## Quick start

```bash
# 1. Build the formation-time catalogs (on the binder, with CAMELS data mounted)
python camels_merger_history_parallel.py --start 0 --end 1023 --nproc 8
# optionally add extended per-snapshot fields:
python camels_add_extended_fields.py

# 2. Train an emulator
#    open shmr_emulator.ipynb (point estimate) or shmr_npe_emulator.ipynb (density)

# 3. Predict with the trained model
#    open run_emulator.ipynb
```

## Pushing changes

A helper script is included to commit and push in one step:

```bash
./push.sh "optional commit message"
```

If no message is given it uses a timestamped default. See [`push.sh`](push.sh).
