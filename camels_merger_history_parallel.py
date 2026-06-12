#!/usr/bin/env python
"""
CAMELS IllustrisTNG L50n512 -- per-snapshot interpolated formation-time catalogs.

Runs the merger-tree analysis for a range of boxes (default SB35_0 .. SB35_1023)
in parallel across N CPU cores. Each box is one independent task: it reads its own
SubLink tree_extended.hdf5, selects z=0 centrals above a mass cut, computes cubic-
interpolated formation times (10/25/50/90% of z=0 M200c), records per-snapshot
Mstar / ISM / CGM / fcgm / SFR along each main-progenitor branch, and writes a single
HDF5 file with the cosmological + astrophysical parameters in the header.

Usage
-----
    python camels_merger_history_parallel.py                 # all 1024 boxes, 8 cores
    python camels_merger_history_parallel.py --start 0 --end 1023 --nproc 8
    python camels_merger_history_parallel.py --overwrite     # redo finished boxes
"""

import os
import argparse
import traceback
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
import h5py
from scipy.interpolate import interp1d
from astropy.cosmology import FlatLambdaCDM

try:
    from tqdm import tqdm
except ImportError:                      # tqdm optional
    def tqdm(x, **kw): return x

import illustris_python as il


# ----------------------------------------------------------------------------- config
SUITE    = 'SB35'
RES      = 'L50n512'
MASS_CUT = 11.0                          # log10(M200c / physical Msun)
PERCENTS = [10, 25, 50, 90]

FOF_ROOT  = f'/home/jovyan/FOF_Subfind/IllustrisTNG/{RES}'
TREE_ROOT = f'/home/jovyan/PUBLIC_RELEASE/SubLink/IllustrisTNG/{RES}'
PARAM_FILE = (f'/home/jovyan/Sims/IllustrisTNG/{RES}/{SUITE}/'
              f'CosmoAstroSeed_IllustrisTNG_{RES}_{SUITE}.txt')
OUT_DIR   = '/home/jovyan/home/camels_mass_history'

TREE_FIELDS = ['SnapNum', 'SubfindID', 'SubhaloGrNr', 'Group_M_Crit200',
               'SubhaloMassType', 'SubhaloMassInRadType', 'SubhaloSFR',
               'FirstProgenitorID', 'SubhaloID']


# ------------------------------------------------------------------- per-tree helpers
def formation_times_from_tree(snapnum, mass, age, percents=PERCENTS, n_grid=2000):
    """Cubic-interpolated ages at which the branch first reached p% of z=0 mass."""
    ages = age[snapnum]
    order = np.argsort(ages)
    a, m = ages[order], mass[order]
    ok = np.isfinite(a) & np.isfinite(m)
    a, m = a[ok], m[ok]

    present_day = m[-1] if len(m) else np.nan
    if len(a) < 2 or not (np.isfinite(present_day) and present_day > 0):
        return [np.nan] * len(percents)

    kind = 'cubic' if len(a) >= 4 else 'linear'
    f = interp1d(a, m, kind=kind)
    ag = np.linspace(a[0], a[-1], n_grid)
    mg = f(ag)

    out = []
    for p in percents:
        target = (p / 100.0) * present_day
        above = np.where(mg >= target)[0]
        out.append(float(ag[above[0]]) if len(above) else np.nan)
    return out


def make_mpb_loader(tree_all):
    """Return a load_mpb(snap, subfind_id) closure for one box's tree arrays."""
    sid = tree_all['SubhaloID']
    if np.array_equal(sid, np.arange(len(sid))):
        def to_row(x): return x
    else:
        _map = {int(s): i for i, s in enumerate(sid)}
        def to_row(x): return _map.get(int(x), -1)

    def load_mpb(snap, subfind_id):
        rows = np.where((tree_all['SnapNum'] == snap) &
                        (tree_all['SubfindID'] == subfind_id))[0]
        if len(rows) == 0:
            return None
        out = {k: [] for k in TREE_FIELDS}
        r = int(rows[0])
        while r is not None and r >= 0:
            for k in TREE_FIELDS:
                out[k].append(tree_all[k][r])
            fp = int(tree_all['FirstProgenitorID'][r])
            r = to_row(fp) if fp >= 0 else -1
        return {k: np.array(v) for k, v in out.items()}

    return load_mpb


# ----------------------------------------------------------------------- box pipeline
def read_params(box):
    """Cosmo/astro parameters for one box from the CosmoAstroSeed table."""
    with open(PARAM_FILE) as fh:
        raw = [ln.rstrip() for ln in fh if ln.strip()]
    header = raw[0].lstrip('#').split()
    ptab = pd.DataFrame([ln.split() for ln in raw[1:]], columns=header)
    row = ptab[ptab[header[0]].astype(str) == box]
    if len(row) != 1:
        raise ValueError(f'{box} not found in {PARAM_FILE}')
    fp = {k: float(row.iloc[0][k]) for k in header[1:]}

    def pick(*names, default=np.nan):
        for n in names:
            for k, v in fp.items():
                if k.lower() == n.lower():
                    return v
        return default

    Om = pick('Omega_m', 'Omega0')
    Ob = pick('Omega_b', 'OmegaBaryon')
    h  = pick('h', 'HubbleParam')
    params = dict(fp)
    params.update({'Omega_m': Om, 'Omega_b': Ob, 'Omega_L': 1.0 - Om,
                   'hubble_h': h, 'sigma_8': pick('sigma_8', 'sigma8')})
    return params, Om, Ob, h


def process_box(box_idx, overwrite=False):
    """Full pipeline for one box. Returns (box, status, info)."""
    box = f'{SUITE}_{box_idx}'
    out_fn = f'{OUT_DIR}/{box}_dm_mass_history.hdf5'
    try:
        if os.path.exists(out_fn) and not overwrite:
            return (box, 'skip', 'exists')

        base = f'{FOF_ROOT}/{SUITE}/{box}'
        tree_path = f'{TREE_ROOT}/{SUITE}/{box}/tree_extended.hdf5'
        if not os.path.exists(tree_path):
            return (box, 'missing', tree_path)

        # cosmology + astro params
        params, Om, Ob, h = read_params(box)
        baryon_fraction = Ob / Om

        # snapshot range (detect from group dirs)
        snaps = sorted(int(d.split('_')[1]) for d in os.listdir(base)
                       if d.startswith('groups_'))
        base_snap, n_snap = snaps[-1], snaps[-1] + 1

        # redshift / age
        redshift = np.array([
            h5py.File(f'{base}/groups_{s:03d}/groups_{s:03d}.0.hdf5',
                      'r')['Header'].attrs['Redshift'] for s in range(n_snap)])
        age = FlatLambdaCDM(H0=h * 100, Om0=Om).age(redshift).value

        # load the tree once, build the MPB loader
        with h5py.File(tree_path, 'r') as f:
            tree_all = {k: f[k][:] for k in TREE_FIELDS}
        load_mpb = make_mpb_loader(tree_all)

        # z=0 selection: central halos above the physical-mass cut
        groups = il.groupcat.loadHalos(base, base_snap,
                                       fields=['GroupFirstSub', 'Group_M_Crit200'])
        gfs_all = groups['GroupFirstSub']
        with np.errstate(divide='ignore'):
            logm = np.log10(groups['Group_M_Crit200'] * 1e10 / h)
        sel = (logm > MASS_CUT) & (gfs_all >= 0)
        roots = gfs_all[sel]

        # accumulate per-snapshot columns
        base_cols = ['GroupNumber', 'GroupFirstSub', 'M200c_snap', 'M200c_z0',
                     'Mstar', 'ISM_mass', 'CGM_mass', 'fcgm', 'SFR', 'SplashbackFlag']
        tcols = [f't_form_{p}' for p in PERCENTS]
        cols = {s: {c: [] for c in base_cols + tcols} for s in range(n_snap)}

        for gfs in roots:
            tree = load_mpb(base_snap, int(gfs))
            if tree is None:
                continue
            tforms = formation_times_from_tree(tree['SnapNum'],
                                               tree['Group_M_Crit200'] * 1e10, age)
            m200c = tree['Group_M_Crit200'] * 1e10 / h
            mstar = tree['SubhaloMassType'][:, 4] * 1e10 / h
            gas   = tree['SubhaloMassType'][:, 0] * 1e10 / h
            ism   = tree['SubhaloMassInRadType'][:, 0] * 1e10 / h
            cgm   = gas - ism
            sfr   = tree['SubhaloSFR']
            with np.errstate(divide='ignore', invalid='ignore'):
                fcgm = cgm / m200c / baryon_fraction

            m200c_z0 = float(m200c[0])
            splash = int((m200c[1:] > m200c_z0 * 10).any()) if len(m200c) > 1 else 0

            for i in range(len(tree['SnapNum'])):
                c = cols[int(tree['SnapNum'][i])]
                c['GroupNumber'].append(int(tree['SubhaloGrNr'][i]))
                c['GroupFirstSub'].append(int(tree['SubfindID'][i]))
                c['M200c_snap'].append(float(m200c[i]))
                c['M200c_z0'].append(m200c_z0)
                c['Mstar'].append(float(mstar[i]))
                c['ISM_mass'].append(float(ism[i]))
                c['CGM_mass'].append(float(cgm[i]))
                c['fcgm'].append(float(fcgm[i]))
                c['SFR'].append(float(sfr[i]))
                c['SplashbackFlag'].append(splash)
                for tc, tv in zip(tcols, tforms):
                    c[tc].append(tv)

        catalogs = {s: pd.DataFrame(cols[s]) for s in range(n_snap)}
        save_box(catalogs, out_fn, box, base, base_snap, redshift, age,
                 params, baryon_fraction)
        return (box, 'ok', f'{int(sel.sum())} halos')

    except Exception as e:
        return (box, 'error', f'{e}\n{traceback.format_exc()}')


def save_box(catalogs, filename, box, basePath, base_snap, redshift, age,
             params, baryon_fraction):
    col_docs = {
        'GroupNumber':    'FoF group index (SubhaloGrNr) at this snapshot',
        'GroupFirstSub':  'main-progenitor subhalo index (SubfindID) at this snapshot',
        'M200c_snap':     'FoF M200c at this snapshot [physical Msun]',
        'M200c_z0':       'FoF M200c of the z=0 root halo [physical Msun]',
        'Mstar':          'subhalo stellar mass [physical Msun]',
        'ISM_mass':       'ISM gas mass within 2 r_half [physical Msun]',
        'CGM_mass':       'total subhalo gas - ISM [physical Msun]',
        'fcgm':           'CGM_mass / M200c / baryon_fraction',
        'SFR':            'subhalo SFR [Msun/yr]',
        'SplashbackFlag': '1 if a progenitor was ever >10x the z=0 mass, else 0',
    }
    for p in PERCENTS:
        col_docs[f't_form_{p}'] = f'interpolated age [Gyr] at {p}% of the z=0 M200c'

    tmp = filename + '.tmp'             # write to tmp then rename (atomic-ish)
    with h5py.File(tmp, 'w') as f:
        f.attrs['title']       = 'CAMELS L50n512 interpolated formation-time catalog'
        f.attrs['simulation']  = f'CAMELS IllustrisTNG {RES} {box}'
        f.attrs['basePath']    = basePath
        f.attrs['z0_snapshot'] = base_snap
        f.attrs['mass_unit']   = 'physical Msun (Group_M_Crit200 * 1e10 / h)'
        f.attrs['mass_cut_log10Msun'] = MASS_CUT
        f.attrs['formation_percents'] = list(PERCENTS)
        f.attrs['baryon_fraction'] = baryon_fraction
        for k, v in params.items():
            f.attrs[k] = v

        for snap, df in sorted(catalogs.items(), reverse=True):
            g = f.create_group(f'snap_{snap:03d}')
            g.attrs['snapshot'] = snap
            g.attrs['redshift'] = float(redshift[snap])
            g.attrs['age_Gyr']  = float(age[snap])
            g.attrs['n_halos']  = len(df)
            for col in df.columns:
                d = g.create_dataset(col, data=df[col].to_numpy())
                d.attrs['description'] = col_docs.get(col, '')
    os.replace(tmp, filename)


# ------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end',   type=int, default=1023)   # inclusive
    ap.add_argument('--nproc', type=int, default=8)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    boxes = list(range(args.start, args.end + 1))
    print(f'{SUITE}: boxes {args.start}..{args.end}  ({len(boxes)} total)  '
          f'on {args.nproc} cores -> {OUT_DIR}')

    worker = partial(process_box, overwrite=args.overwrite)
    counts = {'ok': 0, 'skip': 0, 'missing': 0, 'error': 0}
    with Pool(args.nproc) as pool:
        for box, status, info in tqdm(pool.imap_unordered(worker, boxes),
                                      total=len(boxes)):
            counts[status] = counts.get(status, 0) + 1
            if status in ('error', 'missing'):
                tqdm.write(f'[{status}] {box}: {info.splitlines()[0]}')

    print('\nSummary:', counts)


if __name__ == '__main__':
    main()
