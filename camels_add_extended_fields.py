#!/usr/bin/env python
"""
Augment the existing CAMELS formation-time catalogs with extra per-snapshot fields,
WITHOUT re-walking the merger trees.

The existing catalogs (from camels_merger_history_parallel.py) already store, for every
node at every snapshot, its SubfindID (GroupFirstSub) and FoF index (GroupNumber). The new
quantities live in the per-snapshot group/subhalo catalogs, so we just read each snapshot's
catalog once and index it by those IDs -- no tree load, no SubhaloID dict, tiny memory.

Added columns (physical units, h divided out, per TNG specs):
    M500c_snap        Group_M_Crit500 -> physical Msun  (* 1e10 / h)
    SubhaloVelDisp    velocity dispersion [km/s]        (no h)
    SubhaloVmax       V_max [km/s]                      (no h)
    SubhaloSpin       |spin| -> proper (kpc)(km/s)      (/ h, * a=1/(1+z))

New datasets are appended into the existing files in place (existing columns, including the
t_forms, are left untouched). A box is skipped if it already has the new columns.

Usage
-----
    python camels_add_extended_fields.py                       # all boxes, in place
    python camels_add_extended_fields.py --start 0 --end 255
    python camels_add_extended_fields.py --overwrite           # recompute new columns
"""

import os
import argparse
import traceback

import numpy as np
import h5py

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw): return x

import illustris_python as il


# ----------------------------------------------------------------------------- config
SUITE   = 'SB35'
RES     = 'L50n512'
FOF_ROOT = f'/home/jovyan/FOF_Subfind/IllustrisTNG/{RES}'
CAT_DIR  = '/home/jovyan/home/camels_mass_history'        # existing catalogs (edited in place)

NEW_COLS = ['M500c_snap', 'SubhaloVelDisp', 'SubhaloVmax', 'SubhaloSpin']
SUB_FIELDS = ['SubhaloVelDisp', 'SubhaloVmax', 'SubhaloSpin']

COL_DOCS = {
    'M500c_snap':     'FoF M500c at this snapshot [physical Msun] (Group_M_Crit500 * 1e10 / h)',
    'SubhaloVelDisp': '1-D velocity dispersion of member particles/cells [km/s]',
    'SubhaloVmax':    'maximum of the spherically-averaged rotation curve [km/s]',
    'SubhaloSpin':    'magnitude of the spin vector [proper (kpc)(km/s); h out, x a=1/(1+z)]',
}


def _hubble_h(attrs):
    """Pull little-h from the catalog header (handles a few naming conventions)."""
    for k in ('hubble_h', 'HubbleParam', 'h', 'little_h'):
        if k in attrs:
            return float(attrs[k])
    raise KeyError('no hubble_h / HubbleParam in catalog header')


def process_box(box_idx, overwrite=False):
    box = f'{SUITE}_{box_idx}'
    cat_fn = f'{CAT_DIR}/{box}_dm_mass_history.hdf5'
    try:
        if not os.path.exists(cat_fn):
            return (box, 'missing', cat_fn)

        # skip if already augmented (check the z=0 snapshot group)
        with h5py.File(cat_fn, 'r') as f:
            h = _hubble_h(f.attrs)
            z0 = int(f.attrs['z0_snapshot'])
            snap_names = sorted(n for n in f.keys() if n.startswith('snap_'))
            already = all(c in f[f'snap_{z0:03d}'] for c in NEW_COLS)
        if already and not overwrite:
            return (box, 'skip', 'already has new columns')

        base = f'{FOF_ROOT}/{SUITE}/{box}'

        with h5py.File(cat_fn, 'a') as f:
            for name in snap_names:
                g = f[name]
                snap = int(name.split('_')[1])
                z = float(g.attrs['redshift'])
                a = 1.0 / (1.0 + z)
                n = len(g['GroupFirstSub'])

                if n == 0:
                    out = {c: np.array([], float) for c in NEW_COLS}
                else:
                    sub_id = g['GroupFirstSub'][:].astype(np.int64)   # SubfindID
                    grp_id = g['GroupNumber'][:].astype(np.int64)     # FoF index

                    subs  = il.groupcat.loadSubhalos(base, snap, fields=SUB_FIELDS)
                    halos = il.groupcat.loadHalos(base, snap, fields=['Group_M_Crit500'])
                    n_sub = int(subs['count'])
                    m500_all = (halos['Group_M_Crit500'] if isinstance(halos, dict)
                                else halos)
                    n_grp = len(m500_all)

                    ok_s = (sub_id >= 0) & (sub_id < n_sub)
                    ok_g = (grp_id >= 0) & (grp_id < n_grp)
                    s_idx = np.where(ok_s, sub_id, 0)
                    g_idx = np.where(ok_g, grp_id, 0)

                    veldisp = np.where(ok_s, subs['SubhaloVelDisp'][s_idx], np.nan)
                    vmax    = np.where(ok_s, subs['SubhaloVmax'][s_idx],    np.nan)
                    spin    = np.linalg.norm(subs['SubhaloSpin'][s_idx], axis=1) / h * a
                    spin    = np.where(ok_s, spin, np.nan)
                    m500    = np.where(ok_g, m500_all[g_idx] * 1e10 / h,    np.nan)

                    out = {'M500c_snap': m500, 'SubhaloVelDisp': veldisp,
                           'SubhaloVmax': vmax, 'SubhaloSpin': spin}

                for c in NEW_COLS:
                    if c in g:
                        del g[c]
                    d = g.create_dataset(c, data=out[c])
                    d.attrs['description'] = COL_DOCS[c]

            # record the new unit conventions in the header
            f.attrs['spin_unit']     = 'proper (kpc)(km/s) (h divided out, x a=1/(1+z))'
            f.attrs['velocity_unit'] = 'km/s (SubhaloVmax, SubhaloVelDisp)'

        return (box, 'ok', f'{len(snap_names)} snapshots')

    except Exception as e:
        return (box, 'error', f'{e}\n{traceback.format_exc()}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end',   type=int, default=1023)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    boxes = list(range(args.start, args.end + 1))
    print(f'{SUITE}: augmenting boxes {args.start}..{args.end}  ({len(boxes)} total)  '
          f'-> {CAT_DIR} (in place)')

    counts = {'ok': 0, 'skip': 0, 'missing': 0, 'error': 0}
    for bi in tqdm(boxes):
        box, status, info = process_box(bi, overwrite=args.overwrite)
        counts[status] = counts.get(status, 0) + 1
        if status in ('error', 'missing'):
            tqdm.write(f'[{status}] {box}: {info.splitlines()[0]}')
    print('\nSummary:', counts)


if __name__ == '__main__':
    main()
