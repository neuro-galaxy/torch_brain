# PIPPI DKT And Destrieux Brain Area Labels

Generated: 2026-05-01

This folder contains compact DKT and Destrieux brain-area labels for the
transformed high-coverage PIPPI subjects.

Subjects:

- `sub-01`
- `sub-09`
- `sub-21`
- `sub-23`
- `sub-42`

Coordinate assumption for all rows:

```text
native_t1_scanner_ras_to_freesurfer_tkr_ras
```

## Files

- `brain_area_labels.csv`: compact table with one row per electrode contact.
- `sub-XX_dkt_original_coordinate_labeled_overlay.png`: native-T1 QC overlays colored by DKT label.
- `sub-XX_destrieux_original_coordinate_labeled_overlay.png`: native-T1 QC overlays colored by Destrieux label.

## Label Columns

- `label_dkt`: derived from `aparc.DKTatlas+aseg.mgz`.
- `label_destrieux`: derived from `aparc.a2009s+aseg.mgz`.

## Coordinate Columns

- `native_t1_x`, `native_t1_y`, `native_t1_z`: original electrode coordinates in the subject's native T1/ACPC millimeter space. These are used for the top-level QC plots.
- `freesurfer_tkr_x`, `freesurfer_tkr_y`, `freesurfer_tkr_z`: transformed coordinates in the subject's FreeSurfer TK RAS space. These are used to sample FreeSurfer label volumes.

## Label Method

Both label types use the same transformed-coordinate pipeline:

1. Load original BIDS clinical electrode coordinates.
2. Transform native T1 scanner RAS coordinates into FreeSurfer TK RAS.
3. Sample the corresponding FreeSurfer atlas volume in the transformed coordinate frame.
4. Apply the depth-WM-style rule: keep useful cortical/subcortical labels, reject non-region labels such as `Unknown`, `CSF`, white matter, ventricles, and cerebellum, and search up to 8 mm for the nearest acceptable label if needed.

The DKT labels match the prior v9 DKT-only table. Destrieux labels were derived from `aparc.a2009s+aseg.mgz` with the same transform and depth-WM-style rule.

## Unresolved Counts

| Subject | Contacts | DKT unresolved | Destrieux unresolved |
|---|---:|---:|---:|
| `sub-01` | 103 | 2 | 2 |
| `sub-09` | 69 | 0 | 0 |
| `sub-21` | 110 | 2 | 2 |
| `sub-23` | 173 | 1 | 1 |
| `sub-42` | 112 | 1 | 1 |
