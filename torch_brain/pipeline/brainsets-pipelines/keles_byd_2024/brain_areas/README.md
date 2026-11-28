# ds004798 Template-Space Atlas Labels

Generated: 2026-05-22

This folder is a compact subject/session-level export of DANDI 000623 / ds004798 template-space atlas labels and QC plots. It mirrors the compact shape of the ds003688 `v10-dkt-destrieux-all-subjects-transformed` export while preserving ds004798-specific coordinate naming.

Subjects included: 16 DANDI NWB/ephys subjects.
NWB sessions included: 29.
Contact rows: 4308.

Session IDs:

- `sub-p41cs_ses-P41CSR1`
- `sub-p41cs_ses-P41CSR2`
- `sub-p42cs_ses-P42CSR1`
- `sub-p42cs_ses-P42CSR2`
- `sub-p43cs_ses-P43CSR1`
- `sub-p43cs_ses-P43CSR2`
- `sub-p44cs_ses-P44CSR1`
- `sub-p47cs_ses-P47CSR1`
- `sub-p47cs_ses-P47CSR2`
- `sub-p48cs_ses-P48CSR1`
- `sub-p48cs_ses-P48CSR2`
- `sub-p49cs_ses-P49CSR1`
- `sub-p49cs_ses-P49CSR2`
- `sub-p51cs_ses-P51CSR1`
- `sub-p51cs_ses-P51CSR2`
- `sub-p53cs_ses-P53CSR1`
- `sub-p53cs_ses-P53CSR2`
- `sub-p54cs_ses-P54CSR1`
- `sub-p54cs_ses-P54CSR2`
- `sub-p55cs_ses-P55CSR1`
- `sub-p55cs_ses-P55CSR2`
- `sub-p56cs_ses-P56CSR1`
- `sub-p56cs_ses-P56CSR2`
- `sub-p57cs_ses-P57CSR1`
- `sub-p57cs_ses-P57CSR2`
- `sub-p58cs_ses-P58CSR1`
- `sub-p60cs_ses-P60CSR1`
- `sub-p62cs_ses-P62CSR1`
- `sub-p62cs_ses-P62CSR2`

Coordinate assumption for all rows:

```text
released_nwb_xyz_as_mni_template_ras
```

## Primary Table

- `brain_area_labels.csv`: compact table with one row per electrode contact per NWB session.

Primary label columns:

- `label_dkt`: compatibility name for the filtered DK/aparc-style label sampled from the available FreeSurfer template `aparc+aseg.mgz` volume. The installed template does not provide a ready-made `aparc.DKTatlas+aseg.mgz`, so this should not be treated as true DKT.
- `label_destrieux`: filtered Destrieux label sampled from the FreeSurfer template `aparc.a2009s+aseg.mgz` volume.

Coordinate columns:

- `mni_x`, `mni_y`, `mni_z`: released NWB electrode coordinates treated as common MNI/template RAS coordinates.
- `template_tkr_x`, `template_tkr_y`, `template_tkr_z`: corresponding coordinates converted into the FreeSurfer template TK RAS frame for sampling the available atlas volumes.

Identity/context columns:

- `subject`: DANDI/OpenNeuro-style subject id normalized as `sub-pXXcs`.
- `session`: NWB recording session, e.g. `P53CSR1` or `P53CSR2`.
- `subject_session`: unique subject/session key for provenance in this export.
- `provided_location`: original NWB brain-area/location label supplied by the dataset authors.
- `trajectory_role`: heuristic role along a macro/microwire trajectory (`outer`, `shaft`, `deep_target`, or `single_contact`).

The brainsets BYD pipeline intentionally does not merge labels by session. It
matches channel labels by `subject`, `electrode`, and rounded `mni_x/mni_y/mni_z`.
Duplicate session rows with the same label/context for that key are collapsed
during loading; rows with the same electrode but different coordinates remain
distinct.

## Label Method

1. Extract electrode metadata from every DANDI 000623 NWB `/general/extracellular_ephys/electrodes` table using byte-range reads; full NWB files were not downloaded.
2. Treat released `x`, `y`, `z` values as common MNI/template RAS coordinates, following the official dataset code and paper visualization approach.
3. Keep each NWB session represented in the export. This matters for subjects such as `sub-p53cs`, where `P53CSR1` and `P53CSR2` have different released coordinate tables.
4. Convert coordinates into the FreeSurfer template TK RAS frame for atlas-volume sampling.
5. Sample DK/aparc-style and Destrieux atlas volumes from FreeSurfer `cvs_avg35_inMNI152`.
6. Apply the same nearest-acceptable filtering style used for the earlier depth/white-matter labeling pass: keep useful cortical/subcortical labels, reject non-region labels such as `Unknown`, CSF, white matter, ventricles, corpus callosum, and related labels, then search up to 8 mm for the nearest acceptable label if needed.

The official NWB `provided_location` should remain the primary dataset-provided brain-area label. These atlas labels are secondary exploratory template-space labels.

## Primary QC Images

- `dkt_template_mni152_overlay.png`: all included NWB session contacts on the paper-matched MNI152NLin2009cAsym template backdrop, colored by `label_dkt`.
- `destrieux_template_mni152_overlay.png`: all included NWB session contacts on the paper-matched MNI152NLin2009cAsym template backdrop, colored by `label_destrieux`.

For both images, left/right homologous areas share fill color; black outline indicates left-hemisphere labels and no outline indicates right-hemisphere labels.

## Unresolved Counts

| Subject | Session | Subject session | Contacts | DKT/DK-aparc unresolved | Destrieux unresolved |
|---|---|---|---:|---:|---:|
| `sub-p41cs` | `P41CSR1` | `sub-p41cs_ses-P41CSR1` | 154 | 1 | 1 |
| `sub-p41cs` | `P41CSR2` | `sub-p41cs_ses-P41CSR2` | 154 | 1 | 1 |
| `sub-p42cs` | `P42CSR1` | `sub-p42cs_ses-P42CSR1` | 138 | 0 | 0 |
| `sub-p42cs` | `P42CSR2` | `sub-p42cs_ses-P42CSR2` | 138 | 0 | 0 |
| `sub-p43cs` | `P43CSR1` | `sub-p43cs_ses-P43CSR1` | 158 | 1 | 1 |
| `sub-p43cs` | `P43CSR2` | `sub-p43cs_ses-P43CSR2` | 158 | 1 | 1 |
| `sub-p44cs` | `P44CSR1` | `sub-p44cs_ses-P44CSR1` | 144 | 1 | 1 |
| `sub-p47cs` | `P47CSR1` | `sub-p47cs_ses-P47CSR1` | 159 | 2 | 2 |
| `sub-p47cs` | `P47CSR2` | `sub-p47cs_ses-P47CSR2` | 159 | 2 | 2 |
| `sub-p48cs` | `P48CSR1` | `sub-p48cs_ses-P48CSR1` | 157 | 0 | 0 |
| `sub-p48cs` | `P48CSR2` | `sub-p48cs_ses-P48CSR2` | 157 | 0 | 0 |
| `sub-p49cs` | `P49CSR1` | `sub-p49cs_ses-P49CSR1` | 157 | 0 | 0 |
| `sub-p49cs` | `P49CSR2` | `sub-p49cs_ses-P49CSR2` | 157 | 0 | 0 |
| `sub-p51cs` | `P51CSR1` | `sub-p51cs_ses-P51CSR1` | 159 | 0 | 0 |
| `sub-p51cs` | `P51CSR2` | `sub-p51cs_ses-P51CSR2` | 159 | 0 | 0 |
| `sub-p53cs` | `P53CSR1` | `sub-p53cs_ses-P53CSR1` | 157 | 0 | 0 |
| `sub-p53cs` | `P53CSR2` | `sub-p53cs_ses-P53CSR2` | 141 | 0 | 0 |
| `sub-p54cs` | `P54CSR1` | `sub-p54cs_ses-P54CSR1` | 152 | 0 | 0 |
| `sub-p54cs` | `P54CSR2` | `sub-p54cs_ses-P54CSR2` | 152 | 0 | 0 |
| `sub-p55cs` | `P55CSR1` | `sub-p55cs_ses-P55CSR1` | 157 | 1 | 1 |
| `sub-p55cs` | `P55CSR2` | `sub-p55cs_ses-P55CSR2` | 157 | 1 | 1 |
| `sub-p56cs` | `P56CSR1` | `sub-p56cs_ses-P56CSR1` | 148 | 0 | 0 |
| `sub-p56cs` | `P56CSR2` | `sub-p56cs_ses-P56CSR2` | 148 | 0 | 0 |
| `sub-p57cs` | `P57CSR1` | `sub-p57cs_ses-P57CSR1` | 160 | 0 | 0 |
| `sub-p57cs` | `P57CSR2` | `sub-p57cs_ses-P57CSR2` | 160 | 0 | 0 |
| `sub-p58cs` | `P58CSR1` | `sub-p58cs_ses-P58CSR1` | 160 | 0 | 0 |
| `sub-p60cs` | `P60CSR1` | `sub-p60cs_ses-P60CSR1` | 160 | 1 | 1 |
| `sub-p62cs` | `P62CSR1` | `sub-p62cs_ses-P62CSR1` | 74 | 1 | 1 |
| `sub-p62cs` | `P62CSR2` | `sub-p62cs_ses-P62CSR2` | 74 | 1 | 1 |
