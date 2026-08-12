Official district crop statistics (Phase 2 — yield charts)

This Phase 2 “yield chart” uses ONLY your locally provided official APY files
(no invented values).

Supported local formats (choose one):

1) CSV (long format)
   - Export a district/crop/year table with columns like: Year, State, District, Crop, Yield
   - Save as: `data/apy/district_apy.csv`
   - Or set: `APY_CSV_PATH=/absolute/path/to/your_export.csv`

2) DES/APY wide exports (what you added)
   - `data/apy/Rabi_prod.xls`
   - `data/apy/Kharif_prod.xls`
   - These are HTML-table exports saved with an `.xls` extension.
   - The app parses them and extracts the correct “Yield (...)” column per crop.

Notes
-----
- Units (kg/ha vs bales/ha vs tonnes/ha) are taken from the source yield column in your file.
- If automatic GPS->district matching fails (geocoding), the UI lets you enter state + district manually.
- Always cite the exact dataset title, publisher (e.g. MoAFW / DES), and download date in your report.
