"""
India-only context for district-level crop / yield layers (planning aid).

No statistics are embedded here — only standard references so downstream
code can load official CSV/API exports without inventing numbers.

Authoritative open references
-----------------------------
1. Open Government Data (India) — catalogs under Ministry of Agriculture &
   Farmers' Welfare / DA&FW, e.g. district-wise season-wise crop production
   statistics (area, production, yield by district / crop / season / year).
   Portal: https://www.data.gov.in/

2. Directorate of Economics and Statistics (DES), MoA&FW — Area–Production–
   Yield (APY) query reports and downloads.
   https://www.data.desagri.gov.in/

Southern India focus (state names as typically appear in official tables)
---------------------------------------------------------------------------
Use these strings to filter rows after normalizing spelling/case in ingested
files — exact spellings vary by dataset vintage.
"""

# Lowercase tokens for post-normalization matching (not exhaustive for all UTs).
SOUTH_INDIA_STATE_ALIASES_LOWER = frozenset(
    {
        "andhra pradesh",
        "telangana",
        "karnataka",
        "tamil nadu",
        "kerala",
    }
)
