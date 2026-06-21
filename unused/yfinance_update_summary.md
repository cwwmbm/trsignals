# yfinance Update Summary

## Changes Detected

### Column Structure Changes
- **Before**: yfinance returned columns including `Adj Close` (separate adjusted close column)
- **After**: yfinance 0.2.66 with `auto_adjust=True` (default) no longer returns `Adj Close` column
  - Data is already adjusted, so only `Close`, `High`, `Low`, `Open`, `Volume` columns are returned

### Current Column Structure
After `normalize_dataframe()`:
- `Date` (from index)
- `Close` (already adjusted)
- `High`
- `Low`
- `Open`
- `Volume`

### Files Affected
1. **getdata.py** (line 199): Tries to drop `['Adj close']` column
2. **signal_check.py** (line 74): Tries to drop `['Adj close']` column

### Solution
Make the drop operation conditional - only drop if the column exists.

