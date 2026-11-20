# Power BI Import and Report Setup

The offline dashboard runs in a browser on macOS. [Power BI Desktop](https://learn.microsoft.com/en-us/power-bi/fundamentals/desktop-get-the-desktop) requires Windows. The aggregate CSVs can be used to recreate the report there.

1. In Power BI Desktop, select **Home → Get data → Text/CSV** and import the eight CSVs in `report/`. Use UTF-8 and a comma delimiter. If your locale uses a different decimal separator, use **Change Type → Using Locale** with English (United States) for decimal columns. See the [Microsoft Text/CSV connector guide](https://learn.microsoft.com/en-us/power-query/connectors/text-csv).
2. Set counts and `bin` to Whole Number. Set `loan_amount` and `retained_amount` to Fixed Decimal Number. Format rate fields such as `actual_rate`, `mean_pd`, and `retention` as percentages. Use decimals for ROC-AUC and Brier.
3. On the overview page, chart `vintages[actual_rate]` by `year` and add loan and charge-off counts to the tooltips. Use a matrix with `factor` and `segment` for `risk_factors`. Display `quality` and `missing` as tables.
4. On the model page, use `metrics` to compare both models and probability versions. For the calibration chart, use `calibration[bin]` as the axis and compare `mean_pd` with `actual_rate`. Use single-select slicers for model and version. Each row is already aggregated, so use **Don't summarize** where available or **Maximum** when each axis value has a single filtered row.
5. On the policy page, use `risk_tiers` to show loans and actual charge-off rates by tier. Use `policy` to chart retention and retained charge-off rate by threshold. A threshold of 1 represents the retain-all baseline. Threshold scenarios overlap, so read retained counts and amounts for one model and threshold at a time.
6. Compute combined charge-off rates from counts to weight by portfolio size. For example, for the `vintages` table use:

```dax
Observed Charge-off Rate =
DIVIDE(SUM(vintages[chargeoffs]), SUM(vintages[loans]))
```

These visuals can be built from independent tables. For a shared model slicer, create a dimension with `Models = DISTINCT(metrics[model])` and add one-to-many, single-direction relationships from that dimension to `metrics`, `calibration`, `risk_tiers`, and `policy`.
