skills:
  - name: load_dataset
    description: Reads the budget CSV file, validates the required columns structure, configures missing data states, and reports the exact null count identifying which specific rows have null values before returning.
    input: File path to the ward_budget CSV.
    output: Validated dataset structured in memory, accompanied by a null validation report identifying exception rows.
    error_handling: Halts execution or raises a loud explicit error if core requested columns are missing. Drops or skips malformed non-data rows gracefully.

  - name: compute_growth
    description: Computes tracking growth strictly localized to a specific ward and category based entirely on the requested growth_type, explicitly tracking formulas.
    input: Validated dataset structure, a target ward, a target category, and a specific growth_type (e.g. MoM). 
    output: A periodic data aggregation mapping actual_spends, the computed growth metric, and the explicitly stringified mathematical formula used to determine the result per row.
    error_handling: Must output an explicit error or flag referencing the 'notes' column rationale if the current or comparative period involves a null actual_spend value; silently dropping nulls is strictly prohibited.
