skills:
  - name: classify_complaint
    description: Processes a single complaint row to categorize it, assign priority based on keywords, compute a specific reason, and flag if ambiguous.
    input: A string row or dictionary representation of a civic complaint.
    output: A structured output (e.g., dictionary) mapping category, priority, reason, and flag.
    error_handling: If the complaint description is incomplete, ambiguous, or matches none of the standard categories natively, the flag must be marked as "NEEDS_REVIEW".

  - name: batch_classify
    description: Reads an input CSV file containing citizen complaints, applies the classify_complaint logic row-by-row, and outputs the result.
    input: An input file path (CSV format) to read complaints.
    output: Writes parsed structured data to an output CSV file ensuring the same order and mapped fields.
    error_handling: Fails gracefully if the input CSV path doesn't exist or is improperly formatted. Skips rows that are entirely malformed and logs a warning.
