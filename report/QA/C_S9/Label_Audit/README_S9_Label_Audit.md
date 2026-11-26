# S9 - Label Semantics Audit

Dataset: BRFSS 2015 Diabetes Health Indicators.

This audit compares the binary and 3-class CSV files.

Results:
- 253,680 records.
- 21 identical feature columns in identical row order.
- Binary label equals (Diabetes_012 == 2).
- All 4,631 records with Diabetes_012 == 1
  map to binary label 0.

No model training or performance evaluation
was performed.

Original labels and split indices remain unchanged.

Team approval of the revised target interpretation
is pending.

This package contains aggregate audit evidence only.
