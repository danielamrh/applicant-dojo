# Implementation Notes

**Candidate Name:** Daniel Amrhein  
**Date:** 27.11.2025
**Time Spent:** 21:00

---

## 📝 Summary

Brief overview of what you implemented and your overall approach.

---

## ✅ Completed

List what you successfully implemented:

- [✅] `ingest_data()` - basic functionality
- [✅] `ingest_data()` - deduplication
- [✅] `ingest_data()` - sorting
- [✅] `ingest_data()` - validation
- [✅] `detect_anomalies()` - zscore method
- [✅] `detect_anomalies()` - additional methods (iqr/rolling)
- [✅] `summarize_metrics()` - basic statistics
- [✅] `summarize_metrics()` - quality metrics
- [✅] `summarize_metrics()` - time windowing
- [ ] Additional tests beyond exposed tests

---

## 🤔 Assumptions & Design Decisions

Document key assumptions and why you made certain design choices.

### Data Ingestion
- **Assumption 1:** Duplicates should be identified by an identical set of (timestamp, sensor, value, unit)
  - **Rationale:** In a real-world system, two readings with identical core values are redundant and should be de-duplicated to maintain data integrity and prevent double-counting in metrics. keep='first' is used.
  - **Alternative considered:** Keeping the record with the 'GOOD' quality flag, but this adds complexity when value might be NaN and quality is 'BAD' for both. The conservative approach of matching all core fields was chosen for reliability.

- **Assumption 2:** Rows missing a timestamp or sensor ID are unrecoverable and must be dropped.
  - **Rationale:** These two fields are essential for time-series analysis and grouping. A record cannot be processed without knowing when or what it is.

### Anomaly Detection
- **Method choice:** Implemented "zscore" (required), "iqr," and "rolling" Z-score. The rolling method uses a default window of 20, as no window size was specified in the function signature.
- **Threshold handling:** The threshold defines the boundary for flagging, and the anomaly_score is the distance from the mean (zscore) or the nearest bound (iqr), clipped at 0. This provides a measurable severity score.
- **Missing data:** NaNs are excluded from the mean/std/IQR calculation but are kept in the final DataFrame. They are automatically assigned is_anomaly=False and anomaly_score=0.0, as a missing value cannot statistically be an anomaly of a valid measurement.

### Metrics Summarization
- **Metric selection:** Included mean, std, min, max, count (non-null), null_count, good_quality_pct, and anomaly_rate.
Core stats: mean, std
Data Quality: count, null_count, good_quality_pct
Operational insides: anomaly_rate
- **Aggregation strategy:** Used the standard Pandas groupby().agg() method with a list of functions, which is highly robust and avoids the TypeError encountered with dictionary unpacking in some environments. 


---

## ⚠️ Known Limitations

Be honest about what doesn't work perfectly or edge cases you didn't handle.

### Edge Cases Not Fully Handled
1. **[Edge case 1]:** The current ingestion process uses pd.to_datetime without enforcing a specific timezone.

   - **Impact:** [What breaks or degrades]
   - **Workaround:** [Temporary solution if any]

2. **[Edge case 2]:** he current system assumes all sensor data is kept in its original unit.
   - **Impact:** If metrics are summarized across different sensors with the same name but different units (e.g., "pressure" in kPa and "pressure" in bar), the statistical results will be numerically invalid.
   - **Workaround:** A centralized unit conversion function would be required in ingest_data to normalize all readings to a base unit (e.g., all pressures to kPa) using the unit column.

### Performance Considerations
- **Large datasets:** The process relies heavily on copying DataFrames (.copy() in detect_anomalies), filtering, and full DataFrame merges (pd.merge). For extremely large, streaming industrial datasets (millions of records per hour), this approach might become a bottleneck.
- **Memory usage:** [Any memory-intensive operations]

---

## 🚀 Next Steps

If you had more time, what would you improve or add?

### Priority 1: [Highest priority improvement]
- **What:** [Description]
- **Why:** [Impact/value]
- **Estimated effort:** [Time estimate]

### Priority 2: [Second priority]
- **What:**
- **Why:**
- **Estimated effort:**

### Additional Features
- [Feature idea 1]
- [Feature idea 2]

### Testing & Validation
- [What additional tests you'd write]
- [What validation you'd add]

---

## ❓ Questions for the Team

List any clarifying questions or areas where you'd like feedback.

1. **[Question about requirements]:**  Should data be standardized to a base unit (e.g., SI units) during ingestion, or is it expected to be consumed in its source unit?

2. **[Question about design]:** [e.g., "Would you prefer aggressive duplicate removal or conservative approach?"]

3. **[Technical question]:**  For the rolling Z-score method, what is the recommended default window size in a production environment?

---

## 💡 Interesting Challenges

What did you find most interesting or challenging about this exercise?

- **Most challenging:** Getting the Notes in time 
- **Most interesting:** [What you enjoyed working on]
- **Learned:** Used the IQR for the first time 

---

## 🔧 Development Environment

Document your setup for reproducibility.

- **Python version:** 3.10.12
- **OS:** Ubuntu (WSL2)
- **Editor/IDE:** VS Code
- **Additional tools:** 

---

## 📚 References

Any resources you consulted (documentation, articles, etc.).

- Z-score: (https://de.wikipedia.org/wiki/Standardisierung_(Statistik))
- IQR: https://novustat.com/statistik-glossar/interquartilabstand.html
- Pandas Documentation (GroupBy, Aggregation, TimeGrouper)

---

## 💭 Final Thoughts

Any additional context you want reviewers to know.

This was a nice coding challange, I hope hearing back from you!

---

**Thank you for the opportunity!** I look forward to discussing this implementation.
