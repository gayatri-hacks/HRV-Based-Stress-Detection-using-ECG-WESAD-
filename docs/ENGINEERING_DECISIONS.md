# Engineering decisions

| Decision | Rationale |
| --- | --- |
| Accept only 700 Hz | The model was trained on WESAD chest ECG at 700 Hz; resampling or other rates have not been validated. |
| Cap bytes and samples | Prevents upload and CPU exhaustion while allowing configuration by deployment. |
| Lazy-load trusted artifacts only | Keeps startup light and never deserializes user input. |
| Validate feature order and classes | A matching feature count alone cannot prevent silent semantic mismatch. |
| Keep final-valid-window inference | It preserves existing model semantics; averaging would be a new, unvalidated decision rule. |
| Keep real-data smoke test opt-in | WESAD is licensed and must not be copied into CI. |
