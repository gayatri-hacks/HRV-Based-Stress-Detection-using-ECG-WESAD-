# Interview walkthrough

Start with the problem boundary: this is research software, not a medical device. Explain that WESAD chest ECG is converted into RR intervals and interpretable HRV features in overlapping 60-second windows, then passed through a saved scaler and logistic regression model.

Emphasize data integrity: contiguous labelled WESAD segments, subject-disjoint train/validation/test splits, training-only scaler fitting, and no tuning against test subjects. The committed test metrics are traceable to the configuration and evaluation artifact.

Walk through reliability controls: size/sample limits, strict CSV/finite-value checks, 700 Hz contract enforcement, safe invalid-peak handling, and 413/415/422/503 failures. Contrast `/health` with `/ready`, which verifies artifact loadability, shape/order/classes, and finite probabilities.

Finish with frontend integration: FormData includes the user-confirmed sampling rate, response schema validation prevents malformed output from being rendered, requests time out, submissions are disabled while active, and the UI clearly labels probability versus predicted class. State the key limitation honestly: final-valid-window inference is preserved for compatibility and requires further research before aggregation or clinical interpretation.
