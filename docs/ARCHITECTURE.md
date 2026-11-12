# Architecture

`React/Vite UI → multipart FastAPI endpoint → CSV validation → 60 s ECG windows → R peaks/RR intervals → nine HRV features → saved scaler → saved logistic regression`.

The UI submits only a CSV and explicit sampling rate. The API accepts only 700 Hz because that is the rate recorded in the training configuration. Raw WESAD files remain external and licensed; committed artifacts are derived model/configuration files.

`/health` is a liveness probe. `/ready` is a model-readiness probe: it distinguishes artifact presence from successful deserialization and compatibility validation. The predictor evaluates all usable overlapping windows and scores the final valid one, matching the prior deployed behavior. Response metadata exposes that choice.
