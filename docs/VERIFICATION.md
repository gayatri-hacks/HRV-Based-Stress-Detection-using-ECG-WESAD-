# Verification

Ordinary CI runs Python API/unit tests plus frontend lint, TypeScript checking, Vitest, and production build. It does not require WESAD data.

Backend tests cover liveness/readiness, CSV/media/rate validation, short, nonfinite and constant signals, model artifact readiness, probability confidence, and final-valid-window metadata. Frontend tests cover successful multipart submission; additional UI behavior is guarded by schema validation, explicit 700 Hz input, timeout cancellation, and disabled duplicate submission.

Run `pytest -m 'not integration' -q` and the frontend command sequence in the README. The opt-in `integration` marker requires `WESAD_ECG_SMOKE_CSV` to reference an authorized, extracted genuine WESAD CSV.
