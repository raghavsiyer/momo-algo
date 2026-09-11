# Task Completion Checklist

When a coding task is completed, run through these steps:

## 1. Tests Pass
```bash
python -m pytest
```
All tests must pass. No network calls in tests — everything is mocked.

## 2. Type Hints
- All new/modified function signatures must have type hints
- Use `Optional[T]` for nullable params, not `T | None` (project convention)

## 3. Feature Flag Compatibility
- If touching a data provider, ensure the code degrades gracefully when the provider's feature flag is `False`
- Test with `TRADIER_ENABLED=False`, etc. mentally or via config

## 4. Threading Safety
- Any new shared state must be protected by `threading.Lock`
- New background work should use daemon threads
- Stop signals should use `threading.Event`

## 5. No Hardcoded Keys
- API keys come from environment only
- Never commit `.env`

## 6. Logging
- Use module-level `logger` (not `print` for diagnostics)
- Prefix with `[SYMBOL]` where applicable

## 7. Constants
- New thresholds/config values go in `config.py` as `UPPER_SNAKE_CASE`
- Don't scatter magic numbers in business logic

## 8. Cache Consistency
- If changing `WatchlistEntry` or `ResistanceMap` fields, ensure `TickerCache` serialization still works (JSON-backed)
