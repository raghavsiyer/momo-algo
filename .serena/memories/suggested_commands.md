# Suggested Commands

## Running the Application
```bash
python main.py          # start the CLI REPL (venv auto-activated)
make bunbot             # same via Makefile
make visuals AZI        # render default 10s/1m/5m local premarket charts
make visuals AZI 2m     # render one selected timeframe
```

## Testing
```bash
python -m pytest                    # run all tests
python -m pytest tests/test_eval.py # run a specific test file
python -m pytest -k "test_name"     # run tests matching pattern
python -m pytest -x                 # stop on first failure
```

## Package Management
```bash
pip install -r requirements.txt     # install all dependencies
pip install <package>               # install a new package
```

## Git (macOS/Darwin)
```bash
git status
git add -A && git commit -m "message"
git log --oneline -20
git diff
```

## File Navigation (macOS/Darwin)
```bash
ls -la
find . -name "*.py" -not -path "./.venv/*"
grep -rn "pattern" --include="*.py" .
```

## CLI REPL Commands (inside `main.py`)
- `TICKER` — evaluate a ticker (e.g. `AAPL`)
- `list` — print watchlist
- `cache` — show cached tickers
- `cache clear` — clear the ticker cache
- `cache TICKER` — show cached ticker details
- `TICKER refresh` — re-run pipeline (skip cache)
- `TICKER find` — show when ticker was first found
- `help` — show help text
- `momo stop/go/fast/def/slow/poll/interval N` — control momo scanner
- `yt stop/go/fast/def/slow/auto/poll/interval N` — control YouTube scanner
- `q` / `quit` / `exit` — exit

## Serena
```bash
serena project index    # re-index the project for Serena
```
