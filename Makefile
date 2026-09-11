ifeq ($(OS),Windows_NT)
VENV_PYTHON := .venv/Scripts/python.exe
else
VENV_PYTHON := .venv/bin/python
endif

bunbot:
	$(VENV_PYTHON) main.py

stream:
	$(VENV_PYTHON) tests/test_files/stream_ts.py $(filter-out $@,$(MAKECMDGOALS))

latency:
	$(VENV_PYTHON) tests/test_files/stream_latency.py $(filter-out $@,$(MAKECMDGOALS))

pacing:
	$(VENV_PYTHON) -u tests/test_files/hist_pacing.py $(filter-out $@,$(MAKECMDGOALS))

# make eats any goal starting with `--` as one of its own options, so the bare
# goal `no-api` is accepted as an alias: `make backtest today no-api`.
backtest:
	$(VENV_PYTHON) -m exec.backtest $(patsubst no-api,--no-api,$(filter-out $@,$(MAKECMDGOALS)))

visuals:
	$(VENV_PYTHON) tests/visuals/charts.py $(filter-out $@,$(MAKECMDGOALS))

%:
	@:
