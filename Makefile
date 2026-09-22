PYTHON ?= python3
RESULTS_ROOT ?= ../results

.PHONY: install smoke test publication full-results panel-sensitivity clean

install:
	$(PYTHON) -m pip install -e ".[test]"

smoke:
	$(PYTHON) examples/run_smoke_test.py

test:
	$(PYTHON) -m unittest discover -s tests -v

publication:
	MPLBACKEND=Agg $(PYTHON) src/regenerate_publication_outputs.py \
		--input-dir results/summary --output-dir results/reproduced

full-results:
	MPLBACKEND=Agg $(PYTHON) src/build_final_benchmark.py \
		--output-dir results/reproduced_full

panel-sensitivity:
	MPLBACKEND=Agg $(PYTHON) src/analyze_panel_sensitivity.py \
		--results-root $(RESULTS_ROOT) --output-dir results/panel_sensitivity

clean:
	rm -rf build dist .pytest_cache src/*.egg-info results/reproduced

