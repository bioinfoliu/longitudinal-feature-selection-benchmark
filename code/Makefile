PYTHON ?= python3

.PHONY: test figures biological external simulation benchmark secondary final-benchmark manuscript

test:
	$(PYTHON) -m unittest discover -s tests -v

benchmark:
	$(PYTHON) src/run_expanded_benchmark.py --datasets PE_LOPE_MATCHED --repeats 3 --methods Lasso Stabl-RP LongGroupLasso LLSS SNN STAR Hybrid CTSNN-R --output-dir results/main_benchmark/benchmark_full50
	$(PYTHON) src/summarize_expanded_results.py --input-dir results/main_benchmark/benchmark_full50

simulation:
	$(PYTHON) src/run_simulation_study.py --repeats 8

external:
	$(PYTHON) src/prepare_figshare_pe_multicohort.py
	$(PYTHON) src/run_multicohort_external_validation.py
	$(PYTHON) src/summarize_multicohort_external.py

secondary:
	$(PYTHON) src/run_expanded_benchmark.py --datasets FPG IGG IGM TRA TRB --repeats 2 --methods Lasso Stabl-RP LongGroupLasso LLSS SNN STAR Hybrid CTSNN-R --output-dir results/main_benchmark/benchmark_with_new
	$(PYTHON) src/summarize_secondary_longitudinal.py

biological:
	$(PYTHON) src/run_biological_validation.py
	$(PYTHON) src/fetch_string_ppi.py

figures:
	MPLBACKEND=Agg $(PYTHON) src/make_publication_figures.py

final-benchmark:
	$(PYTHON) src/build_final_benchmark.py

manuscript:
	cd manuscript && pdflatex -interaction=nonstopmode -halt-on-error ctsnn_longitudinal_fs.tex
	cd manuscript && bibtex ctsnn_longitudinal_fs
	cd manuscript && pdflatex -interaction=nonstopmode -halt-on-error ctsnn_longitudinal_fs.tex
	cd manuscript && pdflatex -interaction=nonstopmode -halt-on-error ctsnn_longitudinal_fs.tex
	cp manuscript/ctsnn_longitudinal_fs.pdf output/pdf/CTSNN_longitudinal_feature_selection_manuscript.pdf
