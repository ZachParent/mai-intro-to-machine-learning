#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = mai-iml-work4
PYTHON_VERSION = 3.9
PYTHON_INTERPRETER = python
ENV_NAME = .venv

PROJECT_DIR = $(shell pwd)
SUBMISSION_NAME = ${PROJECT_NAME}-KacperPoniatowski-ZacharyParent-SheenaLang-CarlosJimenezFarfan
ZIP_FILE_NAME = $(SUBMISSION_NAME).zip

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Run all scripts
.PHONY: all
all: run-preprocessing run-dimensionality-reduction run-clustering-non-reduced run-clustering-reduced run-metrics run-plot-metrics run-visualizations

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Clean output directories
.PHONY: clean-outputs
clean-outputs:
	rm -rf data/1_preprocessed/*
	rm -rf data/2_reduced/*
	rm -rf data/3_clustered/*
	rm -rf data/4_metrics.csv
	rm -rf data/5_metrics_plots/*
	rm -rf data/6_visualizations/*

.PHONY: run-preprocessing
run-preprocessing:
	@echo ">>> Running preprocessing for all datasets..."
	$(PYTHON_INTERPRETER) src/1_run_preprocessing.py

.PHONY: run-dimensionality-reduction
run-dimensionality-reduction:
	@echo ">>> Running dimensionality reduction for all datasets..."
	for dataset in hepatitis vowel mushroom; do \
		for method in pca kernel_pca; do \
			python src/2_run_dimensionality_reduction.py --dataset=$$dataset --method=$$method --verbose; \
		done; \
	done

## Run clustering for the non-reduced datasets
.PHONY: run-clustering-non-reduced
run-clustering-non-reduced:
	for filename in $$(find data/1_preprocessed -type f -name "*.csv"); do \
		for model in global_kmeans optics; do \
			python src/3_run_clustering.py --input_file_path=$$filename --model=$$model --verbose; \
		done; \
	done

## Run clustering for the reduced datasets
.PHONY: run-clustering-reduced
run-clustering-reduced:
	for filename in $$(find data/2_reduced -type f -name "*.csv"); do \
		for model in global_kmeans optics; do \
			python src/3_run_clustering.py --reduced --input_file_path=$$filename --model=$$model --verbose; \
		done; \
	done

## Calculate metrics for all clustered datasets
.PHONY: run-metrics
run-metrics:
	@echo ">>> Calculating metrics for all clustered datasets..."
	$(PYTHON_INTERPRETER) src/4_run_metrics.py --verbose

## Plot the computed metrics
.PHONY: run-plot-metrics
run-plot-metrics:
	@echo ">>> Plotting metrics..."
	$(PYTHON_INTERPRETER) src/5_run_plot_metrics.py --verbose

## Calculate and plot visualizations
.PHONY: run-visualizations
run-visualizations:
	@echo ">>> Calculating and plotting visualizations..."
	$(PYTHON_INTERPRETER) src/6_run_visualizations.py --verbose

## Lint using flake8 and black (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 src
	isort --check --diff --profile black src
	black --check --config pyproject.toml src

## Format source code with black
.PHONY: format
format:
	black --config pyproject.toml src


.PHONY: create_environment
create_environment:
ifeq ($(OS),Windows_NT)
	${PYTHON_INTERPRETER} -m venv ${ENV_NAME}
	@echo ">>> New virtualenv created. Activate it manually with:"
	@echo "    .\.venv\Scripts\activate"
else
	${PYTHON_INTERPRETER} -m venv ${ENV_NAME}
	@echo ">>> New virtualenv created. Activate it manually with:"
	@echo "    source .venv/bin/activate"
endif
	@echo ">>> Remember to install the requirements with 'make install_requirements'"

## Install Python Dependencies
.PHONY: install_requirements
install_requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Create a zip archive of the project (excluding .venv and other unnecessary files)
.PHONY: zip
zip: clean
	rm -f $(ZIP_FILE_NAME)
	(cd .. && \
	mkdir -p $(SUBMISSION_NAME) && \
	cp -r $(PROJECT_DIR)/* $(SUBMISSION_NAME)/ && \
	cp $(SUBMISSION_NAME)/report/report.pdf $(SUBMISSION_NAME)/$(PROJECT_NAME)-report.pdf && \
	zip -r $(ZIP_FILE_NAME) $(SUBMISSION_NAME) -x "*.pyc" -x "*.pyo" -x "*.pyd" -x "*.so" -x "*.dll" -x "*.dylib" \
	-x "*/__pycache__/*" -x "*/$(ENV_NAME)/*" -x "*/.git/*" -x "*.zip" -x "*.egg-info/*" \
	-x "*.aux" -x "*.log" -x "*.toc" -x "*.out" -x "*.synctex.gz" -x "*.fls" -x "*.fdb_latexmk" -x "*.bbl" -x "*.blg" \
	-x '**/2_clustered/*' && \
	mv $(ZIP_FILE_NAME) $(PROJECT_DIR)/ && \
	rm -rf $(SUBMISSION_NAME))
	@echo ">>> Created $(ZIP_FILE_NAME)"

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
