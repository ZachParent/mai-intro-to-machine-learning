#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = mai-iml-work3
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
all: run-preprocessing run-models run-metrics run-analysis

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Clean output directories
.PHONY: clean-outputs
clean-outputs:
	rm -rf data/2_clustered/*
	rm -rf data/3_metrics/*

.PHONY: run-preprocessing
run-preprocessing:
	@echo ">>> Running preprocessing for all datasets..."
	$(PYTHON_INTERPRETER) src/1_run_preprocessing.py

## Run all models for all datasets
.PHONY: run-models
run-models:
	for dataset in hepatitis vowel mushroom; do \
		for model in kmeans fuzzy_cmeans gmeans global_kmeans optics spectral_clustering; do \
			python src/2_run_model.py --dataset=$$dataset --model=$$model --verbose; \
		done; \
	done

## Calculate metrics for all clustered datasets
.PHONY: run-metrics
run-metrics:
	@echo ">>> Calculating metrics for all clustered datasets..."
	$(PYTHON_INTERPRETER) src/3_run_metrics.py --verbose

## Perform statistical analysis on results
.PHONY: run-analysis
run-analysis:
	@echo ">>> Performing statistical analysis on results..."
	$(PYTHON_INTERPRETER) src/4_run_analysis.py --verbose

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
