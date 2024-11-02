#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = mai-iml-work2
PYTHON_VERSION = 3.9
PYTHON_INTERPRETER = python
ENV_NAME = .venv

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## run all models and analyses
.PHONY: all
all: run_models run_analysis

## Delete all compiled Python files
.PHONY: clean

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Clean output directories
.PHONY: clean-outputs
clean-outputs:
	rm -f data/cross_validated_results/*
	rm -f data/per_fold_results/*
	rm -f data/reduced_X/*
	rm -f data/reduced_y/*
	rm -f reports/figures/*
	rm -f reports/tables/*


## Lint using flake8 and black (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 tools
	isort --check --diff --profile black tools
	black --check --config pyproject.toml tools

## Format source code with black
.PHONY: format
format:
	black --config pyproject.toml tools


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

.PHONY: run_models
run_models:
	@echo ">>> Running both datasets..."
	
	@echo "\n>>> Running hepatitis dataset..."
	${PYTHON_INTERPRETER} tools/run.py --dataset_name hepatitis
	
	@echo "\n>>> Running mushroom dataset..."
	${PYTHON_INTERPRETER} tools/run.py --dataset_name mushroom

	@echo ">>> Completed running all datasets."


.PHONY: run_analysis
run_analysis:
	@echo ">>> Running hepatitis statistical analysis..."
	${PYTHON_INTERPRETER} tools/analysis/run_statistical_analysis.py --dataset_name hepatitis
	
	@echo "\n>>> Running mushroom statistical analysis..."
	${PYTHON_INTERPRETER} tools/analysis/run_statistical_analysis.py --dataset_name mushroom
	
	@echo "\n>>> Running dataset analysis..."
	${PYTHON_INTERPRETER} tools/analysis/run_dataset_analysis.py

	@echo ">>> Completed running all analyses."

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
