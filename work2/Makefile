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


## Install Python Dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

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
	python -m venv ${ENV_NAME} && \
	echo ">>> New virtualenv created. Activate with:\n.venv\Scripts\activate"
else
	python3 -m venv ${ENV_NAME}
	echo ">>> New virtualenv created. Activate with:\n.venv\Scripts\activate"
endif


## Complete project setup from fresh clone
.PHONY: setup

# Define activation command based on OS
ifeq ($(OS),Windows_NT)
    VENV_DIR = .venv\Scripts
    ACTIVATE_CMD = call $(VENV_DIR)\activate.bat &&
else
    VENV_DIR = .venv/bin
    ACTIVATE_CMD = . $(VENV_DIR)/activate &&

endif

setup: create_environment
	@echo ">>> Setting up complete project environment..."
	$(ACTIVATE_CMD) $(MAKE) requirements && $(MAKE) format && echo ">>> Setup complete. Run base project with: python run.py --dataset_name [mushroom|hepatitis]"

.PHONY: run

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
