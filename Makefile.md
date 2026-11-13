VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

VLLM = data/raw/vllm-0.10.1

REQUIREMENTS = requirements.txt

MYPY_FLAGS = --strict

UV_CACHE_DIR = $(CURDIR)/.uv_cache

export UV_CACHE_DIR
export UV_LINK_MODE = copy

FILE_TRASH = data/output/search_results \
			data/output/search_results_and_answer \
			data/processed \
			data/datasets/private \
			evaluations
TRASH = src/__pycache__ $(VENV) .mypy_cache __pycache__ $(CACHE_DIR) \
		$(UV_CACHE_DIR)


# ANSI codes
GREEN = \033[0;32m
RED = \033[0;31m
YELLOW = \033[0;33m
BOLD = \033[1m

SUCCESS = \033[1;92m
RESET = \033[0m

install:
	@echo "$(BOLD)Initializing project environment:$(RESET)"

	@echo "$(YELLOW)Creating virtual environment...$(RESET)"
	@uv venv $(VENV) --clear > /dev/null 2>&1
	@echo "$(GREEN)Virtual environment setup complete$(RESET)"

	@echo "\n$(YELLOW)Installing dependencies...$(RESET)"
	@uv sync > /dev/null 2>&1
	@echo "$(GREEN)All dependencies successfully installed$(RESET)"

	@echo "\n$(SUCCESS)Project can now be run safely!$(RESET)"

vllm_compile:
	@echo "$(BOLD)Compiling vLLM...$(RESET)"
	@uv pip install setuptools setuptools-scm wheel numpy
	@SETUPTOOLS_SCM_PRETEND_VERSION=0.10.1 uv pip install -e data/raw/vllm-0.10.1 --no-build-isolation
	@echo "$(GREEN)vLLM compilation complete!$(RESET)"

run:
	@uv run python -m src

pipeline:
	@echo "$(GREEN)Indexing$(RESET)"
	@uv run python -m src index --max_chunk_size 2000
	@echo "$(GREEN)Searching best chunk for .py files$(RESET)"
	@uv run python -m src search_dataset \
		--dataset_path data/datasets/public/UnansweredQuestions/dataset_code_public.json \
		--k 10 \
		--save_directory data/output/search_results
	@echo "$(GREEN)Searching best chunk for .md files$(RESET)"
	@uv run python -m src search_dataset \
		--dataset_path data/datasets/public/UnansweredQuestions/dataset_docs_public.json \
		--k 10 \
		--save_directory data/output/search_results
	@echo "$(GREEN)Get answer for .py files$(RESET)"
	@uv run python -m src answer_dataset \
		--student_search_results_path data/output/search_results/dataset_code_public.json \
		--save_directory data/output/search_results_and_answer
	@echo "$(GREEN)Get answer for .md files$(RESET)"
	@uv run python -m src answer_dataset \
		--student_search_results_path data/output/search_results/dataset_docs_public.json \
		--save_directory data/output/search_results_and_answer
	@echo "$(GREEN)Evaluate .py files$(RESET)"
	@uv run python -m src evaluate data/output/search_results/dataset_code_public.json \
	data/datasets/public/AnsweredQuestions/dataset_code_public.json
	@echo "$(GREEN)Evaluate .md files$(RESET)"
	@uv run python -m src evaluate data/output/search_results/dataset_docs_public.json \
	data/datasets/public/AnsweredQuestions/dataset_docs_public.json

lint:
	@echo "$(BOLD)Verifying project source code:$(RESET)"

	@echo "$(YELLOW)Running flake8...$(RESET)"
	@$(PYTHON) -m flake8 src
	@echo "$(GREEN)No norm errors found$(RESET)"

	@echo "\n$(YELLOW)Running mypy...$(RESET)"
	@$(PYTHON) -m mypy src $(MYPY_FLAGS)
	@echo "$(GREEN)No typing errors found$(RESET)"

	@echo "\n$(SUCCESS)Success: no errors found in source code!$(RESET)"

debug:
	@$(PYTHON) -m pdb src

clean_files:
	@echo "$(YELLOW)Initiating cleanup files operation...$(RESET)"
	@found=0; \
	for i in $(FILE_TRASH); do \
		if [ -d "$$i" ]; then \
			echo " - Removed directory $(RED)'$$i'$(RESET)"; \
			found=1; \
		elif [ -e "$$i" ]; then \
			echo " - Removed file $(RED)'$$i'$(RESET)"; \
			found=1; \
		fi; \
	done; \
	if [ $$found -eq 0 ]; then \
		echo "$(GREEN)Nothing to clean :)$(RESET)"; \
	fi
	@rm -rf $(FILE_TRASH)
	@echo "$(SUCCESS)\nCleanup complete!$(RESET)"

clean:
	@echo "$(YELLOW)Initiating cleanup operation...$(RESET)"
	@found=0; \
	for i in $(TRASH); do \
		if [ -d "$$i" ]; then \
			echo " - Removed directory $(RED)'$$i'$(RESET)"; \
			found=1; \
		elif [ -e "$$i" ]; then \
			echo " - Removed file $(RED)'$$i'$(RESET)"; \
			found=1; \
		fi; \
	done; \
	if [ $$found -eq 0 ]; then \
		echo "$(GREEN)Nothing to clean :)$(RESET)"; \
	fi
	@rm -rf $(TRASH)
	@echo "$(SUCCESS)\nCleanup complete!$(RESET)"

fclean: clean_files clean