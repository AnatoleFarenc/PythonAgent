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

run:
	@uv run python -m src

# --- Coding assistant -------------------------------------------------------
# Every value can be changed on the command line, e.g.
#   make serve LLM_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct-AWQ
#   make assistant PROVIDER=groq ROOT=../my_project

# Local model served by vLLM (make serve)
LLM_MODEL ?= Qwen/Qwen2.5-Coder-3B-Instruct-AWQ
LLM_CONTEXT ?= 16384
LLM_GPU ?= 0.85
LLM_PORT ?= 8000

# Assistant (make assistant)
PROVIDER ?= local
ROOT ?= .
ARGS ?=

# FlashInfer's sampler does not compile with nvcc 12.0: use vLLM's own.
serve:
	@echo "$(BOLD)Serving $(LLM_MODEL) on port $(LLM_PORT)$(RESET)"
	@echo "$(YELLOW)Ready when 'Application startup complete' shows up$(RESET)"
	@VLLM_USE_FLASHINFER_SAMPLER=0 uv run vllm serve $(LLM_MODEL) \
		--max-model-len $(LLM_CONTEXT) \
		--gpu-memory-utilization $(LLM_GPU) \
		--enforce-eager \
		--port $(LLM_PORT)

# LOCAL_LLM_* tell the assistant where the local server is and how big its
# context is (must match --max-model-len of make serve).
assistant:
	@LOCAL_LLM_URL=http://localhost:$(LLM_PORT)/v1 \
	LOCAL_LLM_CONTEXT=$(LLM_CONTEXT) \
	uv run python -m src.agent --provider $(PROVIDER) --root $(ROOT) $(ARGS)

assistant-api:
	@$(MAKE) --no-print-directory -f $(firstword $(MAKEFILE_LIST)) \
		assistant PROVIDER=groq

# First full indexing on GPU (stop make serve before: the GPU is too small
# for both). The assistant then reuses the cache and only re-encodes the
# modified files, on CPU. Use the same ROOT as for make assistant.
index:
	@echo "$(BOLD)Indexing $(ROOT) on GPU$(RESET)"
	@AMBEDING_DEVICE=cuda uv run python -c "\
	from src.rag.config import AMBEDING_MODEL; \
	from src.rag.indexer.HybridIndex import HybridIndex; \
	HybridIndex.from_folder('$(ROOT)', AMBEDING_MODEL)"
	@echo "$(GREEN)Index ready in data/chunk$(RESET)"

test:
	@uv run python -m pytest -q

.PHONY: install vllm_compile run serve assistant assistant-api index test lint \
	debug clean_files clean fclean


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