.PHONY: help install-uv setup list clean tests format
.ONESHELL:

help:   ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

install-uv:  ## Install uv package manager
	curl -LsSf https://astral.sh/uv/install.sh | sh

setup:  ## Sets up everything needed for a new deployment
	uv sync --all-extras

tests: ## Run the unit tests
	uv run --extra dev pytest tests/