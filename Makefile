.DEFAULT_GOAL := help

.PHONY: lint
lint: ## Run flake8 linter
	flake8

.PHONY: format
format: ## Run black formatter
	black .

.PHONY: test
test: ## Run tests
	pytest -v

.PHONY: coverage
coverage: ## Run test coverage
	coverage erase
	coverage run -m pytest -v
	coverage report

.PHONY: install
install: ## Install package in editable mode
	pip install -e .

.PHONY: help
help: ## Show help message
		@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m\033[0m\n"} /^[$$()% 0-9a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
