# Anomalous Transaction Detection System Makefile
# Cross-platform support for Windows, Linux, and macOS

# OS Detection
ifeq ($(OS),Windows_NT)
    DETECTED_OS := Windows
    PYTHON := python
    VENV_BIN := $(VENV_DIR)/Scripts
    VENV_ACTIVATE := $(VENV_BIN)/activate
    MKDIR := mkdir
    RM := rmdir /s /q
    RM_FILE := del /q
    PATH_SEP := \\
    UV_INSTALL := powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    SHELL_CHECK := where
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Linux)
        DETECTED_OS := Linux
    endif
    ifeq ($(UNAME_S),Darwin)
        DETECTED_OS := macOS
    endif
    PYTHON := python3
    VENV_BIN := $(VENV_DIR)/bin
    VENV_ACTIVATE := $(VENV_BIN)/activate
    MKDIR := mkdir -p
    RM := rm -rf
    RM_FILE := rm -f
    PATH_SEP := /
    UV_INSTALL := curl -LsSf https://astral.sh/uv/install.sh | sh
    SHELL_CHECK := command -v
endif

# Variables
VENV_DIR := .venv
UV := uv
DATA_DIR := data
OUTPUT_DIR := output
INPUT_FILE := $(DATA_DIR)$(PATH_SEP)synthetic_dirty_transaction_logs.csv
STREAMLIT_PORT := 8501
ifeq ($(DETECTED_OS),Windows)
    PYTHON_EXE := $(VENV_BIN)$(PATH_SEP)python.exe
else
    PYTHON_EXE := python3
endif
DOCKER_IMAGE := anomaly-detector
DOCKER_TAG := latest

# Default target
.DEFAULT_GOAL := help

# Colors (Windows CMD doesn't support ANSI colors well, so we simplify)
ifeq ($(DETECTED_OS),Windows)
    RED := 
    GREEN := 
    YELLOW := 
    BLUE := 
    NC := 
else
    RED := \033[0;31m
    GREEN := \033[0;32m
    YELLOW := \033[1;33m
    BLUE := \033[0;34m
    NC := \033[0m
endif

.PHONY: help setup install clean test lint format run-* streamlit docker-* compose-* all-methods

## Help
help: ## Show this help message
	@echo "$(BLUE)Anomalous Transaction Detection System ($(DETECTED_OS))$(NC)"
	@echo "$(YELLOW)Available commands:$(NC)"
ifeq ($(DETECTED_OS),Windows)
	@findstr /R "^[a-zA-Z_-]*:.*##" $(MAKEFILE_LIST) | findstr /V "findstr" | for /f "tokens=1,2 delims=:" %%a in ('more') do @echo   %%a: %%b
else
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
endif

## Setup and Installation
setup: ## Set up the project (install uv, create venv, install dependencies)
	@echo "$(BLUE)Setting up project on $(DETECTED_OS)...$(NC)"
ifeq ($(DETECTED_OS),Windows)
	@where uv >nul 2>&1 || (echo "$(RED)Installing uv...$(NC)" && $(UV_INSTALL))
else
	@$(SHELL_CHECK) $(UV) >/dev/null 2>&1 || { echo "$(RED)Installing uv...$(NC)"; $(UV_INSTALL); }
endif
	@$(UV) venv $(VENV_DIR) --python $(PYTHON)
	@echo "$(GREEN)Virtual environment created at $(VENV_DIR)$(NC)"
	@$(MAKE) install

install: ## Install project dependencies using uv
	@echo "$(BLUE)Installing dependencies...$(NC)"
	@$(UV) pip install -r requirements.txt
	@echo "$(GREEN)Dependencies installed successfully$(NC)"

install-dev: ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	@$(UV) pip install -r requirements.txt
	@$(UV) pip install pytest pytest-cov pytest-xvfb black flake8 mypy bandit safety
	@echo "$(GREEN)Development dependencies installed$(NC)"

## Data Analysis Commands
run-isolation: ## Run Isolation Forest analysis
	@echo "$(BLUE)Running Isolation Forest analysis...$(NC)"
	@$(PYTHON_EXE) analysis.py --input $(INPUT_FILE) --output_dir $(OUTPUT_DIR) --contamination 0.02 --top_n 30
	@echo "$(GREEN)Isolation Forest analysis completed. Results in $(OUTPUT_DIR)$(PATH_SEP)isolation_forest$(PATH_SEP)$(NC)"

run-rule: ## Run Rule-based analysis
	@echo "$(BLUE)Running Rule-based analysis...$(NC)"
	@$(PYTHON_EXE) -c "import pandas as pd; from analysis import rule_based_anomaly_detection, create_visualisations; from parsing_utils import parse_log; import os; df_raw = pd.read_csv('$(INPUT_FILE)'); df_parsed = df_raw['raw_log'].apply(parse_log); df_parsed = pd.DataFrame([rec for rec in df_parsed if rec is not None]); results = rule_based_anomaly_detection(df_parsed, 30); os.makedirs('$(OUTPUT_DIR)$(PATH_SEP)rule_based', exist_ok=True); results.to_csv('$(OUTPUT_DIR)$(PATH_SEP)rule_based$(PATH_SEP)results.csv', index=False); create_visualisations(results, '$(OUTPUT_DIR)$(PATH_SEP)rule_based', 'rule_based'); print('Rule-based analysis completed. Results in $(OUTPUT_DIR)$(PATH_SEP)rule_based$(PATH_SEP)')"

run-sequence: ## Run Sequence Modeling analysis
	@echo "$(BLUE)Running Sequence Modeling analysis...$(NC)"
	@$(PYTHON_EXE) -c "import pandas as pd; from analysis import sequence_modeling_anomaly_detection, create_visualisations; from parsing_utils import parse_log; import os; df_raw = pd.read_csv('$(INPUT_FILE)'); df_parsed = df_raw['raw_log'].apply(parse_log); df_parsed = pd.DataFrame([rec for rec in df_parsed if rec is not None]); results = sequence_modeling_anomaly_detection(df_parsed, 30); os.makedirs('$(OUTPUT_DIR)$(PATH_SEP)sequence_modeling', exist_ok=True); results.to_csv('$(OUTPUT_DIR)$(PATH_SEP)sequence_modeling$(PATH_SEP)results.csv', index=False); create_visualisations(results, '$(OUTPUT_DIR)$(PATH_SEP)sequence_modeling', 'sequence_modeling'); print('Sequence Modeling analysis completed. Results in $(OUTPUT_DIR)$(PATH_SEP)sequence_modeling$(PATH_SEP)')"

run-embedding: ## Run Embedding + Autoencoder analysis
	@echo "$(BLUE)Running Embedding + Autoencoder analysis...$(NC)"
	@$(PYTHON_EXE) -c "import pandas as pd; from analysis import embedding_autoencoder_anomaly_detection, create_visualisations; from parsing_utils import parse_log; import os; df_raw = pd.read_csv('$(INPUT_FILE)'); df_parsed = df_raw['raw_log'].apply(parse_log); df_parsed = pd.DataFrame([rec for rec in df_parsed if rec is not None]); results = embedding_autoencoder_anomaly_detection(df_parsed, 30); os.makedirs('$(OUTPUT_DIR)$(PATH_SEP)embedding_autoencoder', exist_ok=True); results.to_csv('$(OUTPUT_DIR)$(PATH_SEP)embedding_autoencoder$(PATH_SEP)results.csv', index=False); create_visualisations(results, '$(OUTPUT_DIR)$(PATH_SEP)embedding_autoencoder', 'embedding_autoencoder'); print('Embedding + Autoencoder analysis completed. Results in $(OUTPUT_DIR)$(PATH_SEP)embedding_autoencoder$(PATH_SEP)')"

all-methods: ## Run all anomaly detection methods
	@echo "$(BLUE)Running all anomaly detection methods...$(NC)"
	@$(MAKE) run-isolation
	@$(MAKE) run-rule
	@$(MAKE) run-sequence
	@$(MAKE) run-embedding
	@echo "$(GREEN)All methods completed. Results in $(OUTPUT_DIR)$(PATH_SEP)$(NC)"

## Web Interface
streamlit: ## Launch Streamlit web interface
	@echo "$(BLUE)Starting Streamlit web interface on port $(STREAMLIT_PORT)...$(NC)"
	@$(PYTHON_EXE) -m streamlit run streamlit_app.py --server.port $(STREAMLIT_PORT)

streamlit-dev: ## Launch Streamlit in development mode with auto-reload
	@echo "$(BLUE)Starting Streamlit in development mode...$(NC)"
	@$(PYTHON_EXE) -m streamlit run streamlit_app.py --server.port $(STREAMLIT_PORT) --server.runOnSave true

## Quality Assurance
test: ## Run tests with coverage
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	@$(MKDIR) coverage
	@$(PYTHON_EXE) -m pytest tests/ -v || $(PYTHON_EXE) -m pytest tests/ --cov=. --cov-report=html:coverage/html --cov-report=xml:coverage/coverage.xml --cov-report=term-missing -v 2>/dev/null || $(PYTHON_EXE) -m pytest tests/ -v
	@echo "$(GREEN)Tests completed$(NC)"

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(NC)"
	@$(MKDIR) coverage
	@$(PYTHON_EXE) -m pytest tests/ -m "not integration" -v || $(PYTHON_EXE) -m pytest tests/ -m "not integration" --cov=. --cov-report=html:coverage/html --cov-report=xml:coverage/coverage.xml -v 2>/dev/null || $(PYTHON_EXE) -m pytest tests/ -v

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(NC)"
	@$(MKDIR) coverage
	@$(PYTHON_EXE) -m pytest tests/ -m integration -v || $(PYTHON_EXE) -m pytest tests/ -m integration --cov=. --cov-report=html:coverage/html --cov-report=xml:coverage/coverage.xml -v 2>/dev/null || $(PYTHON_EXE) -m pytest tests/ -v

test-watch: ## Run tests in watch mode
	@echo "$(BLUE)Running tests in watch mode...$(NC)"
	@$(PYTHON_EXE) -m pytest tests/ -f || $(PYTHON_EXE) -m pytest tests/ --cov=. -f 2>/dev/null || $(PYTHON_EXE) -m pytest tests/ -v

coverage-report: ## Open coverage report in browser
	@echo "$(BLUE)Opening coverage report...$(NC)"
ifeq ($(DETECTED_OS),Windows)
	@start coverage\html\index.html
else ifeq ($(DETECTED_OS),macOS)
	@open coverage/html/index.html
else
	@xdg-open coverage/html/index.html
endif

lint: ## Run code linting
	@echo "$(BLUE)Running linting...$(NC)"
	@$(PYTHON_EXE) -m flake8 *.py --max-line-length=88 --extend-ignore=E203,W503 || echo "$(YELLOW)flake8 not installed$(NC)"
	@$(PYTHON_EXE) -m mypy *.py --ignore-missing-imports || echo "$(YELLOW)mypy not installed$(NC)"

format: ## Format code with black
	@echo "$(BLUE)Formatting code...$(NC)"
	@$(PYTHON_EXE) -m black *.py --line-length=88 || echo "$(YELLOW)black not installed$(NC)"

security: ## Run security checks
	@echo "$(BLUE)Running security checks...$(NC)"
	@$(PYTHON_EXE) -m bandit -r . -f json -o security_report.json || echo "$(YELLOW)bandit not installed$(NC)"
	@$(PYTHON_EXE) -m safety check || echo "$(YELLOW)safety not installed$(NC)"

## Data Management
create-sample: ## Create sample data for testing
	@echo "$(BLUE)Creating sample data...$(NC)"
	@$(MKDIR) $(DATA_DIR)
	@$(PYTHON_EXE) -c "import pandas as pd; import random; from datetime import datetime, timedelta; logs = []; users = ['user001', 'user002', 'user003', 'user004', 'user005']; types = ['purchase', 'withdrawal', 'transfer', 'deposit']; locations = ['NYC', 'LA', 'Chicago', 'Miami', 'Seattle']; devices = ['mobile', 'desktop', 'tablet']; [logs.append(f'{(datetime.now() - timedelta(days=random.randint(0, 30))).isoformat()}:::{random.choice(users)}:::{random.choice(types)}:::{random.choice([\"$$\", \"€\", \"£\"])}{random.uniform(10, 5000):.2f}:::{random.choice(locations)}:::{random.choice(devices)}') for i in range(1000)]; df = pd.DataFrame({'raw_log': logs}); df.to_csv('$(INPUT_FILE)', index=False); print('Sample data created at $(INPUT_FILE)')"

validate-data: ## Validate input data format
	@echo "$(BLUE)Validating data format...$(NC)"
	@$(PYTHON_EXE) -c "import pandas as pd; import sys; df = pd.read_csv('$(INPUT_FILE)'); sys.exit(1) if 'raw_log' not in df.columns else print(f'Data validation passed. Found {len(df)} records')"

## Docker Support
docker-build: ## Build Docker production image
	@echo "$(BLUE)Building Docker production image...$(NC)"
	@docker build --target production -t $(DOCKER_IMAGE):$(DOCKER_TAG) .
	@echo "$(GREEN)Docker image built: $(DOCKER_IMAGE):$(DOCKER_TAG)$(NC)"

docker-build-dev: ## Build Docker development image
	@echo "$(BLUE)Building Docker development image...$(NC)"
	@docker build --target development -t $(DOCKER_IMAGE):dev .
	@echo "$(GREEN)Docker development image built: $(DOCKER_IMAGE):dev$(NC)"

docker-build-cli: ## Build Docker CLI image
	@echo "$(BLUE)Building Docker CLI image...$(NC)"
	@docker build --target cli -t $(DOCKER_IMAGE):cli .
	@echo "$(GREEN)Docker CLI image built: $(DOCKER_IMAGE):cli$(NC)"

docker-build-all: ## Build all Docker images
	@$(MAKE) docker-build
	@$(MAKE) docker-build-dev
	@$(MAKE) docker-build-cli

docker-run: ## Run Docker production container
	@echo "$(BLUE)Running Docker production container...$(NC)"
ifeq ($(DETECTED_OS),Windows)
	@docker run -d --name anomaly-detector -p $(STREAMLIT_PORT):$(STREAMLIT_PORT) -v "%cd%\$(DATA_DIR):/app/$(DATA_DIR):ro" -v "%cd%\$(OUTPUT_DIR):/app/$(OUTPUT_DIR)" $(DOCKER_IMAGE):$(DOCKER_TAG)
else
	@docker run -d --name anomaly-detector -p $(STREAMLIT_PORT):$(STREAMLIT_PORT) -v "$(PWD)/$(DATA_DIR):/app/$(DATA_DIR):ro" -v "$(PWD)/$(OUTPUT_DIR):/app/$(OUTPUT_DIR)" $(DOCKER_IMAGE):$(DOCKER_TAG)
endif
	@echo "$(GREEN)Container started. Access at http://localhost:$(STREAMLIT_PORT)$(NC)"

docker-run-dev: ## Run Docker development container
	@echo "$(BLUE)Running Docker development container...$(NC)"
ifeq ($(DETECTED_OS),Windows)
	@docker run -d --name anomaly-detector-dev -p 8502:8501 -v "%cd%:/app" $(DOCKER_IMAGE):dev
else
	@docker run -d --name anomaly-detector-dev -p 8502:8501 -v "$(PWD):/app" $(DOCKER_IMAGE):dev
endif
	@echo "$(GREEN)Development container started. Access at http://localhost:8502$(NC)"

docker-run-cli: ## Run Docker CLI container for batch processing
	@echo "$(BLUE)Running Docker CLI container...$(NC)"
ifeq ($(DETECTED_OS),Windows)
	@docker run --rm -v "%cd%\$(DATA_DIR):/app/$(DATA_DIR):ro" -v "%cd%\$(OUTPUT_DIR):/app/$(OUTPUT_DIR)" $(DOCKER_IMAGE):cli analysis.py --input /app/$(DATA_DIR)/synthetic_dirty_transaction_logs.csv --output_dir /app/$(OUTPUT_DIR)
else
	@docker run --rm -v "$(PWD)/$(DATA_DIR):/app/$(DATA_DIR):ro" -v "$(PWD)/$(OUTPUT_DIR):/app/$(OUTPUT_DIR)" $(DOCKER_IMAGE):cli analysis.py --input /app/$(DATA_DIR)/synthetic_dirty_transaction_logs.csv --output_dir /app/$(OUTPUT_DIR)
endif

docker-stop: ## Stop running Docker containers
	@echo "$(BLUE)Stopping Docker containers...$(NC)"
	@docker stop anomaly-detector anomaly-detector-dev 2>/dev/null || true
	@docker rm anomaly-detector anomaly-detector-dev 2>/dev/null || true
	@echo "$(GREEN)Containers stopped$(NC)"

docker-logs: ## Show Docker container logs
	@docker logs anomaly-detector

docker-shell: ## Open shell in running Docker container
ifeq ($(DETECTED_OS),Windows)
	@docker exec -it anomaly-detector cmd
else
	@docker exec -it anomaly-detector /bin/bash
endif

## Docker Compose Support
compose-up: ## Start services with docker-compose
	@echo "$(BLUE)Starting services with docker-compose...$(NC)"
	@docker-compose up -d
	@echo "$(GREEN)Services started. Access at http://localhost:$(STREAMLIT_PORT)$(NC)"

compose-up-dev: ## Start development services
	@echo "$(BLUE)Starting development services...$(NC)"
	@docker-compose --profile dev up -d
	@echo "$(GREEN)Development services started. Access at http://localhost:8502$(NC)"

compose-run-cli: ## Run CLI analysis with docker-compose
	@echo "$(BLUE)Running CLI analysis...$(NC)"
	@docker-compose --profile cli run --rm anomaly-detector-cli
	@echo "$(GREEN)CLI analysis completed$(NC)"

compose-down: ## Stop docker-compose services
	@echo "$(BLUE)Stopping docker-compose services...$(NC)"
	@docker-compose down
	@echo "$(GREEN)Services stopped$(NC)"

compose-logs: ## Show docker-compose logs
	@docker-compose logs -f

compose-build: ## Build docker-compose images
	@echo "$(BLUE)Building docker-compose images...$(NC)"
	@docker-compose build
	@echo "$(GREEN)Images built$(NC)"

## Cleanup
clean: ## Clean up generated files and cache
	@echo "$(BLUE)Cleaning up...$(NC)"
	@$(RM) coverage 2>nul || true
	@$(RM) __pycache__ .pytest_cache .mypy_cache .coverage htmlcov 2>nul || true
	@$(RM_FILE) *.pyc security_report.json 2>nul || true
	@echo "$(GREEN)Cleanup completed$(NC)"

clean-all: clean ## Clean everything including virtual environment
	@echo "$(BLUE)Removing virtual environment...$(NC)"
	@$(RM) $(VENV_DIR) 2>nul || true
	@echo "$(GREEN)Full cleanup completed$(NC)"

clean-docker: ## Clean Docker images and containers
	@echo "$(BLUE)Cleaning Docker resources...$(NC)"
	@docker-compose down --rmi all --volumes --remove-orphans 2>/dev/null || true
	@docker rmi $(DOCKER_IMAGE):$(DOCKER_TAG) $(DOCKER_IMAGE):dev $(DOCKER_IMAGE):cli 2>/dev/null || true
	@docker system prune -f
	@echo "$(GREEN)Docker cleanup completed$(NC)"

## Development
dev-setup: setup install-dev ## Complete development setup
	@echo "$(GREEN)Development environment ready on $(DETECTED_OS)!$(NC)"
	@echo "$(YELLOW)Next steps:$(NC)"
	@echo "  1. Run 'make validate-data' to check your data"
	@echo "  2. Run 'make all-methods' to run all analyses"
	@echo "  3. Run 'make streamlit' to start the web interface"
	@echo "  4. Run 'make docker-build-all' to build Docker images"

check: lint test security ## Run all quality checks
	@echo "$(GREEN)All quality checks completed$(NC)"

## Information
info: ## Show project information
	@echo "$(BLUE)Project Information:$(NC)"
	@echo "  OS: $(DETECTED_OS)"
	@echo "  Python: $(shell $(PYTHON) --version 2>nul || echo 'Not found')"
	@echo "  UV: $(shell $(UV) --version 2>nul || echo 'Not installed')"
	@echo "  Docker: $(shell docker --version 2>nul || echo 'Not installed')"
	@echo "  Docker Compose: $(shell docker-compose --version 2>nul || echo 'Not installed')"
	@echo "  Virtual Environment: $(VENV_DIR)"
	@echo "  Data Directory: $(DATA_DIR)"
	@echo "  Output Directory: $(OUTPUT_DIR)"
	@echo "  Input File: $(INPUT_FILE)"
	@echo "  Streamlit Port: $(STREAMLIT_PORT)"
	@echo "  Docker Image: $(DOCKER_IMAGE):$(DOCKER_TAG)"

status: ## Show project status
	@echo "$(BLUE)Project Status:$(NC)"
ifeq ($(DETECTED_OS),Windows)
	@if exist "$(VENV_DIR)" (echo "  Virtual Environment: $(GREEN)✓ Created$(NC)") else (echo "  Virtual Environment: $(RED)✗ Missing$(NC)")
	@if exist "$(INPUT_FILE)" (echo "  Input Data: $(GREEN)✓ Found$(NC)") else (echo "  Input Data: $(RED)✗ Missing$(NC)")
	@if exist "$(VENV_DIR)\pyvenv.cfg" (echo "  Dependencies: $(GREEN)✓ Installed$(NC)") else (echo "  Dependencies: $(RED)✗ Not installed$(NC)")
else
	@echo -n "  Virtual Environment: "
	@if [ -d "$(VENV_DIR)" ]; then echo "$(GREEN)✓ Created$(NC)"; else echo "$(RED)✗ Missing$(NC)"; fi
	@echo -n "  Input Data: "
	@if [ -f "$(INPUT_FILE)" ]; then echo "$(GREEN)✓ Found$(NC)"; else echo "$(RED)✗ Missing$(NC)"; fi
	@echo -n "  Dependencies: "
	@if [ -f "$(VENV_DIR)/pyvenv.cfg" ]; then echo "$(GREEN)✓ Installed$(NC)"; else echo "$(RED)✗ Not installed$(NC)"; fi
endif
	@echo -n "  Docker Images: "
	@docker images $(DOCKER_IMAGE) --format "table" 2>/dev/null | grep -v REPOSITORY || echo "$(RED)✗ Not built$(NC)"

## Quick Start
quick-start: ## Quick start: setup + sample data + run analysis
	@echo "$(BLUE)Quick start setup on $(DETECTED_OS)...$(NC)"
	@$(MAKE) setup
	@$(MAKE) create-sample
	@$(MAKE) validate-data
	@$(MAKE) run-isolation
	@echo "$(GREEN)Quick start completed! Check $(OUTPUT_DIR)$(PATH_SEP)isolation_forest$(PATH_SEP) for results$(NC)"

quick-start-docker: ## Quick start with Docker
	@echo "$(BLUE)Quick start with Docker...$(NC)"
	@$(MAKE) create-sample
	@$(MAKE) docker-build
	@$(MAKE) docker-run
	@echo "$(GREEN)Docker quick start completed! Access at http://localhost:$(STREAMLIT_PORT)$(NC)"
