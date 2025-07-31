# Makefile for Housing Market Econometrics Project

.PHONY: help install install-dev test lint format clean run-api run-analysis docker-build docker-up docker-down migrate migrate-up migrate-down

# Default target
help:
	@echo "Available commands:"
	@echo "  install      - Install production dependencies"
	@echo "  install-dev  - Install development dependencies"
	@echo "  test         - Run tests"
	@echo "  lint         - Run linting"
	@echo "  format       - Format code"
	@echo "  clean        - Clean up temporary files"
	@echo "  run-api      - Start the API server"
	@echo "  run-analysis - Run the econometric analysis"
	@echo "  docker-build - Build Docker images"
	@echo "  docker-up    - Start Docker services"
	@echo "  docker-down  - Stop Docker services"
	@echo "  migrate      - Create a new migration"
	@echo "  migrate-up   - Apply migrations"
	@echo "  migrate-down - Rollback migrations"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pip install -e .

# Testing
test:
	pytest tests/ -v --cov=. --cov-report=html --cov-report=term

test-fast:
	pytest tests/ -v --tb=short

# Code quality
lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
	mypy . --ignore-missing-imports

format:
	black . --line-length=127
	isort . --profile=black

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf build/
	rm -rf dist/

# Running applications
run-api:
	python scripts/start_api.py

run-analysis:
	python scripts/run_analysis.py

# Docker commands
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

# Production Docker
docker-prod-build:
	docker-compose -f docker/docker-compose.prod.yml build

docker-prod-up:
	docker-compose -f docker/docker-compose.prod.yml up -d

docker-prod-down:
	docker-compose -f docker/docker-compose.prod.yml down

# Database migrations
migrate:
	alembic revision --autogenerate -m "$(message)"

migrate-up:
	alembic upgrade head

migrate-down:
	alembic downgrade -1

migrate-history:
	alembic history

# Data management
load-data:
	python scripts/load_housing_data.py

backup-db:
	python scripts/backup_database.py

# Development setup
setup-dev: install-dev
	cp env.example .env
	@echo "Please edit .env file with your configuration"
	@echo "Then run: make migrate-up"

# Full setup
setup: install
	cp env.example .env
	@echo "Please edit .env file with your configuration"
	@echo "Then run: make migrate-up"

# CI/CD
ci: lint test

# Documentation
docs:
	@echo "API documentation available at http://localhost:8000/docs"
	@echo "ReDoc documentation available at http://localhost:8000/redoc"

# Health check
health:
	curl -f http://localhost:8000/health || echo "API is not running"

# Monitoring
monitor:
	@echo "Metrics available at http://localhost:8000/metrics"
	@echo "Health check available at http://localhost:8000/health"

# Quick start
quick-start: setup-dev
	@echo "Starting services..."
	make docker-up
	@echo "Waiting for services to start..."
	sleep 10
	make migrate-up
	@echo "Services are ready!"
	@echo "API: http://localhost:8000"
	@echo "Docs: http://localhost:8000/docs" 