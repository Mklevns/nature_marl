# File: Makefile
"""
Development automation commands
"""

.PHONY: help install install-dev test test-all lint format type-check security docs clean

help:
	@echo "Available commands:"
	@echo "  install     - Install package"
	@echo "  install-dev - Install package with dev dependencies"
	@echo "  test        - Run unit tests"
	@echo "  test-all    - Run all tests (unit, integration, performance)"
	@echo "  lint        - Run linting (flake8)"
	@echo "  format      - Format code (black)"
	@echo "  type-check  - Run type checking (mypy)"
	@echo "  security    - Run security checks (bandit, safety)"
	@echo "  docs        - Build documentation"
	@echo "  clean       - Clean build artifacts"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,docs,viz,gpu]"

test:
	pytest tests/ -v --cov=nature_marl

test-all:
	pytest tests/ -v --cov=nature_marl --benchmark-skip
	python tests/test_environment_integration.py
	python tests/test_rllib_integration.py

lint:
	flake8 nature_marl/ tests/

format:
	black nature_marl/ tests/
	isort nature_marl/ tests/

type-check:
	mypy nature_marl/ --ignore-missing-imports

security:
	bandit -r nature_marl/
	safety check

docs:
	cd docs && make html

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
