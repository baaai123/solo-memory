.PHONY: test lint format clean setup

test:
	python3 -m pytest tests/ -q

lint:
	python3 -m ruff check memory_skill/ 2>/dev/null || echo "ruff not installed — skip"

format:
	python3 -m ruff format memory_skill/

clean:
	rm -rf memory.db memory.db-wal memory.db-shm memory.db_chroma
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true

setup:
	./setup.sh

.PHONY: all
all: test
