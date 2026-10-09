.PHONY: install dev lint fmt typecheck test test-pg mutation serve mcp ask bench bench-data llama docker clean

PY ?= python
LLAMA_DIR ?= /path/to/llama.cpp/build/bin
MODEL ?= qwen2.5-0.5b-instruct-q4_k_m.gguf
LLAMA_PORT ?= 8093

install:            ## runtime install with every extra
	$(PY) -m pip install -e ".[all]"

dev:                ## dev install (tests, lint, types, testteeth)
	$(PY) -m pip install -e ".[dev]"

lint:
	ruff check src tests
	ruff format --check src tests

fmt:
	ruff format src tests
	ruff check --fix src tests

typecheck:
	mypy

test:               ## unit + integration tests (Postgres tests skip without SQLPILOT_TEST_PG_URL)
	pytest -q

test-pg:            ## Postgres tests against a throwaway container
	docker run -d --rm --name sqlpilot-pg -e POSTGRES_PASSWORD=pg -p 55432:5432 postgres:16-alpine
	sleep 5
	SQLPILOT_TEST_PG_URL=postgresql://postgres:pg@localhost:55432/postgres pytest -q -m postgres; \
	  status=$$?; docker stop sqlpilot-pg; exit $$status

mutation:           ## mutation score of the core modules (guard, retrieval, routing, auth) with testteeth
	testteeth run --json mutation.json

serve:              ## API + web UI on the bundled Chinook DB, mock LLM, key "demo-key-123"
	SQLPILOT_LLM=$${SQLPILOT_LLM:-mock} sqlpilot serve --demo-key demo-key-123

mcp:
	sqlpilot mcp --llm $${SQLPILOT_LLM:-mock}

ask:
	sqlpilot ask --llm $${SQLPILOT_LLM:-mock} "$(Q)"

bench-data:         ## download Spider (official archive, ~200 MB) into .bench_data/
	mkdir -p .bench_data
	curl -L -o .bench_data/spider_data.zip "https://drive.usercontent.google.com/download?id=1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J&export=download&confirm=t"
	cd .bench_data && unzip -q -o spider_data.zip "spider_data/dev.json" "spider_data/tables.json" "spider_data/database/*" && rm spider_data.zip

llama:              ## start llama.cpp's OpenAI-compatible server (set LLAMA_DIR and MODEL)
	LD_LIBRARY_PATH=$(LLAMA_DIR) $(LLAMA_DIR)/llama-server -m $(MODEL) --port $(LLAMA_PORT) -t 2 -c 4096 -np 1

bench:              ## EX on 150 Spider dev questions (needs bench-data and a running LLM endpoint)
	sqlpilot bench --n 150 --llm openai --llm-base-url http://127.0.0.1:$(LLAMA_PORT)/v1 --out bench/results/latest

docker:
	docker build -t sqlpilot:local .

clean:
	rm -rf dist build .pytest_cache .mypy_cache .ruff_cache mutation.json .sqlpilot
