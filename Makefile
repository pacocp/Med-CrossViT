.DEFAULT_GOAL := run

build:
	@command -v uv >/dev/null 2>&1 || pip install -U uv
	uv run --with med_crossvit -- python -c "import med_crossvit"
