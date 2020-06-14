black:
	@black --version
	@black poisson_solver tests

black_check:
	@black --version
	@black --check poisson_solver tests

isort:
	@isort --version
	@isort --recursive .

isort_check:
	@isort --version
	@isort --recursive --check-only

flake8:
	@flake8 --version
	@flake8 poisson_solver tests

clean_notebooks:
    # This finds Ipython jupyter notebooks in the code
    # base and cleans only its output results. This
    # results in
	@jupyter nbconvert --version
	@find . -maxdepth 3 -name '*.ipynb'\
		| while read -r src; do jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace "$$src"; done

pylint:
	@pylint --version
	@pylint poisson_solver

all:black pylint flake8
ci:black_check flake8
