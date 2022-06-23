# Directories where to look for *.py example scripts
EXAMPLE_DIRECTORIES := function pinn_forward pinn_inverse

# Environment variables for the scripts
# Note that the PYTHONPATH is relative to each example subdirectory, not the
#    directory where the Makefile resides
DDE_TEST_ENV := DDE_INTEGRATION_TEST=1 PYTHONPATH=..:../..



.PHONY: _run_one_example run_all_examples clean

# List every Python script in the example directories
# NB: notebooks are ignored
example_files := $(foreach dir,$(EXAMPLE_DIRECTORIES),$(wildcard $(dir)/*.py))


# Default rule -- run everything
run_all_examples: $(example_files)


# Run matching Python files
%.py: _run_one_example
	@echo -----------------------------------------------------
	@echo $@
	@echo -----------------------------------------------------
	@cd $(dir $@) && $(DDE_TEST_ENV) python ../sample_to_test.py $(notdir $@) | PYTHONPATH=../.. python -
	@echo
	@echo
	@cd ..


# Clean all data files in example directories
clean: $(foreach dir,$(EXAMPLE_DIRECTORIES),$(wildcard $(dir)/*.dat)) $(foreach dir,$(EXAMPLE_DIRECTORIES),$(wildcard $(dir)/*.npz))
	rm $^
