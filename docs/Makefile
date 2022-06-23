# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = DeepXDE
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

tensorflow.compat.v1:
	@echo "##################################################################"
	@echo "#                                                                #"
	@echo "#            Step 1: Building TensorFlow.compat.v1 tutorials     #"
	@echo "#                                                                #"
	@echo "##################################################################"
	@DDE_BACKEND=tensorflow.compat.v1 $(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

tensorflow:
	@echo "##################################################################"
	@echo "#                                                                #"
	@echo "#            Step 2: Building TensorFlow tutorials               #"
	@echo "#                                                                #"
	@echo "##################################################################"
	@DDE_BACKEND=tensorflow $(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

pytorch:
	@echo "##################################################################"
	@echo "#                                                                #"
	@echo "#            Step 3: Building PyTorch tutorials                  #"
	@echo "#                                                                #"
	@echo "##################################################################"
	@DDE_BACKEND=pytorch $(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

html: Makefile tensorflow.compat.v1 tensorflow pytorch

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
