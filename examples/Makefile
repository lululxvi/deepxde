DDE_TEST_ENV = DDE_STEPS_OVERRIDE=1 DDE_SAVE_RESULTS=0 PYTHONPATH=..:../..


examples: pinn_inverse
	@for dir in $^; do                                                   \
	    cd $$dir;                                                        \
		for file in *.py; do                                             \
		    echo -----------------------------------------------------;  \
			echo $$dir/$$file;                                           \
			echo -----------------------------------------------------;  \
			$(DDE_TEST_ENV) python $$file;                               \
			echo;                                                        \
			echo;                                                        \
			echo;                                                        \
		done;                                                            \
		cd ..;                                                           \
	done;
