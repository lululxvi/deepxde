DDE_TEST_ENV = DDE_INTEGRATION_TEST=1 PYTHONPATH=..:../..


run_examples: function pinn_forward pinn_inverse
	@for dir in $^; do                                                   \
	    cd $$dir;                                                        \
		for file in *.py; do                                             \
		    echo -----------------------------------------------------;  \
			echo $$dir/$$file;                                           \
			echo -----------------------------------------------------;  \
			$(DDE_TEST_ENV) python $$file || exit 1;                     \
			echo;                                                        \
			echo;                                                        \
			echo;                                                        \
		done;                                                            \
		cd ..;                                                           \
	done;
