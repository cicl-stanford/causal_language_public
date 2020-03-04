# Causal Language

This is code for Beller et al. (submitted).

## Tour through the Model

### Physics Engine Code

The base of the model is the model.py module in code/python.
You can run simulations and counterfactual tests using the procedures in the model.

Example

Start an interactive window:

	python

Load model:

	>>> import model as m

Run a Simulation:

	>>> trials = m.load_trials("trialinfo/experiment_trials.json")
	>>> test_trial = trials[12]
	>>> m.run_trial(test_trial, animate=True)
	{'collisions': [{'objects': {'A', 'B'}, 'step': 135}], 'wall_bounces': [], 'button_presses': [], 'outcome': 1, 'outcome_fine': Vec2d(-1060.2315946785948, 300.0)}
Video will play.


Run a Counterfactual Test:

	>>> m.whether_test(test_trial, candidate="A", target="B", noise=1.0, num_samples=100, animate=False)
	1.0
Can flip the animate tag to True if you wish observe the test. See model.py file for other counterfactual tests and parameters.


### Generating Aspect Representations

To generate the aspect representations for the model you can use the aspect_generator.py script.

	python aspect_generator.py --i trialinfo/experiment_trials.json --samples 100 --uncertainty_noise 1.0

Example lists the input file, sample number, and uncertainty noise value. To list the arguments to the script, use the -h flag.

Aspect files are written into the aspects folder. Aspect files used for values in the the paper are in the aspects_paper folder.


### RSA and model predictions

With the aspect representations for the clips we can produce model predictions by running the semantics and pragmatics. forced_choice_expt_rsa.py runs the semantics/pragmatics for the model parameters that we report in the paper. You can add other parameter choices by modifying the multi-level for loop. Runtime (and size of the output file) will grow exponentially with the number of parameters. The output is saved in forced_choice_expt_rsa.csv.

	python forced_choice_expt_rsa.py

forced_choice_expt_rsa.py generates predictions for the full model and the lesioned RSA model. The predictions for the Bayesian Ordnial Regression are generated in the analysis file.


## Reproduce paper results

1. Install depdenencies

* R
* RStudio
* python

conda:

* 

pip:

* 

1. Generate aspects. This runs the whether, how, and sufficiency tests accross samples of counterfactual simulations. Output will be saved to `code/python/aspects/`. Note that the generated aspects from the paper are already included in this repo in `code/python/aspects_paper`.

        cd code/python/
        python aspect_generator.py --i trialinfo/experiment_trials.json --samples 1000 --uncertainty_noise 1.5

2. Run RSA pragmatics model. Note the parameters are set to the optimal ones found in the paper. To fit model parameters, change the lines commented with "PARAM SEARCH"

        cd code/python/
        python forced_choice_expt_rsa.py

3. In RStudio, install packages as needed and then knit `forced_choice_expt_analysis.Rmd` to remake all plots and compute reported statistics.

## Run experiment


