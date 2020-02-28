# Causal Language

## Tour through the Model

### Physics Engine Code

The base of the model is the model.py module in code/python.
You can run simulations and counterfactual tests using the procedures in the model.

Example

Start an interactive window:

	(testenv) Aris-MacBook-Pro-2:python aribeller$ python
	Python 3.7.2 (default, Dec 29 2018, 00:00:04) 
	[Clang 4.0.1 (tags/RELEASE_401/final)] :: Anaconda, Inc. on darwin
	Type "help", "copyright", "credits" or "license" for more information.
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

	(testenv) Aris-MacBook-Pro-2:python aribeller$ python aspect_generator.py --i trialinfo/experiment_trials.json --samples 100 --uncertainty_noise 1.0

Example lists the input file, sample number, and uncertainty noise value. To list the arguments to the script, use the -h flag.

Aspect files are written into the aspects folder. Aspect files used for values in the the paper are in the aspects_paper folder.


### RSA and model predictions

With the aspect representations for the clips we can produce model predictions by running the semantics and pragmatics. forced_choice_expt_rsa.py runs the semantics/pragmatics for the model parameters that we report in the paper. You can add other parameter choices by modifying the multi-level for loop. Runtime (and size of the output file) will grow exponentially with the number of parameters. The output is saved in forced_choice_expt_rsa.csv.

	(testenv) Aris-MacBook-Pro-2:python aribeller$ python forced_choice_expt_rsa.py

forced_choice_expt_rsa.py generates predictions for the full model and the lesioned RSA model. The predictions for the Bayesian Ordnial Regression are generated in the analysis file.
