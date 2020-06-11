# Causal Language

This is code for Beller, Bennett, and Gerstenberg (2020). For any questions about the repo, feel free to contact Ari Beller at abeller@stanford.edu

## Repository Structure

### code

#### python

Contains code for running the model and searching across parameter settings for optimal model.

1. model.py includes the underlying physics engine and code to run counterfactual tests.

2. compute_aspect_rep.py will run the model to produce causal representations of the experiment trials. Takes an uncertainty noise value and sample number as command line arguments.

3. rsa.py contains code for the semantics and pragmatics components of the model. The meaning function computes the semantic representations for an utterance, and the l0, s1, l1, s2 functions compute the successive levels of recursive pragmatic reasoning. The lesion_model function computes the no pragmatics model representation.

4. grid_search.py contains tools for running grid searches across for given models across parameter settings. It also contains tools for running cross validation, and saving models to file for analysis in R.

5. model_predictions.py is a script for easy reproduction of model statistics reported in the paper.

Note grid_search.py and model_predictions.py only produce model predictions and cross-validation results for the full model and no pragmatics model. Ordinal Regression training, prediction, and parameter selection is all computed in R.

#### R

Contains code for model analysis and as well as training, prediction, and cross-validation for the Ordinal Regression.

1. forced_choice_expt_analysis.Rmd analysis script. Can be knitted to reproduce model predictions.

2. forced_choice_expt_analysis.md is a pre-knitted markdown file. 

3. crossv_ordreg.R is a script to produce model predictions for ordinal regression cross-validation. Requires a command line argument specifying the split number for which to compute regression models. Requires considerable time to compute regressions as well as memory space as regressions are saved to file. We performed ordinal regression cross-validation on Stanford's high performance computing cluster [Sherlock](https://www.sherlock.stanford.edu/).

#### bash

1. combine_frames.sh takes a set of frames saved in code/python/figures/frames and produces a video clip saved in code/python/video. Requires experiment name and trial number as command line arguments e.g.

`./combine_frames.sh exp_name 5`

Also requires [ffmpeg](https://ffmpeg.org/) multimedia framework.

Frames for video processing be produced using the physics simulator. The following code demonstrates how to produce frames for an arbitrary trial.

	aribeller$ cd code/python/
	aribeller$ python
	Python 3.7.2 (default, Dec 29 2018, 00:00:04) 
	[Clang 4.0.1 (tags/RELEASE_401/final)] :: Anaconda, Inc. on darwin
	Type "help", "copyright", "credits" or "license" for more information.
	>>> import model as m
	pygame 1.9.4
	Hello from the pygame community. https://www.pygame.org/contribute.html
	Loading chipmunk for Darwin (64bit) [/Users/aribeller/miniconda3/envs/testenv/lib/python3.7/site-packages/pymunk/libchipmunk.dylib]
	>>> trials = m.load_trials("trialinfo/experiment_trials.json")
	>>> test_trial = trials[12]
	>>> m.run_trial(test_trial, animate=True, save=True)
	{'collisions': [{'objects': {'B', 'A'}, 'step': 135}], 'wall_bounces': [], 'button_presses': [], 'outcome': 1, 'outcome_fine': Vec2d(-1060.2315946785948, 300.0)}
	>>> 

#### experiment

1. Contains the code for our experiment in the folder experiment_forced_choice. For info on how to run the experiment refer to the [psiturk documentation](https://psiturk.org/).

### data

Contains the raw data file full_database_anonymized.db

### figures

#### paper_plots

Plots presented in the paper.

#### trial_schematics

Single frame depictions of the trial clips.

### videos

Video clips presented to participants in the experiment.

## Reproduce paper results

1. Install depdenencies

* R
* RStudio
* python

conda:

* numpy
* pandas
* scipy

pip:

* pygame==1.9.6
* pymunk

1. Compute aspect representation. This runs the whether, how, sufficiency, and moving tests accross samples of counterfactual simulations. Output will be saved to `code/python/aspects/`. Note that the generated aspects from the paper are already included in this repo in `code/python/aspects_paper`. Downstream model components read from `aspects_paper`, but paths can be modified. 

        cd code/python/
        python compute_aspect_rep.py <uncertainty_noise> <num_samples>

2. To compute and save models for reported paper statistics use the top_models.py script. model_predictions.py produces 4 csvs in the useful_csvs folder. The first top_models.csv contains model predictions for the top full model and no pragmatics model (as well as comparisons with and without combined cause respectively). The second and third are the cross_validation_full_model.csv and the cross_validation_lesion_model.csv which contain model predictions for for cross validation models trained and tested on the splits specified in crossv_splits.csv. Lastly, the final csv is the enabled_comparison.csv, which contains predictions for models with and without the "not how" semantics for the "enabled" causal expression. Files are previously computed and saved in useful_csvs, but can be re-run with the following code:

        cd code/python/
        python model_predictions.py

3. In RStudio, install packages as needed and then knit `forced_choice_expt_analysis.Rmd` to remake all plots and compute reported statistics. The compiled file `forced_choice_expt_analysis.md` contains all findings reported in the paper.