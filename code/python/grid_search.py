import numpy as np
import json
import sqlite3
import pandas as pd
import time
import rsa

vocabulary_dataset = ["cause", "enable", "affect", "no_difference"]
vocabulary = ["caused", "enabled", "affected", "didn't affect"]
w, h, s, m, o = (0, 1, 2, 3, 4)

trial_max = 31


# Code to load the data from the sql file
database_path = "../../data/full_database_anonymized.db"
experiment = "forced_choice_2"

# connect to sql database
con = sqlite3.connect(database_path)
df_sql = pd.read_sql_query("SELECT * from language", con)

# filter database based on experiment, status, and completion
is_live = df_sql["mode"] == "live"
is_experiment = df_sql["codeversion"] == "forced_choice_2"
is_complete = df_sql["status"].isin([3,4,5])

df_sql = df_sql[is_live & is_experiment & is_complete]

# Load the data from each entry in the database
data_objects = [json.loads(x) for x in df_sql["datastring"]]


# Create a dataframe
data = []
for participant in range(len(data_objects)):
  data_obj = data_objects[participant]

  # number of trials including attention check
  for j in range(trial_max):
    trialdata = data_obj["data"][j]['trialdata']
    trial = trialdata["id"]
    response = trialdata['response']

    data.append([participant, trial, response])


df_data = pd.DataFrame(data, columns = ["participant", "trial", "response"])

df_data.sort_values(by = ['participant', 'trial'], inplace=True)

# filter out participant that failed attention check
attention_check_fail = df_data[(df_data['trial'] == 30) & (df_data['response'] != 'no_difference')]
excluded_participants = attention_check_fail['participant']

df_data = df_data[~df_data['participant'].isin(excluded_participants)]

# filter out the attention check
df_data = df_data[~(df_data["trial"] == 30)]

vocab = np.array(['cause', 'enable', 'affect', 'no_difference'])
def convert_verb(response):
  return (vocab == response).astype(int)

# Adds a column with a vector representation of the selection for easy tally
# Order of the vector is determined by the vocap array
df_data = df_data.assign(vec_response = list(map(convert_verb, df_data['response'])))

# sum vectors over trial
trial_counts = df_data.groupby(["trial"]).vec_response.apply(np.sum)

trial_counts = np.array(trial_counts.values.tolist()).T
data_averaged = trial_counts/np.sum(trial_counts, axis = 0)




# Procedure for loading aspects to be used in semantic and pragmatic computations
# Returns the primary aspect values and a list of the alternative aspect value arrays
# Could generalize to consider any candidate as possible primary. For the moment only
# considers the first. Works for the current setup
def load_aspects(uncertainty_noise, num_samples=1000):
  
  clips_data = json.load(open("aspects_paper/aspects_noise_{}_samples_{}.json".format(uncertainty_noise, num_samples)))

  primary_candidate = clips_data[0]
  alternative_candidates = clips_data[1:]

  # Convert representation to numpy array
  primary_aspect_values = np.array([tr['rep'] for tr in primary_candidate])[:,:4]

  alternative_aspect_values = []

  for candidate_aspects in alternative_candidates:
    alternative_aspects = np.array([tr['rep'] for tr in candidate_aspects])[:,:4]
    alternative_aspect_values.append(alternative_aspects)

  return primary_aspect_values, alternative_aspect_values


# compute the sum square error of a model output
# against a subset of the averaged data values.
# Model output will be 4*trial_num, and the trial_set vector allows for an
# arbitrary subset for comparison
def compute_error(model_output, trial_set):
  data_set = data_averaged[:, trial_set]
  model_set = model_output[:, trial_set]

  sq_err = np.sum((data_set - model_set)**2)

  return sq_err


# For a trial set and set of ranges across parameters, find the parameter setting
# That minimizes the squared error on the trial set. Must additionally specify a 
# whether or not to lesion the full model, as well as a caused version and an enabled version
# If caused version involves the dummy combined cause must also provide comparison threshold
# and softener ranges. Can set testing to true to return each parameter setting with its
# squared error value.

# To improve: remove non-relevant param lists from opt_set when producing testing
# output
def grid_search(trial_set, lesion_rsa, unoise_range, not_affect_param_range, stationary_softener_range, speaker_optimality_range, beta_range, caused_version = "and_hm_or_ws_cc", enabled_version = "or_ws", comparison_threshold_range=None, comparison_softener_range=None, testing=False):

  if not testing: 
    opt_set = {'unoise': None,
     'not_affect_param': None, 
     'stationary_softener': None, 
     'speaker_optimality': None,
     'beta': None,
     'comparison_threshold': None,
     'comparison_softener': None,
     'error': np.inf}
  else:
    opt_set = {'unoise': [],
     'not_affect_param': [], 
     'stationary_softener': [], 
     'speaker_optimality': [],
     'beta': [],
     'comparison_threshold': [],
     'comparison_softener': [],
     'error': []
    }


  for uncertainty_noise in unoise_range:

    primary_aspect_values, alternative_aspect_values = load_aspects(uncertainty_noise)

    for how_not_affect_param in not_affect_param_range:

      for stationary_softener in stationary_softener_range:

        if "dummy" not in caused_version:

          if lesion_rsa:

            meanings = np.vstack([rsa.meaning(word, primary_aspect_values, caused_version, enabled_version, how_not_affect_param, stationary_softener, alternative_aspect_values=alternative_aspect_values) for word in vocabulary])

            for beta in beta_range:

              semantic_values = rsa.softmax(meanings, 0, beta=beta)
              sq_err = compute_error(semantic_values, trial_set)

              if sq_err < opt_set['error']:
                opt_set['unoise'] = uncertainty_noise
                opt_set['not_affect_param'] = how_not_affect_param
                opt_set['stationary_softener'] = stationary_softener
                opt_set['beta'] = beta
                opt_set['error'] = sq_err

          else:

            for speaker_optimality in speaker_optimality_range:
              model_output = rsa.s2(primary_aspect_values, caused_version, enabled_version, how_not_affect_param, stationary_softener, speaker_optimality, alternative_aspect_values=alternative_aspect_values)
              sq_err = compute_error(model_output, trial_set)

              if not testing:
                if sq_err < opt_set['error']:
                  opt_set['unoise'] = uncertainty_noise
                  opt_set['not_affect_param'] = how_not_affect_param
                  opt_set['stationary_softener'] = stationary_softener
                  opt_set['speaker_optimality'] = speaker_optimality
                  opt_set['error'] = sq_err
              else:
                opt_set['unoise'].append(uncertainty_noise)
                opt_set['not_affect_param'].append(how_not_affect_param)
                opt_set['stationary_softener'].append(stationary_softener)
                opt_set['speaker_optimality'].append(speaker_optimality)
                opt_set['error'].append(sq_err)
            
        else:

          for comparison_threshold in comparison_threshold_range:

            for comparison_softener in comparison_softener_range:

              if lesion_rsa:

                meanings = np.vstack([rsa.meaning(word, primary_aspect_values, caused_version, enabled_version, how_not_affect_param, stationary_softener, comparison_threshold, comparison_softener, alternative_aspect_values=alternative_aspect_values) for word in vocabulary])

                for beta in beta_range:
                  semantic_values = rsa.softmax(meanings, 0, beta=beta)
                  sq_err = compute_error(semantic_values, trial_set)

                  if sq_err < opt_set['error']:
                    opt_set['unoise'] = uncertainty_noise
                    opt_set['not_affect_param'] = how_not_affect_param
                    opt_set['stationary_softener'] = stationary_softener
                    opt_set['beta'] = beta
                    opt_set['comparison_threshold'] = comparison_threshold
                    opt_set['comparison_softener'] = comparison_softener
                    opt_set['error'] = sq_err

              else:

                for speaker_optimality in speaker_optimality_range:
                  model_output = rsa.s2(primary_aspect_values, caused_version, enabled_version, how_not_affect_param, stationary_softener, speaker_optimality, comparison_threshold, comparison_softener, alternative_aspect_values=alternative_aspect_values)

                  sq_err = compute_error(model_output, trial_set)

                  if sq_err < opt_set['error']:
                    opt_set['unoise'] = uncertainty_noise
                    opt_set['not_affect_param'] = how_not_affect_param
                    opt_set['stationary_softener'] = stationary_softener
                    opt_set['speaker_optimality'] = speaker_optimality
                    opt_set['comparison_threshold'] = comparison_threshold
                    opt_set['comparison_softener'] = comparison_softener
                    opt_set['error'] = sq_err



  return opt_set



# Procedure to generate random splits of the trial set.
# When training Bayesian Ordinal Regression, we noted that the model
# would fail to converge if there where no cases with stationary causes
# in the training set (because the model never saw variation in the movement feature)
# We require that the training set have at least one stationary trial

# Requires a list of trial nums and the number of splits to produce
# Returns a list of tuples where the train list is the left element and
# the test list is the right element
def generate_splits(trials, num_splits):
  stationary_trials = {14, 17, 21, 27}
  splits = []
  i = 0
  while i < num_splits:
    train = np.sort(np.random.choice(trials, size=15, replace=False))
    test = np.setdiff1d(trials, train)
    train_set = set(train)

    # Require at least one stationary trial in training to proceed
    if len(train_set.intersection(stationary_trials)) > 0:
      splits.append((train, test))
      i += 1

  return splits

# Procedure to run cross-validation for a model on a given set of splits.
# The model is identified by the lesion rsa parameter, caused definition and enabled definition.
# On each split the model will search for an optimal parameter setting on the training
# set. If save_models is false, it will then return the squared error value computed on the 
# test set. If save_models is true, it will take the optimal model trained on the training set
# and save model predictions for both testing and training set to file.

# Requires a set of splits to consider, and an indicator of whether or not to lesion the 
# RSA module. Also requires parameter ranges for uncertainty noise, the no difference parameter,
# the stationary softener, and the speaker optimality or beta range (depending on whether the model
# is the full model or the lesion rsa model). Caused version and enabled version can be changed,
# thought default values are set. 

# Returns an array of error scores if save models is false, and a dataframe of model predictions
# if save models is true
def cross_validation(splits, lesion_rsa, unoise_range, not_affect_param_range, stationary_softener_range, speaker_optimality_range, beta_range, caused_version="and_hm_or_ws_cc", enabled_version="or_ws", save_models=False):

  if not save_models:
    error_scores = []
  else:
    df_predictions = {"split": [], "lesion_rsa": [], "caused_version": [], "model_params": [], "trial": [], "verb": [], "model_pred": [], "data_val": [], "use": []}

  for i in range(len(splits)):
    if i % 10 == 0:
      print("Split", str(i))
    spl = splits[i]
    train = spl[0]
    test = spl[1]

    # First find the optimal parameter setting on the training set
    parameters = grid_search(train, lesion_rsa, unoise_range, not_affect_param_range, stationary_softener_range, speaker_optimality_range, beta_range, caused_version=caused_version, enabled_version=enabled_version)

    # extract parameters
    primary_aspect_values, alternative_aspect_values = load_aspects(parameters['unoise'])
    not_affect_param = parameters['not_affect_param']
    stationary_softener = parameters['stationary_softener']
    fine_tune = parameters['speaker_optimality'] if not lesion_rsa else parameters['beta']

    # Compute model output
    if not lesion_rsa:
      model_output = rsa.s2(primary_aspect_values, caused_version, enabled_version, not_affect_param, stationary_softener, fine_tune, alternative_aspect_values=alternative_aspect_values)
    else:
      model_output = rsa.lesion_model(primary_aspect_values, caused_version, enabled_version, not_affect_param, stationary_softener, fine_tune, alternative_aspect_values=alternative_aspect_values)

    # If not save_models, return the error score (computed on the test)
    if not save_models:
      split_error = compute_error(model_output, test)
      error_scores.append(split_error)

    # Otherwise save the model params to file
    else:

      train_model = model_output[:, train]
      train_data = data_averaged[:, train]
      test_model = model_output[:, test]
      test_data = data_averaged[:, test]

      assert test_model.shape[1] == 15

      for j in range(train_model.shape[1]):
        train_trial = train[j]
        test_trial = test[j]
        for k in range(len(vocabulary)):
          verb = vocabulary[k]

          df_predictions['split'].append(i)
          df_predictions['lesion_rsa'].append(lesion_rsa)
          df_predictions['caused_version'] = caused_version
          df_predictions['model_params'].append([parameters['unoise'], not_affect_param, stationary_softener] + ([fine_tune, None] if not lesion_rsa else [None, fine_tune]))
          df_predictions['trial'].append(train_trial)
          df_predictions['verb'].append(verb)
          df_predictions['model_pred'].append(train_model[k,j])
          df_predictions['data_val'].append(train_data[k,j])
          df_predictions['use'].append("train")


          df_predictions['split'].append(i)
          df_predictions['lesion_rsa'].append(lesion_rsa)
          df_predictions['caused_version'] = caused_version
          df_predictions['model_params'].append([parameters['unoise'], not_affect_param, stationary_softener] + ([fine_tune, None] if not lesion_rsa else [None, fine_tune]))
          df_predictions['trial'].append(test_trial)
          df_predictions['verb'].append(verb)
          df_predictions['model_pred'].append(test_model[k,j])
          df_predictions['data_val'].append(test_data[k,j])
          df_predictions['use'].append("test")


  if not save_models:
    return np.array(error_scores)
  else:

    df_predictions = pd.DataFrame(df_predictions)
    filename = "useful_csvs/cross_validation_full_model.csv" if not lesion_rsa else "useful_csvs/cross_validation_lesion_model.csv"

    df_predictions.to_csv(filename)
    return df_predictions


# Saves a given list of models to file for comparison in R
# Each model is represented by a dictionary. The dictionary includes the 
# model params, separated from the caused version, enabled version, and
# lesion rsa version, our axes of comparison.

# Takes a list of dictionaries
# Returns a dataframe

def save_model(models, output_file=None):

  output_dict = {'caused_version': [], 'enabled_version': [], 'lesion_rsa': [], 'trial': [], 'response': [], 'model_y': []}

  for model in models:
    params = model['params']
    lesion_rsa = model['lesion_rsa']
    caused_version = model['caused_version']
    enabled_version = model['enabled_version']
    primary_aspect_values, alternative_aspect_values = load_aspects(params['unoise'])

    if not lesion_rsa:
      model_output = rsa.s2(primary_aspect_values, caused_version=caused_version, enabled_version=enabled_version, how_not_affect_param=params['not_affect_param'], stationary_softener=params['stationary_softener'], speaker_optimality=params['speaker_optimality'], alternative_aspect_values=alternative_aspect_values)
    else:
      model_output = rsa.lesion_model(primary_aspect_values, caused_version=caused_version, enabled_version=enabled_version, how_not_affect_param=params['not_affect_param'], stationary_softener=params['stationary_softener'], beta=params['beta'], alternative_aspect_values=alternative_aspect_values)

    for trial in range(30):
      for j in range(len(vocabulary_dataset)):
        verb = vocabulary_dataset[j]

        output_dict['lesion_rsa'].append(lesion_rsa)
        output_dict['caused_version'].append(caused_version)
        output_dict['enabled_version'].append(enabled_version)
        output_dict['trial'].append(trial)
        output_dict['response'].append(verb)
        output_dict['model_y'].append(model_output[j,trial])

  df_output = pd.DataFrame(output_dict)

  if output_file != None:
    df_output.to_csv(output_file)

  return df_output


trials = np.arange(30)
# testing ranges
# unoise_range = [0.9, 1.0, 1.1, 1.2]
# not_affect_param_range = [0.4, 0.5]
# stationary_softener_range = [0.4, 0.5]
# speaker_optimality_range = [1, 1.25]
# comparison_threshold_range = [0.4, 0.5]
# comparison_softener_range = [0.4, 0.5]

# actual ranges
unoise_range = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
not_affect_param_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
stationary_softener_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
speaker_optimality_range = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]
comparison_threshold_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
comparison_softener_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
speaker_optimality_range = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
beta_range = np.arange(1,15,0.5)

############################## Find and save best models to file ###################################

# Uncomment to find best full and no pragmatics models. Considers both combined cause
# and non combined cause models

# cc_full_model = grid_search(trials, lesion_rsa=0, unoise_range=unoise_range, not_affect_param_range=not_affect_param_range, stationary_softener_range=stationary_softener_range, speaker_optimality_range=speaker_optimality_range, beta_range=None, caused_version = "and_hm_or_ws_cc", enabled_version = "or_ws")

# normal_full_model = grid_search(trials, lesion_rsa=0, unoise_range=unoise_range, not_affect_param_range=not_affect_param_range, stationary_softener_range=stationary_softener_range, speaker_optimality_range=speaker_optimality_range, beta_range=None, caused_version = "and_hm_or_ws", enabled_version = "or_ws")

# cc_lesion_model = grid_search(trials, lesion_rsa=1, unoise_range=unoise_range, not_affect_param_range=not_affect_param_range, stationary_softener_range=stationary_softener_range, speaker_optimality_range=None, beta_range=beta_range, caused_version="and_hm_or_ws_cc", enabled_version="or_ws")

# normal_lesion_model = grid_search(trials, lesion_rsa=1, unoise_range=unoise_range, not_affect_param_range=not_affect_param_range, stationary_softener_range=stationary_softener_range, speaker_optimality_range=None, beta_range=beta_range, caused_version="and_hm_or_ws", enabled_version="or_ws")

# print("CC Full Model params")
# print(cc_full_model)
# print()
# print("Full Model params")
# print(normal_full_model)
# print()
# print("CC Lesion Model params")
# print(cc_lesion_model)
# print()
# print("Lesion Model params")
# print(normal_lesion_model)
# print()

# models = [{'params': cc_full_model, 'lesion_rsa': 0, 'caused_version': "and_hm_or_ws_cc", 'enabled_version': "or_ws"}, {'params': normal_full_model, 'lesion_rsa': 0, 'caused_version': "and_hm_or_ws", 'enabled_version': "or_ws"}, {'params': cc_lesion_model, 'lesion_rsa': 1, 'caused_version': "and_hm_or_ws_cc", "enabled_version": "or_ws"}, {'params': normal_lesion_model, 'lesion_rsa': 1, 'caused_version': "and_hm_or_ws", 'enabled_version': "or_ws"}]

# save_model(models, output_file = "useful_csvs/top_models.csv")


############################## Run grid search on best full and no prag model #################################
# np.random.seed(1)

# # Generate and save splits
# splits = generate_splits(trials, 100)

# df_splits = pd.DataFrame({"train": [spl[0] for spl in splits], "test": [spl[1] for spl in splits]})
# df_splits.to_csv("useful_csvs/crossv_splits.csv")

# # Run grid search for full model. Write opt model for each split to file
# start_time = time.time()
# full = cross_validation(splits, 0, unoise_range, not_affect_param_range, stationary_softener_range, speaker_optimality_range, beta_range, caused_version="and_hm_or_ws_cc", enabled_version="or_ws", save_models=True)
# end_time = time.time()


# print("Runtime:", end_time - start_time)
# print()

# # Run grid search for no pragmatics model. Write opt model for each split to file
# start_time = time.time()
# lesion = cross_validation(splits, 1, unoise_range, not_affect_param_range, stationary_softener_range, speaker_optimality_range, beta_range, caused_version="and_hm_or_ws", enabled_version="or_ws", save_models=True)
# end_time = time.time()

# print("Runtime:", end_time - start_time)

######################## Finding enable and lesion best models ###############################

# Uncomment to run code for writing enable models to file

# old_enb_sem_prag = grid_search(trials, lesion_rsa=0, unoise_range=unoise_range, not_affect_param_range=not_affect_param_range, stationary_softener_range=stationary_softener_range, speaker_optimality_range=speaker_optimality_range, beta_range=None, caused_version = "and_hm_or_ws_cc", enabled_version = "or_ws")

# new_enb_sem_prag = grid_search(trials, lesion_rsa=0, unoise_range=unoise_range, not_affect_param_range=not_affect_param_range, stationary_softener_range=stationary_softener_range, speaker_optimality_range=speaker_optimality_range, beta_range=None, caused_version = "and_hm_or_ws_cc", enabled_version = "or_ws_and_nh")

# old_enb_sem_lesion = grid_search(trials, lesion_rsa=1, unoise_range=unoise_range, not_affect_param_range=not_affect_param_range, stationary_softener_range=stationary_softener_range, speaker_optimality_range=None, beta_range=beta_range, caused_version = "and_hm_or_ws", enabled_version = "or_ws")

# new_enb_sem_lesion = grid_search(trials, lesion_rsa=1, unoise_range=unoise_range, not_affect_param_range=not_affect_param_range, stationary_softener_range=stationary_softener_range, speaker_optimality_range=None, beta_range=beta_range, caused_version = "and_hm_or_ws", enabled_version = "or_ws_and_nh")

# print("Old Enb prag model:")
# print(old_enb_sem_prag)
# print()
# print("New Enb prag model:")
# print(new_enb_sem_prag)
# print()
# print("Old Enb lesion model:")
# print(old_enb_sem_lesion)
# print()
# print("New Enb lesion model:")
# print(new_enb_sem_lesion)
# print()

# models = [{'params': old_enb_sem_prag, 'lesion_rsa': 0, 'caused_version': "and_hm_or_ws_cc", 'enabled_version': "or_ws"}, {'params': new_enb_sem_prag, 'lesion_rsa': 0, 'caused_version': "and_hm_or_ws_cc", 'enabled_version': "or_ws_and_nh"}, {'params': old_enb_sem_lesion, 'lesion_rsa': 1, 'caused_version': "and_hm_or_ws", "enabled_version": "or_ws"}, {'params': new_enb_sem_lesion, 'lesion_rsa': 1, 'caused_version': "and_hm_or_ws", 'enabled_version': "or_ws_and_nh"}]

# save_model(models, output_file = "useful_csvs/enable_comparison.csv")