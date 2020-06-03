import numpy as np
import json
import sqlite3
import pandas as pd
import time

vocabulary_dataset = ["cause", "enable", "affect", "no_difference"]
vocabulary = ["caused", "enabled", "affected", "didn't affect"]
w, h, s, m, o = (0, 1, 2, 3, 4)


trial_max = 31
min_trick = True

def caused_subroutine(primary_aspect_values, caused_version, stationary_softener):
  # base caused definition or (whether, sufficiency) and how
  suff_or_necc = np.sum(primary_aspect_values[:,[w,s]], axis=1) - np.prod(primary_aspect_values[:,[w,s]], axis=1)
  caused_base = suff_or_necc * primary_aspect_values[:,h]
  # and moving feature
  if caused_version == "and_h_or_ws" or caused_version == "and_h_or_ws_cc" or caused_version == "and_h_or_ws_cc_dummy":
    return caused_base
  elif caused_version == "and_hm_or_ws" or caused_version == "and_hm_or_ws_cc" or caused_version == "and_hm_or_ws_cc_dummy":
    moving = primary_aspect_values[:,m]
    # soften the moving feature
    soften_moving = (moving + stationary_softener) - (moving * stationary_softener) 
    return caused_base * soften_moving
  else:
    raise Exception("meaning for caused version '{}' not implemented".format(caused_version))


def meaning(utterance, primary_aspect_values, caused_version, enabled_version, how_not_affect_param, stationary_softener, comparison_threshold=None, comparison_softener=None, alternative_aspect_values = None):
  if utterance == "enabled":
    if enabled_version == "or_ws":
      # or whether, sufficiency
      return np.sum(primary_aspect_values[:,[w,s]], axis=1) - np.prod(primary_aspect_values[:,[w,s]], axis=1)
    elif enabled_version == "or_ws_and_nh":
      or_ws = np.sum(primary_aspect_values[:,[w,s]], axis=1) - np.prod(primary_aspect_values[:,[w,s]], axis=1)
      or_ws_and_nh = or_ws * (1 - primary_aspect_values[:,h])
      return or_ws_and_nh

  elif utterance == "caused":
    if "cc" not in caused_version:
      return caused_subroutine(primary_aspect_values, caused_version, stationary_softener)
    elif "dummy" not in caused_version:
      if alternative_aspect_values == None:
        raise Exception("No alternative aspects for computing combined cause")
      # get the meanings scores of the primary candidate
      primary_meaning = caused_subroutine(primary_aspect_values, caused_version, stationary_softener)
      # initialize the aggregator with the scores of the primary candidate
      cause_assessments = np.expand_dims(primary_meaning,0)

      # collect the meanings in an array
      for alternative in alternative_aspect_values:
        alterative_meaning = np.expand_dims(caused_subroutine(alternative, caused_version, stationary_softener), 0)
        cause_assessments = np.concatenate((cause_assessments, alterative_meaning), 0)


      normalized = primary_meaning.copy()
      comp_indicies = np.argwhere(np.sum(~np.isnan(cause_assessments), axis=0) != 1).T[0]
      no_comp_indicies = np.argwhere(np.sum(~np.isnan(cause_assessments), axis=0) == 1).T[0]

      comp_denom = np.nansum(cause_assessments[:, comp_indicies], axis=0)
      comp_denom[comp_denom == 0] = 1.0
      normalized[comp_indicies] = normalized[comp_indicies]/comp_denom

      if min_trick:
        normalized = np.minimum(normalized, primary_meaning)

      return normalized
    else:
      if alternative_aspect_values == None:
        raise Exception("No alternative aspects for computing combined cause")
      if comparison_threshold == None:
        raise Exception("No threshold for dummy comparison")
      if comparison_softener == None:
        raise Exception("No softener for dummy comparison")

      primary_meaning = caused_subroutine(primary_aspect_values, caused_version, stationary_softener)

      alternative_cause_assessments = tuple([np.expand_dims(caused_subroutine(alt, caused_version, stationary_softener), 0) for alt in alternative_aspect_values])

      alternative_matrix = np.nan_to_num(np.concatenate(alternative_cause_assessments, 0))
      above_threshold = alternative_matrix > comparison_threshold
      # print(alternative_matrix > 0.5)
      no_comparator = ~np.any(above_threshold, axis=0)
      soften_no_comp = (no_comparator + comparison_softener) - (no_comparator * comparison_softener)

      return primary_meaning*soften_no_comp



      # print(aggregator.T)
      # print(np.sum(!np.isnan(aggregator), axis=0))

      # Make sure no divide by zero. (Divide by nan is fine)
      # aggregator[aggregator == 0] = 1.0
      # Normalize by the aggregator
      # For cases that don't have a comparator, use their value from primary
      # normalized = primary_meaning/aggregator
      # no_alternative_indicies = np.argwhere(np.isnan(normalized))
      # normalized[no_alternative_indicies] = primary_meaning[no_alternative_indicies]

      # A hack to make sure cc method doesn't inflate caused values
      # if min_trick:
      #   norm_temp = np.expand_dims(normalized, 0)
      #   primary_temp = np.expand_dims(primary_meaning, 0)
      #   comparison = np.concatenate((norm_temp, primary_temp))
      #   final = np.min(comparison, axis=0)

        # return normalized
      #   return final

      # else:
      #   return normalized

      # Return the primary normalized by sum of causal scores of all entities
      # If primary is zero as well as the alternative scores this will divide by zero
      # And return nan. Replace aggregate zeros with 1s. Result will be 0/1 = 0

      # return primary_meaning/aggregator

  elif utterance == "affected":
    affected = primary_aspect_values[:,h]
    return affected

  elif utterance == "didn't affect":
    whether_or_suff = primary_aspect_values[:,w] + primary_aspect_values[:,s] - primary_aspect_values[:,w]*primary_aspect_values[:,s]
    how_and_flip = primary_aspect_values[:,h] * how_not_affect_param
    return 1 - (whether_or_suff + how_and_flip - whether_or_suff*how_and_flip)

  else:
    raise Exception("meaning for utterance '{}' not implemented.".format(utterance))

                      # L0(clip | utterance) \propto P(clip) * P(utterance true | clip)
def l0(primary_aspect_values, caused_version, enabled_version, how_not_affect_param, stationary_softener, comparison_threshold=None, comparison_softener=None, alternative_aspect_values = None):
  n_worlds = primary_aspect_values.shape[0]
  # assert n_worlds == trial_max
  prior = 1./n_worlds
  # number of utterances X number of worlds
  likelihood = np.array([meaning(utterance, primary_aspect_values, caused_version, enabled_version, how_not_affect_param, stationary_softener, comparison_threshold=comparison_threshold, comparison_softener=comparison_softener, alternative_aspect_values=alternative_aspect_values) for utterance in vocabulary])
  unnormalized_l0 = prior*likelihood

  # normalize within each utterance to get distribution over worlds
  normalization_constants = np.sum(unnormalized_l0, axis=1)

  # There may be some utterance which is false for all worlds.
  # In that case, we can't make a well-defined probability distribution over worlds.
  # But we will just return 0 for every world in that group.
  # This is fine, because if the utterance is false for all worlds,
  # we want the likelihood of that utterance given each world to be 0.
  numerator = unnormalized_l0.T
  denominator = normalization_constants

  # numerator is only nonzero if well defined normalization constant exists
  well_defined = normalization_constants > 0

  # For any denominator that is not well defined (i.e. equal to zero)
  # Add 1 to it. The normalization will then be each world (value 0) divided by 1
  denominator = denominator + (1 - well_defined)

  # number of worlds X number of utterances
  return numerator / denominator

# S1(utterance | clip) \propto P(utterance) * L0(clip | utterance)^optimality
def s1(primary_aspect_values, caused_version, enabled_version, how_not_affect_param, stationary_softener, speaker_optimality, comparison_threshold=None, comparison_softener=None, alternative_aspect_values=None):
  prior = 1./len(vocabulary)
  likelihood = l0(primary_aspect_values, caused_version, enabled_version, how_not_affect_param, stationary_softener, comparison_threshold=comparison_threshold, comparison_softener=comparison_softener, alternative_aspect_values=alternative_aspect_values)
  unnormalized_s1 = prior*likelihood**speaker_optimality

  # normalize within each world to get distribution over utterances
  normalization_constants = np.sum(unnormalized_s1, axis=1)
  numerator = unnormalized_s1.T
  denominator = normalization_constants

  # There may be some worlds for which no utterance is true.
  # In that case, we can't make a well-defined probability distribution over utterances.
  # But we will just return 0 for every utterance in that group.
  # This is fine, because if a world cannot be picked out by any utterance,
  # we want the likelihood of that world given each utterance to be 0
  
  # numerator is only nonzero if well defined normalization constant exists
  well_defined = normalization_constants > 0

  # For any denominator that is not well defined (i.e. equal to zero)
  # Add 1 to it. The normalization will then be each world (value 0) divided by 1
  denominator = denominator + (1 - well_defined)

  return numerator / denominator


# L1(clip | utterance) \propto P(clip) * S1(utterance | clip)
# L1(clip | utterance) =  L1(clip | utterance)
def l1(primary_aspect_values, caused_version, enabled_version, how_not_affect_param, stationary_softener, speaker_optimality, comparison_threshold=None, comparison_softener=None, alternative_aspect_values = None):
  n_worlds = primary_aspect_values.shape[0]
  # assert n_worlds == trial_max
  # prior over world
  prior = 1./n_worlds
  # 4 utterances X 32 worlds
  # for each world, there's a likelihood of the utterance
  likelihood = s1(primary_aspect_values, caused_version, enabled_version, how_not_affect_param, stationary_softener, speaker_optimality, comparison_threshold=comparison_threshold, comparison_softener=comparison_softener, alternative_aspect_values=alternative_aspect_values)
  numerator = prior*likelihood

  # normalize within each utterance to get a distribution over  worlds
  denominator = np.sum(numerator, axis=1)
  return numerator.T / denominator

# S2(utterance | clip) \propto P(utterance) * L1(clip | utterance)^optimality
def s2(primary_aspect_values, caused_version, enabled_version, how_not_affect_param, stationary_softener, speaker_optimality, comparison_threshold=None, comparison_softener=None, alternative_aspect_values = None):
  prior = 1./len(vocabulary)
  likelihood = l1(primary_aspect_values, caused_version, enabled_version, how_not_affect_param, stationary_softener, speaker_optimality, comparison_threshold=comparison_threshold, comparison_softener=comparison_softener, alternative_aspect_values=alternative_aspect_values)
  unnormalized_s2 = prior*likelihood**speaker_optimality
  # normalize within each world to get distribution over utterances
  return unnormalized_s2.T / np.sum(unnormalized_s2, axis=1)


def softmax(arr, ax, beta=1):
  exp_arr = np.exp(beta*arr)
  return exp_arr/np.sum(exp_arr, axis=ax)



box_alternative = 0
aspect_set = "trial_set"



# Code to load the data from the sql file
database_path = "../../data/full_database.db"
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

df_data = df_data.assign(vec_response = list(map(convert_verb, df_data['response'])))

trial_counts = df_data.groupby(["trial"]).vec_response.apply(np.sum)

trial_counts = np.array(trial_counts.values.tolist()).T
data_averaged = trial_counts/np.sum(trial_counts, axis = 0)


# Remove attention check and any excess participant dims
# data = data[:30, :, :(len(data_objects) - num_failed_attention_check)]

# Get the average value for each response across participats
# data_averaged = np.mean(data, axis=2).T

# caused_version = "and_hm_or_ws"

def load_aspects(uncertainty_noise, aspect_version="standard"):

  if aspect_version == "modified":
    clips_data = json.load(open("aspects/aspects_noise_{}.json".format(uncertainty_noise)))
  else:
    clips_data = json.load(open("priorinfo/clang3_exp_extended_trials_samples_1000__uncertainty_noise_{}__seed_123__whether_test_version_basic__gate_alternative_0__box_alternative_0_vector_representation_new.json".format(uncertainty_noise)))

  if isinstance(clips_data[0], list):

    primary_candidate = clips_data[0]
    alternative_candidates = clips_data[1:]

    primary_aspect_values = np.array([tr['rep'] for tr in primary_candidate])[:,:4]

    alternative_aspect_values = []

    for candidate_aspects in alternative_candidates:
      alternative_aspects = np.array([tr['rep'] if tr['rep'] != None else np.zeros(7) for tr in candidate_aspects])[:,:4]
      alternative_aspect_values.append(alternative_aspects)

  return primary_aspect_values, alternative_aspect_values



  # clips_data = json.load(open("aspects_paper/experiment_trials_samples_1000__uncertainty_noise_{}__gate_alternative_0__box_alternative_0_vector_representation.json".format(unoise)))

  # primary_aspect_values = np.array([tr['rep'] for tr in clips_data])[:,:4]

  # return primary_aspect_values


def compute_error(model_output, trial_set):
  data_set = data_averaged[:, trial_set]
  model_set = model_output[:, trial_set]

  sq_err = np.sum((data_set - model_set)**2)

  return sq_err


def lesion_model(primary_aspect_values, caused_version, enabled_version, how_not_affect_param, stationary_softener, beta, comparison_threshold=None, comparison_softener=None, alternative_aspect_values=None):
  semantic_values = np.vstack([meaning(word, primary_aspect_values, caused_version, enabled_version, how_not_affect_param, stationary_softener, comparison_threshold=comparison_threshold, comparison_softener=comparison_softener, alternative_aspect_values=alternative_aspect_values) for word in vocabulary])
  semantic_values = softmax(semantic_values, 0, beta=beta)
  return semantic_values



# For a trial set and set of ranges across parameters, find the parameter setting
# That minimizes the squared error on the trial set
def grid_search(trial_set, lesion_rsa, unoise_range, not_affect_param_range, stationary_softener_range, speaker_optimality_range, beta_range, caused_versions = ["and_hm_or_ws"], enabled_versions = ["or_ws"], comparison_threshold_range=None, comparison_softener_range=None, aspect_version="modified", testing=False):

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

    if aspect_version == "modified":
      clips_data = json.load(open("aspects/aspects_noise_{}.json".format(uncertainty_noise)))
    else:
      clips_data = json.load(open("priorinfo/clang3_exp_extended_trials_samples_1000__uncertainty_noise_{}__seed_123__whether_test_version_basic__gate_alternative_0__box_alternative_0_vector_representation_new.json".format(uncertainty_noise)))

    if isinstance(clips_data[0], list):

      primary_candidate = clips_data[0]
      alternative_candidates = clips_data[1:]

      primary_aspect_values = np.array([tr['rep'] for tr in primary_candidate])[:,:4]

      alternative_aspect_values = []

      for candidate_aspects in alternative_candidates:
        alternative_aspects = np.array([tr['rep'] if tr['rep'] != None else np.zeros(7) for tr in candidate_aspects])[:,:4]
        alternative_aspect_values.append(alternative_aspects)

    else:
      primary_aspect_values = np.array([tr['rep'] for tr in clips_data])[:,:4]
      alternative_aspect_values = None

    for how_not_affect_param in not_affect_param_range:

      for stationary_softener in stationary_softener_range:

        for enabled_version in enabled_versions:

          for caused_version in caused_versions:

            if "dummy" not in caused_version:

              if lesion_rsa:

                # print(alternative_aspect_values)

                meanings = np.vstack([meaning(word, primary_aspect_values, caused_version, enabled_version, how_not_affect_param, stationary_softener, alternative_aspect_values=alternative_aspect_values) for word in vocabulary])

                for beta in beta_range:

                  semantic_values = softmax(meanings, 0, beta=beta)
                  sq_err = compute_error(semantic_values, trial_set)



                  if sq_err < opt_set['error']:
                    opt_set['unoise'] = uncertainty_noise
                    opt_set['not_affect_param'] = how_not_affect_param
                    opt_set['stationary_softener'] = stationary_softener
                    opt_set['beta'] = beta
                    opt_set['error'] = sq_err

              else:

                for speaker_optimality in speaker_optimality_range:
                  model_output = s2(primary_aspect_values, caused_version, enabled_version, how_not_affect_param, stationary_softener, speaker_optimality, alternative_aspect_values=alternative_aspect_values)
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

                    meanings = np.vstack([meaning(word, primary_aspect_values, caused_version, enabled_version, how_not_affect_param, stationary_softener, comparison_threshold, comparison_softener, alternative_aspect_values=alternative_aspect_values) for word in vocabulary])

                    for beta in beta_range:
                      semantic_values = softmax(meanings, 0, beta=beta)
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
                      model_output = s2(primary_aspect_values, caused_version, enabled_version, how_not_affect_param, stationary_softener, speaker_optimality, comparison_threshold, comparison_softener, alternative_aspect_values=alternative_aspect_values)

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



# Split trial set in half
# Return a tuple where the first set is train and the second is test
def generate_splits(trials, num_splits):
  stationary_trials = {14, 17, 21, 27}
  splits = []
  i = 0
  while i < num_splits:
    train = np.sort(np.random.choice(trials, size=15, replace=False))
    test = np.setdiff1d(trials, train)
    train_set = set(train)

    if len(train_set.intersection(stationary_trials)) > 0:
      splits.append((train, test))
      i += 1

  return splits


def cross_validation(splits, lesion_rsa, unoise_range, not_affect_param_range, stationary_softener_range, speaker_optimality_range, beta_range, caused_version=["and_hm_or_ws"], enabled_version = ["or_ws"], aspect_version="modified"):

  # splits = generate_splits(trials, num_splits)
  error_scores = []

  for i in range(len(splits)):
    if i % 10 == 0:
      print("Split", str(i))
    spl = splits[i]
    train = spl[0]
    test = spl[1]

    parameters = grid_search(train, lesion_rsa, unoise_range, not_affect_param_range, stationary_softener_range, speaker_optimality_range, beta_range, caused_versions=caused_version, enabled_versions=enabled_version, aspect_version=aspect_version)

    primary_aspect_values, alternative_aspect_values = load_aspects(parameters['unoise'], aspect_version=aspect_version)
    not_affect_param = parameters['not_affect_param']
    stationary_softener = parameters['stationary_softener']
    fine_tune = parameters['speaker_optimality'] if not lesion_rsa else parameters['beta']
    if not lesion_rsa:
      model_output = s2(primary_aspect_values, caused_version[0], enabled_version[0], not_affect_param, stationary_softener, fine_tune, alternative_aspect_values=alternative_aspect_values)
    else:
      model_output = lesion_model(primary_aspect_values, caused_version[0], enabled_version[0], not_affect_param, stationary_softener, fine_tune, alternative_aspect_values=alternative_aspect_values)
      # semantic_values = np.vstack([meaning(word, primary_aspect_values, caused_version[0], enabled_version[0], not_affect_param, stationary_softener) for word in vocabulary])
      # model_output = softmax(semantic_values, ax=0, beta=fine_tune)

    split_error = compute_error(model_output, test)

    error_scores.append(split_error)

  return np.array(error_scores)


def save_models(splits, lesion_rsa, unoise_range, not_affect_param_range, stationary_softener_range, speaker_optimality_range, beta_range, caused_version=["and_hm_or_ws"], enabled_version = ["or_ws"], aspect_version="modified"):

  # splits = generate_splits(trials, num_splits)
  # error_scores = []

  df_predictions = {"split": [], "lesion_rsa": [], "caused_version": [], "model_params": [], "trial": [], "verb": [], "model_pred": [], "data_val": [], "use": []}

  for i in range(len(splits)):
    if i % 10 == 0:
      print("Split", str(i))
    spl = splits[i]
    train = spl[0]
    test = spl[1]

    parameters = grid_search(train, lesion_rsa, unoise_range, not_affect_param_range, stationary_softener_range, speaker_optimality_range, beta_range, caused_versions=caused_version, enabled_versions=enabled_version)

    primary_aspect_values, alternative_aspect_values = load_aspects(parameters['unoise'], aspect_version=aspect_version)
    not_affect_param = parameters['not_affect_param']
    stationary_softener = parameters['stationary_softener']
    fine_tune = parameters['speaker_optimality'] if not lesion_rsa else parameters['beta']
    if not lesion_rsa:
      model_output = s2(primary_aspect_values, caused_version[0], enabled_version[0], not_affect_param, stationary_softener, fine_tune, alternative_aspect_values=alternative_aspect_values)
    else:
      model_output = lesion_model(primary_aspect_values, caused_version[0], enabled_version[0], not_affect_param, stationary_softener, fine_tune, alternative_aspect_values=alternative_aspect_values)
      # semantic_values = np.vstack([meaning(word, primary_aspect_values, caused_version[0], enabled_version[0], not_affect_param, stationary_softener) for word in vocabulary])
      # model_output = softmax(semantic_values, ax=0, beta=fine_tune)


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
        df_predictions['caused_version'] = caused_version[0]
        df_predictions['model_params'].append([parameters['unoise'], not_affect_param, stationary_softener] + ([fine_tune, None] if not lesion_rsa else [None, fine_tune]))
        df_predictions['trial'].append(train_trial)
        df_predictions['verb'].append(verb)
        df_predictions['model_pred'].append(train_model[k,j])
        df_predictions['data_val'].append(train_data[k,j])
        df_predictions['use'].append("train")


        df_predictions['split'].append(i)
        df_predictions['lesion_rsa'].append(lesion_rsa)
        df_predictions['caused_version'] = caused_version[0]
        df_predictions['model_params'].append([parameters['unoise'], not_affect_param, stationary_softener] + ([fine_tune, None] if not lesion_rsa else [None, fine_tune]))
        df_predictions['trial'].append(test_trial)
        df_predictions['verb'].append(verb)
        df_predictions['model_pred'].append(test_model[k,j])
        df_predictions['data_val'].append(test_data[k,j])
        df_predictions['use'].append("test")



    # split_error = compute_error(model_output, test)

    # error_scores.append(split_error)

  return pd.DataFrame(df_predictions)

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
# speaker_optimality_range = np.arange(0.5,3,0.1)
beta_range = np.arange(1,15,0.5)


############################## Writing opt models to file by split #################################
# np.random.seed(1)

# splits = generate_splits(trials, 100)

# df_splits = pd.DataFrame({"train": [spl[0] for spl in splits], "test": [spl[1] for spl in splits]})
# df_splits.to_csv("crossv_splits.csv")

# start_time = time.time()
# full = save_models(splits, 0, unoise_range, not_affect_param_range, stationary_softener_range, speaker_optimality_range, beta_range)
# end_time = time.time()

# # print("Average Full Error:", np.mean(full_error))
# full.to_csv("split_performance_full.csv")
# print("Runtime:", end_time - start_time)
# print()

# start_time = time.time()
# lesion = save_models(splits, 1, unoise_range, not_affect_param_range, stationary_softener_range, speaker_optimality_range, beta_range)
# end_time = time.time()

# lesion.to_csv("split_performance_lesion.csv")
# # print("Average Lesion Error:", np.mean(lesion_error))
# print("Runtime:", end_time - start_time)

######################## Finding enable and lesion best models ###############################
old_enb_sem_prag = grid_search(trials, lesion_rsa=0, unoise_range=unoise_range, not_affect_param_range=not_affect_param_range, stationary_softener_range=stationary_softener_range, speaker_optimality_range=speaker_optimality_range, beta_range=None, caused_versions = ["and_hm_or_ws_cc"], enabled_versions = ["or_ws"], aspect_version="modified")

new_enb_sem_prag = grid_search(trials, lesion_rsa=0, unoise_range=unoise_range, not_affect_param_range=not_affect_param_range, stationary_softener_range=stationary_softener_range, speaker_optimality_range=speaker_optimality_range, beta_range=None, caused_versions = ["and_hm_or_ws_cc"], enabled_versions = ["or_ws_and_nh"], aspect_version="modified")

old_enb_sem_lesion = grid_search(trials, lesion_rsa=1, unoise_range=unoise_range, not_affect_param_range=not_affect_param_range, stationary_softener_range=stationary_softener_range, speaker_optimality_range=None, beta_range=beta_range, caused_versions = ["and_hm_or_ws"], enabled_versions = ["or_ws"], aspect_version="modified")

new_enb_sem_lesion = grid_search(trials, lesion_rsa=1, unoise_range=unoise_range, not_affect_param_range=not_affect_param_range, stationary_softener_range=stationary_softener_range, speaker_optimality_range=None, beta_range=beta_range, caused_versions = ["and_hm_or_ws"], enabled_versions = ["or_ws_and_nh"], aspect_version="modified")

print("Old Enb prag model:")
print(old_enb_sem_prag)
print()
print("New Enb prag model:")
print(new_enb_sem_prag)
print()
print("Old Enb lesion model:")
print(old_enb_sem_lesion)
print()
print("New Enb lesion model:")
print(new_enb_sem_lesion)
print()

param_settings = [(old_enb_sem_prag, "or_ws", 0), (new_enb_sem_prag, "or_ws_and_nh", 0), (old_enb_sem_lesion, "or_ws", 1), (new_enb_sem_lesion, "or_ws_and_nh", 1)]

enb_comp_dict = {"lesion_rsa": [], "enabled_version": [], "trial": [], "response": [], "model_y": []}

for i in range(len(param_settings)):
  params, enabled_version, lesion_rsa = param_settings[i]
  primary_aspect_values, alternative_aspect_values = load_aspects(params['unoise'], aspect_version="modified")
  if not lesion_rsa:
    model_output = s2(primary_aspect_values, caused_version="and_hm_or_ws_cc", enabled_version=enabled_version, how_not_affect_param=params['not_affect_param'], stationary_softener=params['stationary_softener'], speaker_optimality=params['speaker_optimality'], alternative_aspect_values=alternative_aspect_values)
  else:
    model_output = lesion_model(primary_aspect_values, caused_version="and_hm_or_ws", enabled_version=enabled_version, how_not_affect_param=params['not_affect_param'], stationary_softener=params['stationary_softener'], beta=params['beta'], alternative_aspect_values=alternative_aspect_values)


  for trial in range(30):
    for j in range(len(vocabulary_dataset)):
      verb = vocabulary_dataset[j]

      enb_comp_dict['lesion_rsa'].append(lesion_rsa)
      enb_comp_dict['enabled_version'].append(enabled_version)
      enb_comp_dict['trial'].append(trial)
      enb_comp_dict['response'].append(verb)
      enb_comp_dict['model_y'].append(model_output[j,trial])

df_enb_comp = pd.DataFrame(enb_comp_dict)
df_enb_comp.to_csv('useful_csvs/enabled_definition.csv')



################### grid search for best combined cause model full trial set ######################

# primary_aspect_values, alternative_aspect_values = load_aspects(1.0)

# temp = caused_subroutine(alternative_aspect_values[0], "and_hm_or_ws_cc", 0.4)

# primary_caused = caused_subroutine(primary_aspect_values, "and_hm_or_ws_cc", 0.4)
# alternative_caused = caused_subroutine(alternative_aspect_values[0], "and_hm_or_ws_cc", 0.4)

# caused_meaning = meaning("caused", primary_aspect_values, "and_hm_or_ws_cc", "or_ws", 0.4, 0.4, alternative_aspect_values = alternative_aspect_values)

# trial_nums = np.arange(primary_caused.shape[0])

# comparison = np.concatenate((np.expand_dims(trial_nums,1), np.expand_dims(caused_meaning,1), np.expand_dims(primary_caused, 1)), axis=1)
# print(comparison)

# caused_meaning_old = meaning("caused", primary_aspect_values, "and_hm_or_ws", "or_ws", 0.4, 0.4, alternative_aspect_values = None)


# cc_full_model = grid_search(trials, lesion_rsa=0, unoise_range=unoise_range, not_affect_param_range=not_affect_param_range, stationary_softener_range=stationary_softener_range, speaker_optimality_range=speaker_optimality_range, beta_range=None, caused_versions = ["and_hm_or_ws_cc"], enabled_versions = ["or_ws"])

# normal_full_model = grid_search(trials, lesion_rsa=0, unoise_range=unoise_range, not_affect_param_range=not_affect_param_range, stationary_softener_range=stationary_softener_range, speaker_optimality_range=speaker_optimality_range, beta_range=None, caused_versions = ["and_hm_or_ws"], enabled_versions = ["or_ws"])

# cc_lesion_model = grid_search(trials, lesion_rsa=1, unoise_range=unoise_range, not_affect_param_range=not_affect_param_range, stationary_softener_range=stationary_softener_range, speaker_optimality_range=None, beta_range=beta_range, caused_versions=["and_hm_or_ws_cc"], enabled_versions=["or_ws"])

# normal_lesion_model = grid_search(trials, lesion_rsa=1, unoise_range=unoise_range, not_affect_param_range=not_affect_param_range, stationary_softener_range=stationary_softener_range, speaker_optimality_range=None, beta_range=beta_range, caused_versions=["and_hm_or_ws"], enabled_versions=["or_ws"])

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



# param_settings = [(cc_full_model, "and_hm_or_ws_cc", 0), (normal_full_model, "and_hm_or_ws", 0), (cc_lesion_model, "and_hm_or_ws_cc", 1), (normal_lesion_model, "and_hm_or_ws", 1)]

# primary_aspect_values, alternative_aspect_values = load_aspects(normal_lesion_model['unoise'])

# model_output = lesion_model(primary_aspect_values, caused_version="and_hm_or_ws", enabled_version="or_ws", how_not_affect_param=normal_lesion_model['not_affect_param'], stationary_softener=normal_lesion_model['stationary_softener'], beta=normal_lesion_model['beta'], alternative_aspect_values=None)

# cc_comp_dict = {"caused_version": [], "lesion_rsa": [], "trial": [], "response": [], "model_y": []}

# for i in range(len(param_settings)):
#   params, caused_version, lesion_rsa = param_settings[i]
#   primary_aspect_values, alternative_aspect_values = load_aspects(params['unoise'])

#   if not lesion_rsa:
#     model_output = s2(primary_aspect_values, caused_version=caused_version, enabled_version="or_ws", how_not_affect_param=params['not_affect_param'], stationary_softener=params['stationary_softener'], speaker_optimality=params['speaker_optimality'], alternative_aspect_values=alternative_aspect_values)
#   else:
#     model_output = lesion_model(primary_aspect_values, caused_version=caused_version, enabled_version="or_ws", how_not_affect_param=params['not_affect_param'], stationary_softener=params['stationary_softener'], beta=params['beta'], alternative_aspect_values=alternative_aspect_values)


#   for trial in range(30):
#     for j in range(len(vocabulary_dataset)):
#       verb = vocabulary_dataset[j]

#       cc_comp_dict['caused_version'].append(caused_version)
#       cc_comp_dict['lesion_rsa'].append(lesion_rsa)
#       cc_comp_dict['trial'].append(trial)
#       cc_comp_dict['response'].append(verb)
#       cc_comp_dict['model_y'].append(model_output[j,trial])

# df_cc_comp = pd.DataFrame(cc_comp_dict)
# df_cc_comp.to_csv('cc_comp.csv')


##################### Compare models with modified alternative choosing #####################

# start = time.time()

# modified_aspects_cc_full = grid_search(trials, lesion_rsa=0, unoise_range=unoise_range, not_affect_param_range=not_affect_param_range, stationary_softener_range=stationary_softener_range, speaker_optimality_range=speaker_optimality_range, beta_range=None, caused_versions = ["and_hm_or_ws_cc"], enabled_versions = ["or_ws"], aspect_version="modified")

# print("Modified Aspects CC full params")
# print(modified_aspects_cc_full)
# print()

# modified_aspects_cc_dummy_full = grid_search(trials, lesion_rsa=0, unoise_range=unoise_range, not_affect_param_range=not_affect_param_range, stationary_softener_range=stationary_softener_range, speaker_optimality_range=speaker_optimality_range, beta_range=None, caused_versions=['and_hm_or_ws_cc_dummy'], enabled_versions=['or_ws'], comparison_threshold_range=comparison_threshold_range, comparison_softener_range=comparison_softener_range, aspect_version="modified")

# print("Modified Aspects CC dummy full params")
# print(modified_aspects_cc_dummy_full)
# print()

# modified_aspects_normal_cause_full = grid_search(trials, lesion_rsa=0, unoise_range=unoise_range, not_affect_param_range=not_affect_param_range, stationary_softener_range=stationary_softener_range, speaker_optimality_range=speaker_optimality_range, beta_range=None, caused_versions = ["and_hm_or_ws"], enabled_versions = ["or_ws"], aspect_version="modified")

# print("Modified Aspects normal cause full params")
# print(modified_aspects_normal_cause_full)
# print()

# standard_aspects_cc_full = grid_search(trials, lesion_rsa=0, unoise_range=unoise_range, not_affect_param_range=not_affect_param_range, stationary_softener_range=stationary_softener_range, speaker_optimality_range=speaker_optimality_range, beta_range=None, caused_versions=["and_hm_or_ws_cc"], enabled_versions=["or_ws"], aspect_version="standard")

# print("Standard Aspects CC full params")
# print(standard_aspects_cc_full)
# print()

# standard_aspects_cc_dummy_full = grid_search(trials, lesion_rsa=0, unoise_range=unoise_range, not_affect_param_range=not_affect_param_range, stationary_softener_range=stationary_softener_range, speaker_optimality_range=speaker_optimality_range, beta_range=None, caused_versions=['and_hm_or_ws_cc_dummy'], enabled_versions=['or_ws'], comparison_threshold_range=comparison_threshold_range, comparison_softener_range=comparison_softener_range, aspect_version="standard")

# print("Standard Aspects CC dummy full params")
# print(standard_aspects_cc_dummy_full)
# print()

# standard_aspects_normal_cause_full = grid_search(trials, lesion_rsa=0, unoise_range=unoise_range, not_affect_param_range=not_affect_param_range, stationary_softener_range=stationary_softener_range, speaker_optimality_range=speaker_optimality_range, beta_range=None, caused_versions=["and_hm_or_ws"], enabled_versions=["or_ws"], aspect_version="standard")

# print("Standard Aspects normal cause full params")
# print(standard_aspects_normal_cause_full)
# print()

# modified_aspects_cc_lesion = grid_search(trials, lesion_rsa=1, unoise_range=unoise_range, not_affect_param_range=not_affect_param_range, stationary_softener_range=stationary_softener_range, speaker_optimality_range=None, beta_range=beta_range, caused_versions = ["and_hm_or_ws_cc"], enabled_versions = ["or_ws"], aspect_version="modified")

# print("Modified Aspects CC lesion params")
# print(modified_aspects_cc_lesion)
# print()

# modified_aspects_cc_dummy_lesion = grid_search(trials, lesion_rsa=1, unoise_range=unoise_range, not_affect_param_range=not_affect_param_range, stationary_softener_range=stationary_softener_range, speaker_optimality_range=None, beta_range=beta_range, caused_versions = ["and_hm_or_ws_cc_dummy"], enabled_versions = ["or_ws"], comparison_threshold_range=comparison_threshold_range, comparison_softener_range=comparison_softener_range, aspect_version="modified")

# print("Modified Aspects CC dummy lesion params")
# print(modified_aspects_cc_dummy_lesion)
# print()

# modified_aspects_normal_cause_lesion = grid_search(trials, lesion_rsa=1, unoise_range=unoise_range, not_affect_param_range=not_affect_param_range, stationary_softener_range=stationary_softener_range, speaker_optimality_range=None, beta_range=beta_range, caused_versions = ["and_hm_or_ws"], enabled_versions = ["or_ws"], aspect_version="modified")

# print("Modified Aspects normal cause lesion params")
# print(modified_aspects_normal_cause_lesion)
# print()

# standard_aspects_cc_lesion = grid_search(trials, lesion_rsa=1, unoise_range=unoise_range, not_affect_param_range=not_affect_param_range, stationary_softener_range=stationary_softener_range, speaker_optimality_range=None, beta_range=beta_range, caused_versions = ["and_hm_or_ws_cc"], enabled_versions = ["or_ws"], aspect_version="standard")

# print("Standard Aspects CC lesion params")
# print(standard_aspects_cc_lesion)
# print()

# standard_aspects_cc_dummy_lesion = grid_search(trials, lesion_rsa=1, unoise_range=unoise_range, not_affect_param_range=not_affect_param_range, stationary_softener_range=stationary_softener_range, speaker_optimality_range=None, beta_range=beta_range, caused_versions = ["and_hm_or_ws_cc_dummy"], enabled_versions = ["or_ws"], comparison_threshold_range=comparison_threshold_range, comparison_softener_range=comparison_softener_range, aspect_version="standard")

# print("Standard Aspects CC dummy lesion params")
# print(standard_aspects_cc_dummy_lesion)
# print()

# standard_aspects_normal_cause_lesion = grid_search(trials, lesion_rsa=1, unoise_range=unoise_range, not_affect_param_range=not_affect_param_range, stationary_softener_range=stationary_softener_range, speaker_optimality_range=None, beta_range=beta_range, caused_versions = ["and_hm_or_ws"], enabled_versions = ["or_ws"], aspect_version="standard")

# print("Standard Aspects normal cause lesion params")
# print(standard_aspects_normal_cause_lesion)
# print()

# primary_aspect_values, alternative_aspect_values = load_aspects(1.0, aspect_version="modified")

# meaning('caused', primary_aspect_values, caused_version="and_hm_or_ws_cc_dummy", enabled_version="or_ws", how_not_affect_param=0.5, stationary_softener=0.5, comparison_threshold=0.5, comparison_softener=0.5, alternative_aspect_values = alternative_aspect_values)


# primary_aspect_values, alternative_aspect_values = load_aspects(modified_aspects_cc_dummy['unoise'], aspect_version="modified")

# cause_cc_dummy_aspects_modified = meaning('caused', primary_aspect_values, caused_version='and_hm_or_ws_cc', enabled_version='or_ws', how_not_affect_param=modified_aspects_cc_dummy['not_affect_param'], stationary_softener=modified_aspects_cc_dummy['stationary_softener'], comparison_threshold=modified_aspects_cc_dummy['comparison_threshold'], comparison_softener=modified_aspects_cc_dummy['comparison_softener'], alternative_aspect_values=alternative_aspect_values)

# primary_aspect_values, alternative_aspect_values = load_aspects(modified_aspects_normal_cause['unoise'], aspect_version="modified")

# cause_normal_aspects_modified = meaning('caused', primary_aspect_values, caused_version="and_hm_or_ws", enabled_version="or_ws", how_not_affect_param=modified_aspects_normal_cause['not_affect_param'], stationary_softener=modified_aspects_normal_cause['stationary_softener'], alternative_aspect_values=alternative_aspect_values)

# dummy_cause_comparison = np.concatenate((np.expand_dims(np.arange(32), 1), np.expand_dims(cause_cc_dummy_aspects_modified, 1), np.expand_dims(cause_normal_aspects_modified, 1)), 1)

# print(dummy_cause_comparison)


# param_settings = [(modified_aspects_cc_full, "and_hm_or_ws_cc", "modified", 0), (modified_aspects_cc_dummy_full, "and_hm_or_ws_cc_dummy", "modified", 0), (modified_aspects_normal_cause_full, "and_hm_or_ws", "modified", 0), (standard_aspects_cc_full, "and_hm_or_ws_cc", "standard", 0), (standard_aspects_cc_dummy_full, "and_hm_or_ws_cc_dummy", "standard", 0), (standard_aspects_normal_cause_full, "and_hm_or_ws", "standard", 0), (modified_aspects_cc_lesion, "and_hm_or_ws_cc", "modified", 1), (modified_aspects_cc_dummy_lesion, "and_hm_or_ws_cc_dummy", "modified", 1), (modified_aspects_normal_cause_lesion, "and_hm_or_ws", "modified", 1), (standard_aspects_cc_lesion, "and_hm_or_ws_cc", "standard", 1), (standard_aspects_cc_dummy_lesion, "and_hm_or_ws_cc_dummy", "standard", 1), (standard_aspects_normal_cause_lesion, "and_hm_or_ws", "standard", 1)]

# aspect_cc_lesion_dict = {"caused_version": [], "aspect_version": [], "lesion_rsa": [], "trial": [], "response": [], "model_y": []}

# for i in range(len(param_settings)):
#   params, caused_version, aspect_version, lesion_rsa = param_settings[i]
#   primary_aspect_values, alternative_aspect_values = load_aspects(params['unoise'], aspect_version=aspect_version)

#   if "dummy" in caused_version:
#     if lesion_rsa:
#       model_output = lesion_model(primary_aspect_values, caused_version=caused_version, enabled_version="or_ws", how_not_affect_param=params['not_affect_param'], stationary_softener=params['stationary_softener'], beta=params['beta'], comparison_threshold=params['comparison_threshold'], comparison_softener=params['comparison_softener'], alternative_aspect_values=alternative_aspect_values)
#     else:
#       model_output = s2(primary_aspect_values, caused_version=caused_version, enabled_version="or_ws", how_not_affect_param=params['not_affect_param'], stationary_softener=params['stationary_softener'], speaker_optimality=params['speaker_optimality'], comparison_threshold=params['comparison_threshold'], comparison_softener=params['comparison_softener'], alternative_aspect_values=alternative_aspect_values)
#   else:
#     if lesion_rsa:
#       model_output = lesion_model(primary_aspect_values, caused_version=caused_version, enabled_version="or_ws", how_not_affect_param=params['not_affect_param'], stationary_softener=params['stationary_softener'], beta=params['beta'], alternative_aspect_values=alternative_aspect_values)
#     else:
#       model_output = s2(primary_aspect_values, caused_version=caused_version, enabled_version="or_ws", how_not_affect_param=params['not_affect_param'], stationary_softener=params['stationary_softener'], speaker_optimality=params['speaker_optimality'], alternative_aspect_values=alternative_aspect_values)

#   for trial in range(30):
#     for j in range(len(vocabulary_dataset)):
#       verb = vocabulary_dataset[j]

#       aspect_cc_lesion_dict['caused_version'].append(caused_version)
#       aspect_cc_lesion_dict['aspect_version'].append(aspect_version)
#       aspect_cc_lesion_dict['lesion_rsa'].append(lesion_rsa)
#       aspect_cc_lesion_dict['trial'].append(trial)
#       aspect_cc_lesion_dict['response'].append(verb)
#       aspect_cc_lesion_dict['model_y'].append(model_output[j,trial])

# df_aspect_cc_lesion = pd.DataFrame(aspect_cc_lesion_dict)
# df_aspect_cc_lesion.to_csv('useful_csvs/aspect_cc_lesion.csv')

# print("Runtime")
# print(time.time() - start)




#################### Running cross validation for best model lesion and full #######################
# Uncomment to run the grid search
# np.random.seed(1)

# splits = generate_splits(trials, 100)

# df_splits = pd.DataFrame({"train": [spl[0] for spl in splits], "test": [spl[1] for spl in splits]})
# df_splits.to_csv("crossv_splits.csv")

# df_splits = pd.read_csv("crossv_splits.csv")

# start_time = time.time()
# full_error = cross_validation(splits, 0, unoise_range, not_affect_param_range, stationary_softener_range, speaker_optimality_range, beta_range, caused_version=["and_hm_or_ws_cc"], aspect_version="modified")
# end_time = time.time()

# print("Average Full Error:", np.mean(full_error))
# print("Runtime:", end_time - start_time)
# print()

# start_time = time.time()
# lesion_error = cross_validation(splits, 1, unoise_range, not_affect_param_range, stationary_softener_range, speaker_optimality_range, beta_range, caused_version=["and_hm_or_ws"], aspect_version="modified")
# end_time = time.time()

# print("Average Lesion Error:", np.mean(lesion_error))
# print("Runtime:", end_time - start_time)

# df_error = pd.DataFrame({"full_model": full_error, "lesion_model": lesion_error})
# df_error.to_csv("crossv_error_dist.csv")


# ### 
# # Uncomment to load and plot error graphs
# from matplotlib import pyplot as plt
# import seaborn as sns

# df_error = pd.read_csv("crossv_error_dist.csv")

# full_error = np.sort(np.array(df_error.full_model))
# lesion_error = np.sort(np.array(df_error.lesion_model))

# print("Full Ave Error:", np.mean(full_error))
# print("Full 2.5%:", (full_error[1] + full_error[2])/2)
# print("Full 97.5%:", (full_error[96] + full_error[97])/2)
# print()
# print("Lesion Ave Error:", np.mean(lesion_error))
# print("Lesion 2.5%:", (lesion_error[1] + lesion_error[2])/2)
# print("Lesion 97.5%:", (lesion_error[96] + lesion_error[97])/2)
# print()

# sns.distplot(full_error, color='green', label="Full Model")
# sns.distplot(lesion_error, color='blue', label="Lesion Model")
# plt.title("Cross Validation Error Comparison")
# plt.legend()

# plt.show()


##################### Write Best Models to file #####################

# modified_aspects_cc_full = grid_search(trials, lesion_rsa=0, unoise_range=unoise_range, not_affect_param_range=not_affect_param_range, stationary_softener_range=stationary_softener_range, speaker_optimality_range=speaker_optimality_range, beta_range=None, caused_versions = ["and_hm_or_ws_cc"], enabled_versions = ["or_ws"], aspect_version="modified")

# print("Modified Aspects CC full params")
# print(modified_aspects_cc_full)
# print()


# modified_aspects_normal_cause_lesion = grid_search(trials, lesion_rsa=1, unoise_range=unoise_range, not_affect_param_range=not_affect_param_range, stationary_softener_range=stationary_softener_range, speaker_optimality_range=None, beta_range=beta_range, caused_versions = ["and_hm_or_ws"], enabled_versions = ["or_ws"], aspect_version="modified")

# print("Modified Aspects normal cause lesion params")
# print(modified_aspects_normal_cause_lesion)
# print()




# # Best model full
# unoise = 0.9
# not_affect_param = 0.5
# stationary_softener = 0.4
# speaker_optimality = 1.5
# aspect_version = "modified"
# caused_version = "and_hm_or_ws_cc"

# primary_aspect_values, alternative_aspect_values = load_aspects(unoise, aspect_version=aspect_version)

# cause_meanings = meaning("caused", primary_aspect_values, caused_version, "or_ws", not_affect_param, stationary_softener, comparison_threshold=None, comparison_softener=None, alternative_aspect_values = alternative_aspect_values)

# full = s2(primary_aspect_values, caused_version=caused_version, enabled_version="or_ws", how_not_affect_param=not_affect_param, stationary_softener=stationary_softener, speaker_optimality=speaker_optimality, alternative_aspect_values=alternative_aspect_values)


# # Best model lesion
# unoise = 0.9
# not_affect_param = 0.1
# stationary_softener = 0.8
# beta = 2.5
# aspect_version = "modified"
# caused_version = "and_hm_or_ws"

# primary_aspect_values, alternative_aspect_values = load_aspects(unoise, aspect_version=aspect_version)

# lesion = lesion_model(primary_aspect_values, caused_version=caused_version, enabled_version="or_ws", how_not_affect_param=not_affect_param, stationary_softener=stationary_softener, beta=beta, alternative_aspect_values=alternative_aspect_values)

# best_model_dict = {'model_version': [], 'trial': [], 'response': [], 'model_y': [], 'data_y': []}


# for trial_index in range(30):
#   for response_index in range(len(vocabulary_dataset)):
#     response = vocabulary_dataset[response_index]

#     full_value = full[response_index, trial_index]
#     lesion_value = lesion[response_index, trial_index]
#     data_value = data_averaged[response_index, trial_index]

#     best_model_dict['model_version'].append('Full Model')
#     best_model_dict['trial'].append(trial_index)
#     best_model_dict['response'].append(response)
#     best_model_dict['model_y'].append(full_value)
#     best_model_dict['data_y'].append(data_value)
#     best_model_dict['model_version'].append('No Pragmatics')
#     best_model_dict['trial'].append(trial_index)
#     best_model_dict['response'].append(response)
#     best_model_dict['model_y'].append(lesion_value)
#     best_model_dict['data_y'].append(data_value)


# df_best_model = pd.DataFrame(best_model_dict)
# df_best_model.to_csv('useful_csvs/best_models.csv')




############# Perform Cross validation and write each model to file ###########
# np.random.seed(1)

# splits = generate_splits(trials, 100)

# df_splits = pd.DataFrame({"train": [spl[0] for spl in splits], "test": [spl[1] for spl in splits]})
# df_splits.to_csv("crossv_splits.csv")

# start_time = time.time()
# full = save_models(splits, 0, unoise_range, not_affect_param_range, stationary_softener_range, speaker_optimality_range, beta_range, caused_version = ["and_hm_or_ws_cc"], aspect_version="modified")
# end_time = time.time()

# # print("Average Full Error:", np.mean(full_error))
# full.to_csv("split_performance_full.csv")
# print("Runtime:", end_time - start_time)
# print()

# start_time = time.time()
# lesion = save_models(splits, 1, unoise_range, not_affect_param_range, stationary_softener_range, speaker_optimality_range, beta_range, caused_version = ["and_hm_or_ws"], aspect_version="modified")
# end_time = time.time()

# lesion.to_csv("split_performance_lesion.csv")
# # print("Average Lesion Error:", np.mean(lesion_error))
# print("Runtime:", end_time - start_time)



############### Split Testing #################

# df_splits = pd.read_csv("crossv_splits.csv")

# train = df_splits.train[19][2:-1].split()
# test = df_splits.test[19][2:-1].split()
# train = np.array([int(tr) for tr in train])
# test = np.array([int(te) for te in test])


# out = grid_search(train, 0, unoise_range, not_affect_param_range, stationary_softener_range, speaker_optimality_range, None, caused_versions = ["and_hm_or_ws_cc"], enabled_versions = ["or_ws"], comparison_threshold_range=None, comparison_softener_range=None, aspect_version="modified", testing=False)

# # primary_aspect_values, alternative_aspect_values = load_aspects(out['unoise'], aspect_version="modified")
# primary_aspect_values, alternative_aspect_values = load_aspects(out['unoise'], aspect_version="modified")


# mean = meaning("caused", primary_aspect_values, "and_hm_or_ws_cc", "or_ws", out['not_affect_param'], out['stationary_softener'], comparison_threshold=None, comparison_softener=None, alternative_aspect_values = alternative_aspect_values)

# # print(m)

# model_output = s2(primary_aspect_values, "and_hm_or_ws_cc", "or_ws", out['not_affect_param'], out['stationary_softener'], out['speaker_optimality'], alternative_aspect_values=alternative_aspect_values)

# print(compute_error(model_output, train))


# del out['beta']
# del out['comparison_threshold']
# del out['comparison_softener']

# df_params = pd.DataFrame(out)


