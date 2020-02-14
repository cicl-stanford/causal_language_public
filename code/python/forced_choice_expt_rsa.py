"""
python enumerate_causal_rsa.py --n_samples 1000 --prior_type all
python enumerate_causal_rsa.py --n_samples 1000 --prior_type target_only
"""

import numpy as np
import json
import argparse
from sys import exit
import pandas as pd


trial_max = 32


vocabulary = ["caused", "enabled", "affected", "didn't affect"]

# """
# Lists of worlds, each world is an array:
# [w, h, s, m, o]
# w := whether-dependence
# h := how-dependence
# s := is A sufficient to get B thru gate?
# r := robustness
# o := outcome, 1 if B goes thru gate, 0 if it doesn't
# d := difference maker test
# p := anti-robustness
# """
w, h, s, m, o = (0, 1, 2, 3, 4)


# clips_data = json.load(open("aspects/experiment_trials_samples_1000__uncertainty_noise_{}__gate_alternative_0__box_alternative_{}_vector_representation.json".format(1.0, 0)))

# clips_aspect_values = np.array([tr['rep'] for tr in clips_data])

# caused_version = "and_hm_or_ws"
# stationary_softener = 0.5
# how_not_affect_param = 0.2
# speaker_optimality = 2


def meaning(utterance):
  if utterance == "enabled":
    # or whether, sufficiency
    return np.sum(clips_aspect_values[:,[w,s]], axis=1) - np.prod(clips_aspect_values[:,[w,s]], axis=1)

  elif utterance == "caused":
    # base caused definition or (whether, sufficiency) and how
    suff_or_necc = np.sum(clips_aspect_values[:,[w,s]], axis=1) - np.prod(clips_aspect_values[:,[w,s]], axis=1)
    caused_base = suff_or_necc * clips_aspect_values[:,h]

    # and moving feature
    if caused_version == "and_h_or_ws":
      return caused_base
    elif caused_version == "and_hm_or_ws":
      moving = clips_aspect_values[:,m]
      # soften the moving feature
      soften_moving = (moving + stationary_softener) - (moving * stationary_softener) 
      return caused_base * soften_moving
    else:
      raise Exception("meaning for caused version '{}' not implemented".format(caused_version))

  elif utterance == "affected":
    affected = clips_aspect_values[:,h]
    return affected

  elif utterance == "didn't affect":
    whether_or_suff = clips_aspect_values[:,w] + clips_aspect_values[:,s] - clips_aspect_values[:,w]*clips_aspect_values[:,s]
    how_and_flip = clips_aspect_values[:,h] * how_not_affect_param
    return 1 - (whether_or_suff + how_and_flip - whether_or_suff*how_and_flip)

  else:
    raise Exception("meaning for utterance '{}' not implemented.".format(utterance))

                      # L0(clip | utterance) \propto P(clip) * P(utterance true | clip)
def l0():
  n_worlds = clips_aspect_values.shape[0]
  assert n_worlds == trial_max
  prior = 1./n_worlds
  # number of utterances X number of worlds
  likelihood = np.array([meaning(utterance) for utterance in vocabulary])
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
def s1():
  prior = 1./len(vocabulary)
  likelihood = l0()
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
def l1():
  n_worlds = clips_aspect_values.shape[0]
  assert n_worlds == trial_max
  # prior over world
  prior = 1./n_worlds
  # 4 utterances X 32 worlds
  # for each world, there's a likelihood of the utterance
  likelihood = s1()
  numerator = prior*likelihood

  # normalize within each utterance to get a distribution over  worlds
  denominator = np.sum(numerator, axis=1)
  return numerator.T / denominator

# S2(utterance | clip) \propto P(utterance) * L1(clip | utterance)^optimality
def s2():
  prior = 1./len(vocabulary)
  likelihood = l1()
  unnormalized_s2 = prior*likelihood**speaker_optimality
  # normalize within each world to get distribution over utterances
  return unnormalized_s2.T / np.sum(unnormalized_s2, axis=1)



def softmax(arr, ax, beta=1):
  exp_arr = np.exp(beta*arr)
  return exp_arr/np.sum(exp_arr, axis=ax)


data = []

# for how_not_affect_param in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
for how_not_affect_param in [0.2]:

  # for stationary_softener in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
  for stationary_softener in [0.7]:

    for enabled_version in ["or_ws"]:

      for caused_version in ["and_h_or_ws", "and_hm_or_ws"]:

        for affected_version in ["how"]:

          for not_affect_version in ["not((how and param) or whether or sufficient)"]:

            # for uncertainty_noise in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]:
            for uncertainty_noise in [1.4, 1.5, 1.6]:

              for box_alternative in [0]:

                clips_data = json.load(open("aspects/experiment_trials_samples_1000__uncertainty_noise_{}__gate_alternative_0__box_alternative_{}_vector_representation.json".format(uncertainty_noise, box_alternative)))

                clips_aspect_values = np.array([tr['rep'] for tr in clips_data])


                for lesion_rsa in [0, 1]:

                  if lesion_rsa:
                    for beta in np.arange(1,15,0.5):
                      semantic_values = np.vstack([meaning(word) for word in vocabulary])
                      # beta fit for optimal value
                      # redo beta for new semantics
                      semantic_values = softmax(semantic_values, 0, beta=beta)
                      for u, utterance in enumerate(vocabulary):
                        for c in range(trial_max):
                          data.append({
                            "speaker_optimality": np.nan,
                            "caused_version": caused_version,
                            "affected_version": affected_version,
                            "enabled_version": enabled_version,
                            "utterance": utterance,
                            "trial": c,
                            "uncertainty_noise": uncertainty_noise,
                            "level": "s2",
                            "p": semantic_values[u,c],
                            "whether": clips_aspect_values[c,w],
                            "how": clips_aspect_values[c,h],
                            "sufficient": clips_aspect_values[c,s],
                            "moving": clips_aspect_values[c,m],
                            "not_affect_version": not_affect_version,
                            "box_alternative": box_alternative,
                            "lesion_rsa": lesion_rsa,
                            "stationary_softener": stationary_softener,
                            "how_not_affect_param": how_not_affect_param,
                            "beta": beta
                          })

                  # if not lesion_rsa:
                  else:
                    for speaker_optimality in [2]:
                          
                      s2_output = s2()

                      for u, utterance in enumerate(vocabulary):
                        for c in range(trial_max):
                          data.append({
                            "speaker_optimality": speaker_optimality,
                            "caused_version": caused_version,
                            "affected_version": affected_version,
                            "enabled_version": enabled_version,
                            "utterance": utterance,
                            "trial": c,
                            "uncertainty_noise": uncertainty_noise,
                            "level": "s2",
                            "p": s2_output[u, c],
                            "whether": clips_aspect_values[c,w],
                            "how": clips_aspect_values[c,h],
                            "sufficient": clips_aspect_values[c,s],
                            "moving": clips_aspect_values[c,m],
                            "not_affect_version": not_affect_version,
                            "box_alternative": box_alternative,
                            "lesion_rsa": lesion_rsa,
                            "stationary_softener": stationary_softener,
                            "how_not_affect_param": how_not_affect_param,
                            "beta": np.nan
                          })







pd.DataFrame(data).to_csv("forced_choice_expt_rsa.csv")