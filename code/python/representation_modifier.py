import json


with open("aspects/clang3_exp_extended_trials_samples_1000__uncertainty_noise_0.5__seed_123__whether_test_version_basic__gate_alternative_0__box_alternative_0_vector_representation.json") as f:
	aspects = json.load(f)


a_reps = [reps for reps in aspects if reps[0]['candidate'] == 'A'][0]

modified_reps = []

for rep in a_reps:
	w, h, s = rep[:3]