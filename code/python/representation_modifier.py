import json


def modify_rep(noise):

	with open("aspects/clang3_exp_extended_trials_samples_1000__uncertainty_noise_{}__seed_123__whether_test_version_basic__gate_alternative_0__box_alternative_0_vector_representation.json".format(noise)) as f:
		aspects = json.load(f)


	a_reps = [reps for reps in aspects if reps[0]['candidate'] == 'A'][0]

	modified_reps = []

	for tr in a_reps:
		rep = tr['rep']
		w, h, s = rep[:3]
		m = 1 - int(rep[6])
		o = rep[4]

		new_rep = {'trial': tr['trial'], 'rep': [w,h,s,m,o]}
		modified_reps.append(new_rep)


	jstr = json.dumps(modified_reps)

	with open("aspects/experiment_trials_samples_1000__uncertainty_noise_{}__gate_alternative_0__box_alternative_0_vector_representation.json".format(noise), "w+") as f:
		f.write(jstr)


for noise_val in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]:
	modify_rep(noise_val)