import model as m
import json
from sys import argv
import time


# Generate Aspect file for a given noise value and number of samples
# Computes aspect values for A, the blue ball, and the box
def make_prior(trials_file, prior_file, noise_val, num_samples=100, include_box=True):
	trials = m.load_trials(trials_file)

	alternative_list = m.load_alternative_assessments()

	prior = []

	candidates = ['A', 'D']
	if include_box:
		candidates.append('box')

	for cand in candidates:
		print('Candidate', cand)
		worlds = []
		for i in range(len(trials)):
			print('Clip', i)
			tr = trials[i]

			alternatives = alternative_list[i] - {cand}

			rep = m.aspect_rep(tr, cand, 'B', alternatives, noise_val, perturb=0.01, num_samples=num_samples)

			worlds.append({"candidate": cand, "trial": i, "rep": rep})

		prior.append(worlds)

		print()


	with open(prior_file, 'w+') as f:
		json.dump(prior, f)



noise_val = argv[1]
num_samples = argv[2]

start = time.time()
make_prior(trials_file = 'trialinfo/experiment_trials.json',
 prior_file = 'aspects/aspects_noise_' + str(noise_val) + '_samples_' + str(num_samples) + '.json',
 noise_val = float(noise_val),
 num_samples = int(num_samples))


print('Runtime:', time.time() - start)