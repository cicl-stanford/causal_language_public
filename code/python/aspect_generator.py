import json
import model as m
import numpy as np
import argparse
import time
import os

"""Get aspect representation for world ([w, h, s, m, o]) given a set of initial conditions.
	w := dependence of *whether* E goes thru the gate on *whether* A exists [0,1]
	h := fine-grained dependence of ball E's final state on ball A's initial state {0,1}
	s := sufficiency: in the absence of other objects, what would `w` be?
	m := moving: whether the candidate was moving when it collided with the target
	o := outcome truth value: in the actual world, did E go thru the gate?
"""

# Weird meta parameter. Come back to fix this
# include_robustness = False

parser = argparse.ArgumentParser()
parser.add_argument('--i', type=str, help='input json file of objects and initial conditions')
parser.add_argument('--samples', type=int, default=100, help='number of cf samples to take')
parser.add_argument('--uncertainty_noise', type=float, default=1.6, help='standard deviation of the angle perturbation when adding step-wise noise due to a removed collision')
parser.add_argument("--gate_alternative", type=int, default=0, help='whether to include the gate as an alternative for the sufficiency test')
parser.add_argument("--box_alternative", type=int, default=0, help="whether to include the box as an alternative for the sufficiency test")
args = parser.parse_args()

argstring = "__".join(["{}_{}".format(p, a) for p, a in vars(args).items() if p!="i"])
print(argstring)

np.random.seed(123)


def generate_aspects(trials_file, representation_file):
	trials = m.load_trials(trials_file)

	aspect_reps = []

	for i in range(len(trials)):
		print('Clip', i+1)
		tr = trials[i]
		o = m.outcome(tr)

		rep = m.aspect_rep(tr, "A", "B", noise=args.uncertainty_noise, perturb=1, num_samples=args.samples, gate_alt=args.gate_alternative, box_alt=args.box_alternative)

		aspect_reps.append({"trial": i, "rep": rep})

	print()

	with open(representation_file, 'w+') as f:
		json.dump(aspect_reps, f)

start = time.time()
input_file = args.i
output_file = "aspects/" + os.path.basename(input_file)[:-5] + "_" + argstring + "_vector_representation.json"
generate_aspects(input_file, output_file)
print('Runtime:', time.time() - start)