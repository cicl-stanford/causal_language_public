import sys
import pygame
from pygame.locals import *
import pymunk
import pymunk.pygame_util
import itertools
import json
import numpy as np
import math
from pymunk import Vec2d
import collections #for keeping the order in which dictionaries were created
import time
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import linregress
import os

# WARNING: Pygame and Pymunk have reverse labeling conventions along the Y axis.
# For pymunk the top is higher values and for pygame the top is lower values
# be aware when interpreting coordinates

class World():

	def __init__(self, gate=False, start_step=0):
		self.width = 800
		self.height = 600
		self.ball_size = 60
		self.box_size = (70,70)
		self.speed = 200 # scales how fast balls are moving 
		self.step_size = 1/50.0
		self.step_max = 500 # step at which to stop the animation
		self.step = start_step # used to record when events happen 
		self.space = pymunk.Space()
		self.events = {'collisions': [],
						'wall_bounces': [],
						'button_presses': [],
						'outcome': None,
						'outcome_fine': None} # used to record events 
		# containers for bodies and shapes
		self.bodies = collections.OrderedDict()
		self.shapes = collections.OrderedDict()	
		self.sprites = collections.OrderedDict()

		self.record_bodies = {'A', 'B', 'D', 'E', 'box', 'gate'}
		self.record_outcome = True
		self.cause_ball = 'A'
		self.target_ball = 'B'
		self.gate = gate

		self.collision_types = {
			'static': 0,
			'dynamic': 1,
			'gate': 2,
			'gate_sensor': 3,
			'button': 4,
			'button_sensor': 5,
			'button_stopper': 6
		}

		# add walls 
		self.add_wall(position = (400, 590), length = 800, height = 20, name = 'top_wall')
		self.add_wall(position = (400, 10), length = 800, height = 20, name = 'bottom_wall')

		# Add walls for presence or absence of a gate
		if not self.gate:
			self.add_wall(position = (10, 500), length = 20, height = 200, name = 'top_left_wall')
			self.add_wall(position = (10, 100), length = 20, height= 200, name = 'bottom_left_wall')
		else:
			# Top left wall setup
			self.add_wall(position = (10, 540.8425), length = 20, height = 78.334, name = 'top_top_left_wall')
			self.add_wall(position = (10, 432.5075), length = 20, height = 78.335, name = 'bottom_top_left_wall')


			# Bottom left wall setup
			self.add_wall(position=(10, 167.5025), length=20, height=78.335, name='top_bottom_left_wall')
			self.add_wall(position=(10, 59.1675), length=20, height=78.335, name='bottom_bottom_left_wall')


	###################### Simulation setup and running #######################
	def add_wall(self, position, length, height, name):
		body = pymunk.Body(body_type = pymunk.Body.STATIC)
		body.position = position
		body.name = name
		wall = pymunk.Poly.create_box(body, size = (length, height))
		wall.elasticity = 1
		# wall.name = name 
		wall.collision_type = self.collision_types['static']
		self.space.add(wall)
		return wall

	def animation_setup(self):
		# animation setup
		pygame.init()
		self.clock = pygame.time.Clock()
		self.screen = pygame.display.set_mode((self.width, self.height))
		pygame.display.set_caption("Animation")
		self.pic_count = 0 # used for saving images

		# Save the image for each body in a dictionary. Skip over sensors which are
		# invisible and do not have sprites
		for bname in self.bodies:
			if 'sensor' not in bname:
				sprite = pygame.image.load(os.path.join(self.img_dir, 'figures/' + bname + '.png'))
				self.sprites[bname] = sprite

	# Runs one step of animation
	def animation_step(self, save, save_dir="", save_frames=[]):
		# quit conditions
		for event in pygame.event.get():
			if event.type==QUIT:
				pygame.quit()
				sys.exit(0)
			elif event.type == KEYDOWN and event.key == K_ESCAPE:
					animate=False

		if len(save_frames)==0 or self.pic_count in save_frames:

			# draw background, walls, exit and bodies
			# background and exit come first so bodies will be superimposed on top
			self.screen.fill((255,255,255)) #background
			if not self.gate:
				pygame.draw.rect(self.screen, pygame.color.THECOLORS['red'], [0,200,20,200])
			else:
				pygame.draw.rect(self.screen, pygame.color.THECOLORS['red'], [0,206.67,20,187]) #goal

			for body in self.bodies:
				if 'sensor' not in body:
					self.update_sprite(body = self.bodies.get(body), sprite = self.sprites.get(body),screen = self.screen)

			# pygame.draw.rect(screen, pygame.color.THECOLORS['red'], [0,200,20,200]) #goal
			pygame.draw.rect(self.screen, pygame.color.THECOLORS['black'], [0,0,800,20]) #top wall
			pygame.draw.rect(self.screen, pygame.color.THECOLORS['black'], [0,580,800,20]) #bottom wall

			if not self.gate:
				pygame.draw.rect(self.screen, pygame.color.THECOLORS['black'], [0,0,20,200])
				pygame.draw.rect(self.screen, pygame.color.THECOLORS['black'], [0,400,20,200])
			else:
				# Top left drawing with button
				pygame.draw.rect(self.screen, pygame.color.THECOLORS['black'], [0,20,20,78.334]) #top left
				pygame.draw.rect(self.screen, pygame.color.THECOLORS['black'], [0,128.334,20,78.335])
				pygame.draw.rect(self.screen, pygame.color.THECOLORS['black'], [0,98.334,10,30])
				pygame.draw.rect(self.screen, pygame.color.THECOLORS['grey'], [5,98.334,5,30])
				pygame.draw.rect(self.screen, pygame.color.THECOLORS['grey'], [5,95.334,15,3])
				pygame.draw.rect(self.screen, pygame.color.THECOLORS['grey'], [5,128.334,15,3])

				# Bottom left drawing with button
				pygame.draw.rect(self.screen, pygame.color.THECOLORS['black'], [0,393.339,20,78.335])
				pygame.draw.rect(self.screen, pygame.color.THECOLORS['black'], [0,501.674,20,79])
				pygame.draw.rect(self.screen, pygame.color.THECOLORS['black'], [0,471.674,10,30])
				pygame.draw.rect(self.screen, pygame.color.THECOLORS['grey'], [5,471.674,5,30])
				pygame.draw.rect(self.screen, pygame.color.THECOLORS['grey'], [5,468.674,15,3])
				pygame.draw.rect(self.screen, pygame.color.THECOLORS['grey'], [5,501.674,15,3])

			# Update the screen
			pygame.display.flip()
			pygame.display.update()
			self.clock.tick(100)

			if save:
				if save_dir == "":
					save_dir = 'figures/frames'
				pygame.image.save(self.screen, os.path.join(save_dir, 'animation'+'{:03}'.format(self.pic_count)+'.png'))
		self.pic_count += 1

	# General function to run the simulation
	# Returns the events dictionary or events dictionary and the set of paths depending on value
	# of rec_paths
	def simulate(self, animate=False, noise=0, perturb=0, prob=0, save=False, rec_paths=False, info=[], img_dir="", save_dir="", save_frames=[], cut_sim=False):
		self.img_dir = img_dir

		self.collision_setup()
		done = False # pointer to say when animation is done
		if rec_paths:
			paths = {}
			for name, body in self.bodies.items():
				if name in self.record_bodies:
					paths[name] = {'position': [], 'velocity': []}

		# animation setup
		if animate:
			self.animation_setup()

		while not done:
			# animation code
			if animate:
				self.animation_step(save, save_dir, save_frames)

			# cf manipulations 
			# if len(info) != 0:
			for action in info:
				if action['action'] == 'remove':
					self.remove(obj=action['obj'], step=action['step'], animate=animate)
				if action['action'] == 'perturb':
					self.perturb(obj=action['obj'], step=action['step'], magnitude=action['magnitude'])
				if action['action'] == 'noise':
					self.apply_noise(obj=action['obj'], step=action['step'], noise=noise)
				if action['action'] == 'earthquake':
					self.earthquake(obj=action['obj'], prob=action['prob'], noise=action['noise'], earthquake_type=action['earthquake_type'])

			# If we want to record paths, save the position and velocity for each body
			# on the recording list
			if rec_paths:
				for name, body in self.bodies.items():
					if name in self.record_bodies:
						path = paths[name]
						positions = path['position']
						velocities = path['velocity']

						x,y = body.position
						x_vel,y_vel = body.velocity

						positions.append([x,y])
						velocities.append([x_vel,y_vel])

			# check completion
			done = self.end_clip(cut_sim)

			# Update the world itself
			self.space.step(self.step_size)
			self.step += 1

		# Double check collisions are in temporal order and return
		collisions = self.events['collisions']
		assert all([collisions[i]['step'] <= collisions[i+1]['step'] for i in range(len(collisions) - 1)])
		if not rec_paths:
			return self.events
		else:
			return self.events, paths

	# Initialize ball. Place it in the space. Note it in the shape and body dicts
	def add_ball(self, position, velocity, size, name):
		mass = 1
		radius = size/2
		moment = pymunk.moment_for_circle(mass, 0, radius)
		body = pymunk.Body(mass, moment)
		body.position = position
		body.size = (size,size)
		body.angle = 0
		velocity = [x*self.speed for x in velocity] 
		body.apply_impulse_at_local_point(velocity) #set velocity
		body.name = name 
		shape = pymunk.Circle(body, radius)
		shape.elasticity = 1.0
		shape.friction = 0
		shape.collision_type = self.collision_types['dynamic']
		self.space.add(body, shape)
		self.bodies[name] = body
		self.shapes[name] = shape
		return body, shape

	# Initialize box. Place in space. Record in shape and body dicts
	def add_box(self, position, size, angle, name):
		mass = 1
		moment = pymunk.moment_for_box(mass, size)
		body = pymunk.Body(mass, moment)
		body.position = position
		body.size = size
		body.angle = angle
		velocity = (0,0)

		# Create custom velocity update and override standard
		def update_velocity(body, gravity, damping, dt):
			pymunk.cp.cpBodyUpdateVelocity(body._body, tuple(gravity), .96, dt)
		body._set_velocity_func(update_velocity)


		body.name = name
		shape = pymunk.Poly.create_box(body, size)
		shape.elasticity = 1.0
		shape.collision_type = self.collision_types['dynamic']
		self.space.add(body, shape)
		self.bodies[name] = body
		self.shapes[name] = shape
		return body, shape

	# Initialize button
	def add_button(self, pos, gate_pos):
		body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
		# set starting position
		# Starting position for a button depends on where the gate starts
		# e.g. if the gate starts at top, the top button must already be pressed back
		if pos == 'top':
			if gate_pos == 'top':
				body.position = (10, 486.675)
			else:		
				body.position = (20, 486.675)
		if pos == 'bottom':
			if gate_pos == 'bottom':
				body.position = (10, 113.326)
			else:
				body.position = (20, 113.326)
		body.name = pos
		# Avoid overlap
		body.size = (20,29.9)
		body.angle = 0
		shape = pymunk.Poly.create_box(body, body.size)
		shape.elasticity = 1.0
		shape.collision_type = self.collision_types['button']
		# Add to space, record in body and shape dicts
		self.space.add(body, shape)
		self.bodies[body.name] = body
		self.shapes[body.name] = shape

		# For either sensor, put a wall behind to stop its backward movement
		# and an invisible sensor in front to stop its forward movement
		if pos == 'top':
			self.add_wall(position=(-10, 486.675), length=20, height=30, name='top_button_stopper')
			self.add_sensor((40, 486.675), 'top_front_sensor', 'button')
		elif pos == 'bottom':
			self.add_wall(position=(-10, 113.335), length=20, height=30, name='bottom_button_stopper')
			self.add_sensor((40,113.335), 'bottom_front_sensor', 'button')
		return body, shape

	# Initialize a sensor for stopping either a button or the gate
	def add_sensor(self, pos, name, stop_obj):
		body = pymunk.Body(body_type=pymunk.Body.STATIC)
		body.position = pos
		body.name = name
		body.size = (20, 20)
		body.angle = 0
		shape = pymunk.Poly.create_box(body=body, size=body.size)
		# Set its collision type based on the stop object
		if stop_obj == 'button':
			shape.collision_type = self.collision_types['button_sensor']
		elif stop_obj == 'gate':
			shape.collision_type = self.collision_types['gate_sensor']
		shape.sensor = True
		self.space.add(body, shape)
		self.bodies[body.name] = body
		self.shapes[body.name] = shape
		return body, shape

	# Add the gate
	def add_gate(self, start_pos):
		body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
		# Choose its starting position based on instruction
		if start_pos == 'middle':
			body.position = (30, 300)
		elif start_pos == 'top':
			body.position = (30, 486.67)
		elif start_pos == 'bottom':
			body.position = (30, 113.33)
		# Avoid tapping the wall and stopping the gate early
		size = (19.9, 186.67)
		body.size = size
		body.angle = 0
		body.name = 'gate'
		shape = pymunk.Poly.create_box(body, size)
		shape.elasticity = 1.0
		shape.collision_type = self.collision_types['gate']
		self.space.add(body, shape)
		self.bodies['gate'] = body
		self.shapes['gate'] = shape

		# Save the gate attribute for use by the overarching world object
		self.gate = body

		# If we add the gate, we need to add the activation buttons and also
		# stop sensors for when the gate returns to middle position from top or bottom
		self.add_button('top', start_pos)
		self.add_button('bottom', start_pos)
		self.add_sensor((30, 402), 'gate_top_sensor', 'gate')
		self.add_sensor((30, 197), 'gate_bottom_sensor', 'gate')
		return body, shape


	def flipy(self, y):
	    """Small hack to convert chipmunk physics to pygame coordinates"""
	    return -y+600

	# A procedure to re-draw the image of a dynamic body at each step
	# of animation
	# No return item
	def update_sprite(self,body,sprite,screen):
		p = body.position
		p = Vec2d(p.x, self.flipy(p.y))
		angle_degrees = math.degrees(body.angle)
		rotated_shape = pygame.transform.rotate(sprite, angle_degrees)
		offset = Vec2d(rotated_shape.get_size()) / 2.
		p = p - offset
		screen.blit(rotated_shape, p)

	# setup collision handlers 
	def collision_setup(self):	
		handler_dynamic = self.space.add_collision_handler(self.collision_types['dynamic'], self.collision_types['dynamic'])
		handler_dynamic.begin = self.collisions

		handler_ball_wall = self.space.add_collision_handler(self.collision_types['dynamic'], self.collision_types['static'])
		handler_ball_wall.begin = self.ball_other_col

		handler_ball_gate = self.space.add_collision_handler(self.collision_types['dynamic'], self.collision_types['gate'])
		handler_ball_gate.begin = self.ball_other_col

		handler_ball_button = self.space.add_collision_handler(self.collision_types['dynamic'], self.collision_types['button'])
		handler_ball_button.begin = self.ball_button_col

		handler_button_sensor = self.space.add_collision_handler(self.collision_types['button'], self.collision_types['button_sensor'])
		handler_button_sensor.begin = self.button_sensor_col
		# handler_button_sensor.post_solve = self.button_sensor_col

		handler_button_wall = self.space.add_collision_handler(self.collision_types['button'], self.collision_types['static'])
		handler_button_wall.pre_solve = self.button_wallstop_col

		handler_gate_wall = self.space.add_collision_handler(self.collision_types['gate'], self.collision_types['static'])
		handler_gate_wall.begin = self.gate_wall_col

		handler_gate_sensor = self.space.add_collision_handler(self.collision_types['gate'], self.collision_types['gate_sensor'])
		handler_gate_sensor.begin = self.gate_sensor_col

	# handle dynamic events. Specifically for two dynamic objects
	def collisions(self,arbiter,space,data):
		# print arbiter.is_first_contact #checks whether it was the first contact between the shapes 
		event = {
			'objects': {arbiter.shapes[0].body.name,arbiter.shapes[1].body.name},
			'step': self.step
		}
		self.events['collisions'].append(event)
		return True

	# handle dynamic events. For dynamic objects and a wall
	def ball_other_col(self,arbiter,space,data):
		# print arbiter.is_first_contact #checks whether it was the first contact between the shapes 
		event = {
			'objects': {arbiter.shapes[0].body.name,arbiter.shapes[1].body.name},
			'step': self.step
		}
		self.events['wall_bounces'].append(event)
		return True

	# handle ball (or box) button collisions
	def ball_button_col(self, arbiter, space, data):
		event = {
			'objects': {arbiter.shapes[0].body.name, arbiter.shapes[1].body.name},
			'step': self.step
		}
		self.events['button_presses'].append(event)

		# Move the button backward and move the gate up or down
		# Depending on which button was pressed
		button = arbiter.shapes[1].body
		button.velocity = (-20,0)
		if button.name == 'top':
			self.gate.velocity = (0,60)
		elif button.name == 'bottom':
			self.gate.velocity = (0,-60)
		else:
			raise Exception('Invalid button name: {}'.format(button.name))

		return True

	#  Stop button from moving when contacting sensor
	def button_sensor_col(self, arbiter, space, data):
		button = arbiter.shapes[0].body
		button.velocity = (0,0)
		return True

	# Stop button from moving when contacting wall
	def button_wallstop_col(self, arbiter, space, data):
		button = arbiter.shapes[0].body
		x,y = button.velocity
		# hack that lets the button return to pressable condition
		if x < 0:
			button.velocity = (0,0)
		return True

	# Stop the gate when it contacts a wall
	def gate_wall_col(self, arbiter, space, data):
		gate = arbiter.shapes[0].body
		gate.velocity = (0,0)
		return True

	# Stop the gate when it returns to middle and contacts a stop sensor
	def gate_sensor_col(self, arbiter, space, data):
		gate = arbiter.shapes[0].body
		gate.velocity = (0,0)
		# ran into a bug where the buttons started moving forward as soon as
		# the scene started because the gate first "contacted" the sensors
		# This hack prevents that from happening. Could be a more elegant way
		if self.step != 0:
			self.bodies['bottom'].velocity = (20,0)
			self.bodies['top'].velocity = (20,0)
		return True

	# A method to check whether to end the simulation and record outcome info
	# Would be good to refactor to think about how the world knows about the target
	# It's specified in the initialization but not fed into the constructor
	def end_clip(self, cut_sim=False):
		# If we are truncating the simulation cut it once the target has exited
		if cut_sim:
			# Should be able to assume in this condition that the target is in the scene
			# Since we cut based on when the target exits
			target = self.bodies[self.target_ball]

			# Only matters if the ball exits (for now, this may be incomplete)
			if target.position[0] < -self.ball_size/2:
				if self.record_outcome:
					self.events['outcome'] = 1
					self.events['outcome_fine'] = target.position
				pygame.display.quit()
				return True

		# Otherwise if we have passed the max step
		if self.step > self.step_max:
			if self.target_ball in self.bodies:
				# if we want to record the outcome
				if self.record_outcome:
					# Save the outcome event at coarse and fine level
					b = self.bodies[self.target_ball]
					if b.position[0] > -self.ball_size/2:
						self.events['outcome'] = 0
					else:
						self.events['outcome'] = 1
					self.events['outcome_fine'] = b.position
			# quit pygame
			pygame.display.quit()
			return True
		else:
			return False
	###################################################################

	############ Counterfactual manipulation procedures ###############
	# Remove a given object
	def remove(self,obj,step,animate):
		if self.step == step:
			self.space.remove(self.shapes[obj]) #remove body from space 
			self.space.remove(self.bodies[obj]) #remove body from space 
			del self.bodies[obj] #remove body 
			del self.shapes[obj] #remove shape
			if animate: 		
				del self.sprites[obj] #remove sprite 

	# Perturb a given object
	def perturb(self,obj,step,magnitude=0):
		if self.step == step:
			b = self.bodies[obj]
			b.position = (b.position.x+self.gaussian_noise()*magnitude,
				b.position.y+self.gaussian_noise()*magnitude)

	# step-wise noise
	# use when there's a removed collision (represent uncertainty about where the ball would have gone)
	# apply gaussian noise to velocity at each step
	# step := which step do we start applying noise (the removed collision point)
	# noise := standard deviation of the angle perturbation
	def apply_noise(self,obj,step,noise):
		if not noise == 0:
			b = self.bodies[obj]
			if self.step > step:
				x_vel = b.velocity[0]
				y_vel = b.velocity[1]
				perturb = self.gaussian_noise()*noise
				cos_noise = np.cos(self.deg_to_rad(perturb))
				sin_noise = np.sin(self.deg_to_rad(perturb))
				x_vel_noise = x_vel * cos_noise - y_vel * sin_noise
				y_vel_noise = x_vel * sin_noise + y_vel * cos_noise
				b.velocity = x_vel_noise,y_vel_noise

	# Box-Muller transform
	def gaussian_noise(self):
		u = 1 - np.random.random()
		v = 1 - np.random.random()
		return np.sqrt(-2*np.log(u)) * np.cos(2 * np.pi * v)

	def deg_to_rad(self, degrees): return degrees*(math.pi/180)

	##################################################################

################ Model 1 for pragmatic judgement #################


# A procedure to determine which balls are downstream of a given ball in a 
# chain of collisions, and the timestep at which those items enter the chain
# Returns a list of tuples indicating the items in the chain and the time steps
# at which they enter the chain
def collision_chain(collisions, in_chain, start_time, reverse_time=False):
	# Base case. When the list is empty return the empty list
	if collisions == []:
		return []
	else:
		# Otherwise grab the first collision. Diff is all balls in the collision
		# that are not currently in the causal chain
		hd = collisions[0]
		diff = hd['objects'] - in_chain

		# If both balls are not in the causal chain, then the collision is not connected
		# to the causal chain. Because the collisions are sorted by time, we know it can be ignored

		# If both balls are already in the causal chain, then they already should have noise applied
		# (or have been removed). We don't need to add noise.

		# If only one ball is not in the causal chain, then this collision makes that ball a part
		# of the causal chain. Check to make sure it did not occur simultaneously with the previous
		# collision (a counterexample). If it did not then we add that ball and the timestep of the
		# collision to the output. We also add the ball to the set of in_chain balls and update the
		# timestep

		if not reverse_time:
			time_condtion = hd['step'] > start_time
		else:
			# For the reverse situation we want to note simultaneous collisions so <=
			time_condtion = hd['step'] <= start_time

		if len(diff) == 1 and time_condtion:
			b2 = diff.pop()
			new_set = in_chain | {b2}
			return [(b2, hd['step'])] + collision_chain(collisions[1:], new_set, hd['step'], reverse_time=reverse_time)
		else:
			return collision_chain(collisions[1:], in_chain, start_time, reverse_time=reverse_time)
		
# Quick helper to extract the actual outcome of a trial
def outcome(trial):
	events = run_trial(trial=trial)
	outcome = events["outcome"]
	return outcome

# Run the whether cf test on a trial given a trial, candidate cause, target entitiy,
# uncertainty noise value, and number of samples
# Return a value indicating the proportion of samples in which the candidate satisfied the whether
# cause definition
def whether_test(trial, candidate, target, noise, num_samples, animate=False, test_noise=False):
	# Run the simulation naturally
	events = run_trial(trial=trial, animate=animate)

	# Determine the chain of collisions and the outcome
	col_actual = events['collisions']
	in_chain = {candidate}
	noise_steps = collision_chain(col_actual, in_chain, -1)
	outcome_actual = events['outcome']

	# Add removal manipulation and noise manipulations
	info = []
	info.append({
		'action': 'remove',
		'obj': candidate,
		'step': 0
		})

	for item in noise_steps:
		info.append({
			'action': 'noise',
			'obj': item[0],
			'step': item[1] 
		})

	# Run the counterfactual world and get the outcome for n samples
	outcome_cf = np.zeros(num_samples)
	for i in range(num_samples):	
		events_cf = run_trial(trial=trial, animate=animate, noise=noise, info=info)
		outcome_cf[i] = events_cf['outcome']


	# Return portion of difference making events. Return testing info if desired
	if not test_noise:
		return np.mean(outcome_actual != outcome_cf)
	else:
		return np.mean(outcome_actual != outcome_cf), {'info': info}


# Computes the how test on a given trial, candidate cause, target entity,
# for a perturb value and number of samples
# Return a value representing the portion of samples in which the candidate
# was a difference maker at fine granularity
def how_test(trial, candidate, target, perturb, num_samples, animate=False):
	events = run_trial(trial=trial, animate=animate)
	x,y = events['outcome_fine']
	outcome_actual = [x,y]

	# apply perturbation
	info = []
	info.append({
		'action': 'perturb',
		'obj': candidate,
		'step': 0,
		'magnitude': perturb
		})

	# run forward cf world and record outcome
	outcome_cf = np.zeros((num_samples,2))
	for i in range(num_samples):
		cf_events = run_trial(trial=trial, animate=animate, perturb=perturb, info=info)
		x,y = cf_events['outcome_fine']
		outcome_cf[i,:] = [x,y]

	# return portion of cases that are difference makers
	return np.mean(np.all(outcome_actual != outcome_cf, axis=1))


# Computes the sufficiency test on a given trial, candidate, and target
# for a noise value and number of samples
# Optional to include the box and gate as alternatives
# Optional also to test whether the cf events took place in the actual world (event_test)
# Returns a value indicating the proportion of samples for which the candidate is
# determined sufficient to make the target go through the gate
def sufficient_test(trial, candidate, target, noise, num_samples, gate_alt=False, box_alt=True, animate=False, test_noise=False, event_test=True):
	events = run_trial(trial=trial, animate=False)
	# Get the outcome
	outcome_actual = events['outcome']

	# Get the core and alternative objects for counterfactuals
	# For now at least, the candidate and target must be balls
	core = {candidate, target}
	alternatives = {b['name'] for b in trial['balls']}
	if box_alt:
		if 'boxes' in trial:
			alternatives.add('box')
	if gate_alt:
		if 'gate' in trial:
			alternatives.add('gate')
	alternatives = alternatives - core

	# some objects may need noise after you remove alternatives
	# Obtain all relevant collisions and sort by timestep
	col_actual = events['collisions']
	noise_steps = []
	for alt in alternatives:
		noise_steps = noise_steps + collision_chain(col_actual, {alt}, -1)
	noise_steps.sort(key=lambda x: x[1])

	# We can then filter noise additions to ensure that they only take place
	# for non alternatives and only the first time for each
	first_cause = False
	first_target = False
	filtered_noise = []
	for item in noise_steps:
		if item[0] == candidate and not first_cause:
			first_cause = True
			filtered_noise.append(item)
		if item[0] == target and not first_target:
			first_target = True
			filtered_noise.append(item)


	# Setup manipulations for counterfactual
	info_cf = []
	for obj in alternatives:
		manip = {
			'action': 'remove',
			'obj': obj,
			'step': 0
		}
		info_cf.append(manip)

	for item in filtered_noise:
		manip = {
			'action': 'noise',
			'obj': item[0],
			'step': item[1]
		}
		info_cf.append(manip)


	# setup manipulations for counterfactual contingency
	info_cf_cont = []
	# Collect all noise additions from the candidate and alternatives
	noise_add_cont = collision_chain(events['collisions'], {candidate}, -1) + filtered_noise
	# filter out anything noise application that isn't the target (since the target
	# is the only body left) and add the first if one exists
	noise_add_cont = [x for x in noise_add_cont if x[0] == target]
	# Multiple things could have hit the target in separate chains
	# Sort by timestep and choose the first one if it exists
	# Add it to manipulations for the counterfactual contingency
	noise_add_cont.sort(key=lambda x: x[1])
	if len(noise_add_cont) > 0:
		tar_noise = noise_add_cont[0]
		info_cf_cont.append({'action': 'noise', 'obj': tar_noise[0], 'step': tar_noise[1]})


	# add removals for counterfactual contingency (remove everything except target)
	for obj in alternatives:
		manip = {
			'action': 'remove',
			'obj': obj,
			'step': 0
		}
		info_cf_cont.append(manip)

	manip = {
		'action': 'remove',
		'obj': candidate,
		'step': 0
	}
	info_cf_cont.append(manip)

	# Run the counterfactuals and cf contingencies
	outcomes = np.zeros(num_samples)
	for i in range(num_samples):
		events_cf = run_trial(trial=trial, animate=animate, noise=noise, info=info_cf, cut_sim=True)
		outcome_cf = events_cf['outcome']

		events_cf_cont = run_trial(trial=trial, animate=animate, noise=noise, info=info_cf_cont, cut_sim=True)
		outcome_cf_cont = events_cf_cont['outcome']

		outcomes[i] = (outcome_actual == outcome_cf) and (outcome_cf != outcome_cf_cont)

		# check whether the sufficiency events took place in the actual world
		if event_test:

			actual_collisions = events['collisions']
			cf_collisions = events_cf['collisions']

			actual_bpresses = events['button_presses']
			cf_bpresses = events_cf['button_presses']

			# Checks whether each counterfactual collision and each counterfactual button press
			# took place in the actual world
			collision_containment = [col in actual_collisions for col in cf_collisions]
			bpress_containment = [press in actual_bpresses for press in cf_bpresses]

			containment = all(collision_containment) and all(bpress_containment)

			outcomes[i] = outcomes[i] and containment

	# Return the proportion of sufficient outcomes and noise info if desired
	if not test_noise:
		return np.mean(outcomes)
	else:
		return np.mean(outcomes), {'info_cf': info_cf, 'info_cf_cont': info_cf_cont}



# Procedure to test whether the candidate cause is stationary
def moving_test(trial, candidate, target):
	# Get the event and paths
	events, paths = run_trial(trial=trial, rec_paths=True)
	moving = 1
	# For any collision if the candidate and the target are in a collision
	# and the candidate's velocity is zero beforehand, then the candidate
	# cause was stationary
	for col in events["collisions"]:
		objects = col["objects"]
		if candidate in objects and target in objects:
			candidate_vel = paths[candidate]['velocity']
			if candidate_vel[col['step'] - 1] == [0.0, 0.0]:
				moving = 0

	return moving



# A procdedure to compute an aspect representation for a trial
def aspect_rep(trial, candidate, target, noise=2, perturb=0.01, num_samples=100, gate_alt=False, box_alt=True):
	o = outcome(trial)
	w = whether_test(trial, candidate, target, noise, num_samples)
	h = float(how_test(trial, candidate, target, perturb, 1))
	s = sufficient_test(trial, candidate, target, noise, num_samples, gate_alt=gate_alt, box_alt=box_alt)
	st = moving_test(trial, candidate, target)

	return [w, h, s, st, o]


# Given a trial dictionary, setup the world and simulate the trial
def run_trial(trial, animate=False, noise=0, perturb=0, prob=0, rec_paths=False, save=False, info=[], img_dir="", save_dir="", save_frames=[], cut_sim=False):
	balls = trial['balls']
	if 'boxes' in trial:
		boxes = trial['boxes']
	else:
		boxes = []

	w = World(gate=('gate' in trial))
	for ball in balls:
		w.add_ball(tuple(ball['position']), tuple(ball['velocity']), w.ball_size, ball['name'])
	for box in boxes:
		angle = box['angle'] if 'angle' in box else 0
		w.add_box(tuple(box['position']), w.box_size, w.deg_to_rad(angle), box['name'])
	if w.gate:
		gate = trial['gate']
		w.add_gate(gate['start_pos'])

	return w.simulate(animate=animate, noise=noise, perturb=perturb, prob=prob, save=save, rec_paths=rec_paths, info=info, img_dir=img_dir, save_dir=save_dir, save_frames=save_frames, cut_sim=cut_sim)


def viewers(trials, prior):

	def cf(x):
		t = trials[x].copy()
		t["balls"] = t["balls"][1:]
		print(json.dumps(t, indent=2))
		run_trial(t, animate=True)

	def suff(x):
		t = trials[x].copy()
		t["boxes"] = []
		print(json.dumps(t, indent=2))
		run_trial(t, animate=True)

	def anim(x):
		t = trials[x]
		print(json.dumps(t, indent=2))
		print(prior[x])
		run_trial(t, animate=True)

	return cf, suff, anim

def load_trials(path):
	with open(path) as f:
		data_str = f.read()

	return json.loads(data_str)
