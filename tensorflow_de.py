#!/bin/python
# encoding: utf-8

import tensorflow as tf
print("TensorFlow version:", tf.__version__)

import tensorflow.contrib.eager as tfe
import numpy as np
import time

tfe.enable_eager_execution()

print("Available GPUs:", tfe.num_gpus())

assert tfe.num_gpus() > 0, "You need a GPU with the CUDA etc to run this!"


def run_differential_evolution(solution_size=5,
                               population_size=1000,
                               iteration_count=100,
                               print_results=False,
                               differential_weight=1.0,
                               crossover_probability=0.5,
                               lower=0.0,
                               upper=1.0):
    # pad population for mutation and cross evolution use
    while population_size % 3 != 0:
        population_size += 1

    # generate initial target solution matrix, all zero
    target_solution = np.zeros((solution_size))

    mean = np.mean([lower, upper])
    std = np.std([lower, upper])
    # generate initial population in random
    population = tf.random_normal(shape=[population_size, solution_size], mean=mean, stddev=std)

    for i in range(iteration_count):
        random_1 = np.arange(0, population_size)
        np.random.shuffle(random_1)
        random_2 = np.arange(0, population_size)
        np.random.shuffle(random_2)
        random_3 = np.arange(0, population_size)
        np.random.shuffle(random_3)
        random_trio_1 = tf.reshape(tf.gather(population, random_1), shape=[-1, 3, solution_size])  # set column to 3 to perform mutation
        random_trio_2 = tf.reshape(tf.gather(population, random_2), shape=[-1, 3, solution_size])
        random_trio_3 = tf.reshape(tf.gather(population, random_3), shape=[-1, 3, solution_size])

        mutation_trios = tf.concat([random_trio_1, random_trio_2, random_trio_3], axis=0)
        # vertically split generated random set to 3 parts
        vectors_1, vectors_2, vectors_3 = tf.unstack(mutation_trios, axis=1, num=3)
        # perform mutation
        doners = vectors_1 + differential_weight * (vectors_2 - vectors_3)

        # generate crossover probability for each element
        crossover_probabilities = tf.random_uniform(minval=0, maxval=1, shape=[population_size, solution_size])

        # perform crossover, crossover_probability scale of present population are replaced by crossover elements
        trial_population = tf.where(crossover_probabilities < crossover_probability, x=doners, y=population)
        # compute standard deviation of crossover population and target, use as crossover evaluation standard
        trial_fitnesses = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(trial_population, target_solution)), axis=1))
        # compute standard deviation of present population and target, use as population evaluation standard
        og_fitnesses = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(population, target_solution)), axis=1))
        if print_results: print(tf.gather(og_fitnesses, tf.argmin(og_fitnesses)))  # compute overall minimum

        population = tf.where(trial_fitnesses < og_fitnesses, x=trial_population, y=population)
        print("trial_population at iter {}: {}".format(i, trial_population))

    print(population)


tf.reset_default_graph()
with tf.device("/gpu:0"):
    run_differential_evolution(iteration_count=50,
                               solution_size=5,
                               crossover_probability=0.1,
                               print_results=True)

string = "took {}s to run population size of {} for {} iterations"
population_size = 300
iteration_count = 1000
solution_size = 64 * 64

# tf.reset_default_graph()
# with tf.device("/cpu:0"):
#     start = time.time()
#     run_differential_evolution(solution_size=solution_size,
#                                population_size=population_size,
#                                iteration_count=iteration_count,
#                                print_results=False)
#     end = time.time()
#     time_taken = end - start
#     print("CPU: " + string.format(time_taken, population_size, iteration_count))
#
# tf.reset_default_graph()
# with tf.device("/gpu:0"):
#     start = time.time()
#     run_differential_evolution(solution_size=solution_size,
#                                population_size=population_size,
#                                iteration_count=iteration_count,
#                                print_results=False)
#     end = time.time()
#     time_taken = end - start
#     print("GPU: " + string.format(time_taken, population_size, iteration_count))
