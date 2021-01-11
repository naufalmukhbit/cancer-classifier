import tensorflow as tf

from neupy.utils import dot, function_name_scope, asfloat, make_single_vector
from neupy.core.properties import (ChoiceProperty, NumberProperty,
                                   WithdrawProperty)
from neupy.utils.tf_utils import setup_parameter_updates
from neupy.algorithms import BaseOptimizer
from neupy.algorithms.gd.quasi_newton import safe_division, WolfeLineSearchForStep


__all__ = ('ConjugateGradient_Custom',)


@function_name_scope
def fletcher_reeves(old_g, new_g, delta_w, epsilon=1e-7):
    return safe_division(
        dot(new_g, new_g),
        dot(old_g, old_g),
        epsilon,
    )


@function_name_scope
def polak_ribiere(old_g, new_g, delta_w, epsilon=1e-7):
    return safe_division(
        dot(new_g, new_g - old_g),
        dot(old_g, old_g),
        epsilon,
    )


@function_name_scope
def powell_beale(old_g, new_g, delta_w, epsilon=1e-7):
    return safe_division(
        dot(new_g, new_g - old_g),
        dot(delta_w, new_g - old_g),
        epsilon,
    )


class ConjugateGradient_Custom(WolfeLineSearchForStep, BaseOptimizer):
    epsilon = NumberProperty(default=1e-7, minval=0)
    update_function = ChoiceProperty(
        default='fletcher_reeves',
        choices={
            'fletcher_reeves': fletcher_reeves,
            'polak_ribiere': polak_ribiere,
            'powell_beale': powell_beale,
        }
    )
    step = WithdrawProperty()

    def init_functions(self):
        n_parameters = self.network.n_parameters
        self.variables.update(
            prev_delta=tf.Variable(
                tf.zeros([n_parameters]),
                name="conj-grad/prev-delta",
                dtype=tf.float32,
            ),
            prev_gradient=tf.Variable(
                tf.zeros([n_parameters]),
                name="conj-grad/prev-gradient",
                dtype=tf.float32,
            ),
            iteration=tf.Variable(
                asfloat(self.last_epoch),
                name='conj-grad/current-iteration',
                dtype=tf.float32
            ),
        )
        super(ConjugateGradient_Custom, self).init_functions()

    def init_train_updates(self):
        iteration = self.variables.iteration
        previous_delta = self.variables.prev_delta
        previous_gradient = self.variables.prev_gradient

        n_parameters = self.network.n_parameters
        variables = self.network.variables
        parameters = [var for var in variables.values() if var.trainable]
        param_vector = make_single_vector(parameters)

        gradients = tf.gradients(self.variables.loss, parameters)
        full_gradient = make_single_vector(gradients)

        beta = self.update_function(
            previous_gradient, full_gradient, previous_delta, self.epsilon)

        parameter_delta = tf.where(
            tf.equal(tf.mod(iteration, n_parameters), 0),
            -full_gradient,
            -full_gradient + beta * previous_delta
        )

        step = self.find_optimal_step(param_vector, parameter_delta)
        updated_parameters = param_vector + step * parameter_delta
        updates = setup_parameter_updates(parameters, updated_parameters)

        with tf.control_dependencies([full_gradient, parameter_delta]):
            updates.extend([
                previous_gradient.assign(full_gradient),
                previous_delta.assign(parameter_delta),
                iteration.assign(iteration + 1),
            ])

        return updates