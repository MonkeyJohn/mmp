"""

Define a Value network Q(s,a,theta), where
    s is the state,
    a is the action,
    and theta is the network parameter.

Here we focus on the problems with finite action space.

"""

import numpy as np
import tensorflow as tf
from keras.models import sequential
from keras.layers import Flatten, Dense, Conv2D

def mlp( hiddens, num_actions):
    """This model takes as input an observation and returns values of all actions. The network used here is a fully-connected neural network.

    Parameters
    ----------
    hiddens: [int]
        list of sizes of hidden layers
    num_actions: int
        size of the action space

    Returns
    -------
    model: function
        q_function for DQN algorithm.
    """

    model = Sequential()

    for num_outputs, kernel_size, stride in convs:
        model.add(Conv2D(num_outputs, kernel_size, (stride, stride), activation = 'relu') ) # convolution layer
    model.add(Flatten())

    for hidden in hiddens:
        model.add(Dense(hideen, activation='relu')) # fully connected layer

    model.add(Dense(num_actions)) # the output has dimension (num_actions,)

    return  model



def cnn_to_mlp(convs, hiddens, num_actions):
    """This model takes as input an observation and returns values of all actions. The network used here is a convolutional neural network.

    Parameters
    ----------
    convs: [(int, int int)]
        list of convolutional layers in form of
        (num_outputs, kernel_size, stride)
    hiddens: [int]
        list of sizes of hidden layers
    num_actions: int
        size of the action space

    Returns
    -------
    model: function
        q_function for DQN algorithm.
    """

    model = Sequential()

    for num_outputs, kernel_size, stride in convs:
        model.add(Conv2D(num_outputs, kernel_size, (stride, stride), activation = 'relu') ) # convolution layer
    model.add(Flatten())

    for hidden in hiddens:
        model.add(Dense(hideen, activation='relu')) # fully connected layer

    model.add(Dense(num_actions)) # the output has dimension (num_actions,)

    return  model

# Image_dim1, Image_dim2, num_frames = input_size
# s = tf.placeholder(tf.float32, [None, STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT])


# class DQNAgent(AbstractDQNAgent):
#     """Write me
#     """
#     def __init__(self, model, policy=None, test_policy=None, enable_double_dqn=True, enable_dueling_network=False,
#                  dueling_type='avg', *args, **kwargs):
#         super(DQNAgent, self).__init__(*args, **kwargs)
#
#         # Validate (important) input.
#         if hasattr(model.output, '__len__') and len(model.output) > 1:
#             raise ValueError('Model "{}" has more than one output. DQN expects a model that has a single output.'.format(model))
#         if model.output._keras_shape != (None, self.nb_actions):
#             raise ValueError('Model output "{}" has invalid shape. DQN expects a model that has one dimension for each action, in this case {}.'.format(model.output, self.nb_actions))
#
#         # Parameters.
#         self.enable_double_dqn = enable_double_dqn
#         self.enable_dueling_network = enable_dueling_network
#         self.dueling_type = dueling_type
#         if self.enable_dueling_network:
#             # get the second last layer of the model, abandon the last layer
#             layer = model.layers[-2]
#             nb_action = model.output._keras_shape[-1]
#             # layer y has a shape (nb_action+1,)
#             # y[:,0] represents V(s;theta)
#             # y[:,1:] represents A(s,a;theta)
#             y = Dense(nb_action + 1, activation='linear')(layer.output)
#             # caculate the Q(s,a;theta)
#             # dueling_type == 'avg'
#             # Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-Avg_a(A(s,a;theta)))
#             # dueling_type == 'max'
#             # Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-max_a(A(s,a;theta)))
#             # dueling_type == 'naive'
#             # Q(s,a;theta) = V(s;theta) + A(s,a;theta)
#             if self.dueling_type == 'avg':
#                 outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(a[:, 1:], keepdims=True), output_shape=(nb_action,))(y)
#             elif self.dueling_type == 'max':
#                 outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.max(a[:, 1:], keepdims=True), output_shape=(nb_action,))(y)
#             elif self.dueling_type == 'naive':
#                 outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:], output_shape=(nb_action,))(y)
#             else:
#                 assert False, "dueling_type must be one of {'avg','max','naive'}"
#
#             model = Model(input=model.input, output=outputlayer)
#
#         # Related objects.
#         self.model = model
#         if policy is None:
#             policy = EpsGreedyQPolicy()
#         if test_policy is None:
#             test_policy = GreedyQPolicy()
#         self.policy = policy
#         self.test_policy = test_policy
#
#         # State.
#         self.reset_states()
#
#     def get_config(self):
#         config = super(DQNAgent, self).get_config()
#         config['enable_double_dqn'] = self.enable_double_dqn
#         config['dueling_type'] = self.dueling_type
#         config['enable_dueling_network'] = self.enable_dueling_network
#         config['model'] = get_object_config(self.model)
#         config['policy'] = get_object_config(self.policy)
#         config['test_policy'] = get_object_config(self.test_policy)
#         if self.compiled:
#             config['target_model'] = get_object_config(self.target_model)
#         return config
