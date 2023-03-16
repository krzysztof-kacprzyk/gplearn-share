"""The underlying data structure used in gplearn.

The :mod:`gplearn._program` module contains the underlying representation of a
computer program. It is used for creating and evolving programs used in the
:mod:`gplearn.genetic` module.
"""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

from copy import copy
import os

import numpy as np
from sklearn.utils.random import sample_without_replacement
from sklearn.metrics import auc, r2_score, roc_auc_score

from .functions import _Function, _sigmoid
from .utils import check_random_state
from .functions import _function_map

from functools import reduce

from .model import Model, LitModel

import pytorch_lightning as pl


import torch

import time

from datetime import datetime

import matplotlib.pyplot as plt

import pandas as pd

class _Program(object):

    """A program-like representation of the evolved program.

    This is the underlying data-structure used by the public classes in the
    :mod:`gplearn.genetic` module. It should not be used directly by the user.

    Parameters
    ----------
    function_set : list
        A list of valid functions to use in the program.

    arities : dict
        A dictionary of the form `{arity: [functions]}`. The arity is the
        number of arguments that the function takes, the functions must match
        those in the `function_set` parameter.

    init_depth : tuple of two ints
        The range of tree depths for the initial population of naive formulas.
        Individual trees will randomly choose a maximum depth from this range.
        When combined with `init_method='half and half'` this yields the well-
        known 'ramped half and half' initialization method.

    init_method : str
        - 'grow' : Nodes are chosen at random from both functions and
          terminals, allowing for smaller trees than `init_depth` allows. Tends
          to grow asymmetrical trees.
        - 'full' : Functions are chosen until the `init_depth` is reached, and
          then terminals are selected. Tends to grow 'bushy' trees.
        - 'half and half' : Trees are grown through a 50/50 mix of 'full' and
          'grow', making for a mix of tree shapes in the initial population.

    n_features : int
        The number of features in `X`.

    const_range : tuple of two floats
        The range of constants to include in the formulas.

    metric : _Fitness object
        The raw fitness metric.

    p_point_replace : float
        The probability that any given node will be mutated during point
        mutation.

    parsimony_coefficient : float
        This constant penalizes large programs by adjusting their fitness to
        be less favorable for selection. Larger values penalize the program
        more which can control the phenomenon known as 'bloat'. Bloat is when
        evolution is increasing the size of programs without a significant
        increase in fitness, which is costly for computation time and makes for
        a less understandable final result. This parameter may need to be tuned
        over successive runs.

    random_state : RandomState instance
        The random number generator. Note that ints, or None are not allowed.
        The reason for this being passed is that during parallel evolution the
        same program object may be accessed by multiple parallel processes.

    transformer : _Function object, optional (default=None)
        The function to transform the output of the program to probabilities,
        only used for the SymbolicClassifier.

    feature_names : list, optional (default=None)
        Optional list of feature names, used purely for representations in
        the `print` operation or `export_graphviz`. If None, then X0, X1, etc
        will be used for representations.

    program : list, optional (default=None)
        The flattened tree representation of the program. If None, a new naive
        random tree will be grown. If provided, it will be validated.
    
    function_probs : list, optional (default=None)
        The probabilities of choosing functions during program generation.
        The numbers in the list should sum up to 1.

    Attributes
    ----------
    program : list
        The flattened tree representation of the program.

    raw_fitness_ : float
        The raw fitness of the individual program.

    fitness_ : float
        The penalized fitness of the individual program.

    oob_fitness_ : float
        The out-of-bag raw fitness of the individual program for the held-out
        samples. Only present when sub-sampling was used in the estimator by
        specifying `max_samples` < 1.0.

    parents : dict, or None
        If None, this is a naive random program from the initial population.
        Otherwise it includes meta-data about the program's parent(s) as well
        as the genetic operations performed to yield the current program. This
        is set outside this class by the controlling evolution loops.

    depth_ : int
        The maximum depth of the program tree.

    length_ : int
        The number of functions and terminals in the program.

    """

    def __init__(self,
                 function_set,
                 arities,
                 init_depth,
                 init_method,
                 n_features,
                 const_range,
                 metric,
                 p_point_replace,
                 parsimony_coefficient,
                 random_state,
                 transformer=None,
                 feature_names=None,
                 program=None,
                 function_probs=None,
                 optim_dict=None,
                 timestamp="unknown"):

        self.function_set = function_set
        self.arities = arities
        self.init_depth = (init_depth[0], init_depth[1] + 1)
        self.init_method = init_method
        self.n_features = n_features
        self.const_range = const_range
        self.metric = metric
        self.p_point_replace = p_point_replace
        self.parsimony_coefficient = parsimony_coefficient
        self.transformer = transformer
        self.feature_names = feature_names
        self.program = program
        self.function_probs = function_probs
        self.model = None
        self.optim_dict = optim_dict
        self.timestamp = timestamp

        if self.function_probs is None:
            # Uniform distribution over all functions
            self.function_probs = np.ones(len(self.function_set)) / len(self.function_set)

        operator_indices = [i for i, fun in enumerate(self.function_set) if fun.name != 'shape']
        self.operator_set = [self.function_set[i] for i in operator_indices]
        self.operator_probs = self.function_probs[operator_indices]
        self.operator_probs /= np.sum(self.operator_probs)

        if self.program is not None:
            if not self.validate_program():
                raise ValueError('The supplied program is incomplete.')
        else:
            # Create a naive random program
            n_active_variables = self.random_skewed_integer(random_state,1,self.n_features+1)
            active_variables = set(random_state.choice(list(range(self.n_features)),size=n_active_variables,replace=False))
            self.program = self.build_program(random_state,variables=active_variables)
        
        if not self.validate_unique_leaves():
            print(self.program)
            raise ValueError('The supplied program does not have unique leaves')

        self.active_variables = self.find_active_variables()
        
        self.raw_fitness_ = None
        self.fitness_ = None
        self.parents = None
        self._n_samples = None
        self._max_samples = None
        self._indices_state = None

    def random_function(self, random_state, only_operators=False):
        if not only_operators:
            return random_state.choice(self.function_set,replace=True,p=self.function_probs)
        else:
            return random_state.choice(self.operator_set,replace=True,p=self.operator_probs)
    
    def is_shape_function(self, function):
        if isinstance(function, _Function):
            if function.name == 'shape':
                return True
            else:
                return False
        else:
            return False

    def find_active_variables(self, program=None):
        if program is None:
            program = self.program
        active_variables = set()
        for node in program:
            if isinstance(node, int):
                active_variables.add(node)
        
        return active_variables

    def validate_unique_leaves(self):
        active_variables = set()
        for node in self.program:
            if isinstance(node, int):
                if node in active_variables:
                    return False
                active_variables.add(node)
        
        return True

    def is_fitting_necessary(self, categorical_variables):
        return self.any_shapes() or self.any_categorical_variables(categorical_variables)
    
    def random_skewed_integer(self, random_state, low, high):
        # low - inclusive, high - exclusive
        ps = np.array(list(range(1,high-low+1)))
        ps = ps / ps.sum()
        return random_state.choice(list(range(low,high)),p=ps)



    def build_program(self, random_state, variables=None):
        """Build a naive random program.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.
        
        variables: set
            Set of integers corresponding to available variables.
            If None then all variables are considered

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        if variables is None:
            variables = set([int(i) for i in range(self.n_features)])

        if self.init_method == 'half and half':
            method = ('full' if random_state.randint(2) else 'grow')
        else:
            method = self.init_method

        # max_depth = random_state.randint(*self.init_depth)

        # Start a program with a function to avoid degenerative programs
        if len(variables) > 1:
            function = self.random_function(random_state)
        else:
            function = _function_map['shape']
        program = [function]
        terminal_stack = [function.arity]

        n_planned_variables = function.arity

        while terminal_stack:
            possibilities = []
            if not self.is_shape_function(program[-1]):
                # You can choose a shape function as the previous element was not a shape function
                possibilities.append('shape')
              
            if n_planned_variables < len(variables):
                # You can choose a binary operator as there are still some variables unaccounted for
                possibilities.append('operator')
            
            if (n_planned_variables > 1) or (len(variables) == 1):
                # You can choose a variable as this would not prevent other variables from being chosen in the future
                possibilities.append('variable')
            
            if method == 'full': # If the method is 'full' then we choose functions as long as we can

                # If the variable is the only option then choose variable
                if (len(possibilities) == 1) and ('variable' in possibilities):
                    node = int(random_state.choice(list(variables)))
                else: # If a function is possible then we choose a function
                    if ('shape' in possibilities) and ('operator' in possibilities):
                        node = self.random_function(random_state)
                    elif 'operator' in possibilities:
                        node = self.random_function(random_state, only_operators=True)
                    elif 'shape' in possibilities:
                        node = _function_map['shape']
                    else:
                        print(possibilities)
                        raise ValueError("That is weird. We should never get here")

            elif method == 'grow': # If the method is 'grow' then we can choose a leaf earlier
                node_name = random_state.choice(possibilities) # This is uniform, may be changed later TODO:
                if node_name in ['shape','operator']:
                    if 'operator' in possibilities:
                        node = self.random_function(random_state, only_operators=(not ('shape' in possibilities)))
                    elif 'shape' in possibilities:
                        node = _function_map['shape']
                    else:
                        print(possibilities)
                        print(node_name)
                        print(program)
                        raise ValueError("That is weird. We should never get here")
                else:
                    node = int(random_state.choice(list(variables)))

            if isinstance(node, _Function):
                # The next node is a function
                program.append(node)
                terminal_stack.append(node.arity)
                n_planned_variables += (node.arity - 1)
            else: # it is a variable as we exclude numeric constants
                node = int(node)
                program.append(node)
                variables.remove(node)
                n_planned_variables -= 1 # as the plan was executed

                terminal_stack[-1] -= 1
                while terminal_stack[-1] == 0:
                    terminal_stack.pop()
                    if not terminal_stack:
                        return program
                    terminal_stack[-1] -= 1
        
        # We should never get here
        raise ValueError("That is weird. We should never get here")

        # while terminal_stack:
        #     depth = len(terminal_stack)
        #     choice = self.n_features + len(self.function_set)
        #     choice = random_state.randint(choice)
        #     # Determine if we are adding a function or terminal
        #     if (depth < max_depth) and (method == 'full' or
        #                                 choice <= len(self.function_set)):
        #         function = random_state.randint(len(self.function_set))
        #         function = self.function_set[function]
        #         program.append(function)
        #         terminal_stack.append(function.arity)
        #     else:
        #         # We need a terminal, add a variable or constant
        #         if self.const_range is not None:
        #             terminal = random_state.randint(self.n_features + 1)
        #         else:
        #             terminal = random_state.randint(self.n_features)
        #         if terminal == self.n_features:
        #             terminal = random_state.uniform(*self.const_range)
        #             if self.const_range is None:
        #                 # We should never get here
        #                 raise ValueError('A constant was produced with '
        #                                  'const_range=None.')
        #         program.append(terminal)
        #         terminal_stack[-1] -= 1
        #         while terminal_stack[-1] == 0:
        #             terminal_stack.pop()
        #             if not terminal_stack:
        #                 return program
        #             terminal_stack[-1] -= 1

        # # We should never get here
        # return None

    def validate_program(self):
        """Rough check that the embedded program in the object is valid."""
        terminals = [0]
        for node in self.program:
            if isinstance(node, _Function):
                terminals.append(node.arity)
            else:
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
        return terminals == [-1]

    def __str__(self):
        """Overloads `print` output of the object to resemble a LISP tree."""
        terminals = [0]
        output = ''
        for i, node in enumerate(self.program):
            if isinstance(node, _Function):
                terminals.append(node.arity)
                output += node.name + '('
            else:
                if isinstance(node, int):
                    if self.feature_names is None:
                        output += 'X%s' % node
                    else:
                        output += self.feature_names[node]
                else:
                    output += '%.3f' % node
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
                    output += ')'
                if i != len(self.program) - 1:
                    output += ', '
        return output

   


    def get_argument_ranges_for_shape_functions(self, numerical_arguments, categorical_arguments):
        """
        numerical_arguments is a dictionary variable: (lower_bound, upper_bound)
        categorical_arguments is a dictionary: [categorical_value_1, ...]
        """

        if self.is_fitting_necessary(categorical_arguments.keys()):
            if self.model is None:
                raise ValueError("The equation was not fitted. Call raw_fitness function")
        else:
            print("No shape functions to plot")
            return

        
        program = self.model.program_list # this one contains the shape functions before categorical variables

        # No need to deal with single node as the model requires fitting, so it has at least two nodes

        def get_variable_range(variable):
            if variable in numerical_arguments:
                return numerical_arguments[variable]
            elif variable in categorical_arguments:
                return variable
            else:
                raise ValueError("Not all ranges are provided")
        
        def get_operator_range(fun, argument_ranges):
            division_threshold = 1e-3

            if fun.arity == 2:
                a = argument_ranges[0][0]
                b = argument_ranges[0][1]
                c = argument_ranges[1][0]
                d = argument_ranges[1][1]

                if fun.name == 'add':
                    return (a+c,b+d)
                elif fun.name == 'sub':
                    return (a-d,b-c)
                elif fun.name == 'mul':
                    return (min([a*c,a*d,b*c,b*d]), max([a*c,a*d,b*c,b*d]))
                elif fun.name == 'div':
                    zero_possible = False
                    # intersect with (division_threshold, +inf)
                    if argument_ranges[1][1] < division_threshold:
                        return (0.0,0.0)
                    elif argument_ranges[1][0] < division_threshold:
                        new_range = (division_threshold,argument_ranges[1][1])
                        zero_possible = True
                    else:
                        new_range = argument_ranges[1]
                    c = new_range[0]
                    d = new_range[1]
                    if zero_possible:
                        return (min([a/c,a/d,b/c,b/d,0]),max([a/c,a/d,b/c,b/d,0]))
                    else:
                        return (min([a/c,a/d,b/c,b/d]),max([a/c,a/d,b/c,b/d]))

        def get_shape_range(shape_index, argument_range, steps=10000):
            shape_function = self.model.shape_functions[shape_index]

            shape_function.to(torch.device('cpu'))

            t = torch.linspace(argument_range[0],argument_range[1],steps)
            
            pred = shape_function(t)
            lower = torch.min(pred).item()
            upper = torch.max(pred).item()

            return (lower,upper)

        def get_categorical_range(categorical_variable):
            weights = self.model.cat_shape_functions[str(categorical_variable)]
            lower = torch.min(weights)
            upper = torch.max(weights)
            return (lower,upper)

        stack = []

        shape_counter = 0
        shape_ranges = {}


        for node in program:
            if isinstance(node, _Function):
                if node.name == 'shape':
                    stack.append([(shape_counter,node)])
                    shape_counter += 1
                else:
                    stack.append([(-1,node)])
            else: # it's a variable 
                stack[-1].append(get_variable_range(node))

            while stack[-1][0][1].arity == len(stack[-1][1:]):
                f = stack[-1][0][1]
                index = stack[-1][0][0]
                if f.name == 'shape':
                    if not isinstance(stack[-1][1],tuple):
                        intermediate_range = get_categorical_range(stack[-1][1])
                    else:
                        intermediate_range = get_shape_range(index,stack[-1][1])
                        shape_ranges[index] = stack[-1][1]
                else:
                    intermediate_range = get_operator_range(f,stack[-1][1:])
                
                if len(stack) != 1:
                    stack.pop()
                    stack[-1].append(intermediate_range)
                else:
                    print(shape_ranges)
                    return shape_ranges

    
    def plot_shape_functions(self, numerical_arguments, categorical_arguments, steps=1000):

        shape_arg_ranges = self.get_argument_ranges_for_shape_functions(numerical_arguments, categorical_arguments)

        shapes = self.model.shape_functions

        for i, shape in enumerate(shapes):
            t = torch.linspace(shape_arg_ranges[i][0],shape_arg_ranges[i][1],steps)
            shape.to(torch.device('cpu'))
            with torch.no_grad():
                y = shape(t).flatten()
                plt.plot(t.numpy(),y.numpy())
                plt.show()




    def export_graphviz(self, fade_nodes=None):
        """Returns a string, Graphviz script for visualizing the program.

        Parameters
        ----------
        fade_nodes : list, optional
            A list of node indices to fade out for showing which were removed
            during evolution.

        Returns
        -------
        output : string
            The Graphviz script to plot the tree representation of the program.

        """
        terminals = []
        if fade_nodes is None:
            fade_nodes = []
        output = 'digraph program {\nnode [style=filled]\n'
        for i, node in enumerate(self.program):
            fill = '#cecece'
            if isinstance(node, _Function):
                if i not in fade_nodes:
                    fill = '#136ed4'
                terminals.append([node.arity, i])
                output += ('%d [label="%s", fillcolor="%s"] ;\n'
                           % (i, node.name, fill))
            else:
                if i not in fade_nodes:
                    fill = '#60a6f6'
                if isinstance(node, int):
                    if self.feature_names is None:
                        feature_name = 'X%s' % node
                    else:
                        feature_name = self.feature_names[node]
                    output += ('%d [label="%s", fillcolor="%s"] ;\n'
                               % (i, feature_name, fill))
                else:
                    output += ('%d [label="%.3f", fillcolor="%s"] ;\n'
                               % (i, node, fill))
                if i == 0:
                    # A degenerative program of only one node
                    return output + '}'
                terminals[-1][0] -= 1
                terminals[-1].append(i)
                while terminals[-1][0] == 0:
                    output += '%d -> %d ;\n' % (terminals[-1][1],
                                                terminals[-1][-1])
                    terminals[-1].pop()
                    if len(terminals[-1]) == 2:
                        parent = terminals[-1][-1]
                        terminals.pop()
                        if not terminals:
                            return output + '}'
                        terminals[-1].append(parent)
                        terminals[-1][0] -= 1

        # We should never get here
        return None

    def _depth(self):
        """Calculates the maximum depth of the program tree."""
        terminals = [0]
        depth = 1
        for node in self.program:
            if isinstance(node, _Function):
                terminals.append(node.arity)
                depth = max(len(terminals), depth)
            else:
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
        return depth - 1

    def _length(self):
        """Calculates the number of functions and terminals in the program."""
        return len(self.program)

    def execute(self, X, ohe_matrices={}):
        """Execute the program according to X.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        y_hats : array-like, shape = [n_samples]
            The result of executing the program on X.

        """
        if self.is_fitting_necessary(ohe_matrices.keys()) :
            if self.model is None:
                raise ValueError("The model was not trained")

            dataset = torch.utils.data.TensorDataset(X,*[ohe_matrices[k] for k in self.keys])
            pred_dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.optim_dict['batch_size'], shuffle=False, num_workers=self.optim_dict['num_workers_dataloader'])
            
            accelerator = "gpu" if self.optim_dict['device'] == 'cuda' else 'cpu'

            trainer = pl.Trainer(deterministic=True,devices=1,accelerator=accelerator)
            
            y_pred = torch.concat(trainer.predict(self.model, pred_dataloader)).cpu().numpy()

            # y_pred = self.model.predict(X,ohe_matrices,device=self.optim_dict['device']).cpu().detach().numpy()
            return y_pred
        else:
            X = X.cpu().detach().numpy()
            # Check for single-node programs
            node = self.program[0]
            if isinstance(node, float):
                return np.repeat(node, X.shape[0])
            if isinstance(node, int):
                return X[:, node]

            apply_stack = []

            for node in self.program:

                if isinstance(node, _Function):
                    apply_stack.append([node])
                else:
                    # Lazily evaluate later
                    apply_stack[-1].append(node)

                while len(apply_stack[-1]) == apply_stack[-1][0].arity + 1:
                    # Apply functions that have sufficient arguments
                    function = apply_stack[-1][0]
                    terminals = [np.repeat(t, X.shape[0]) if isinstance(t, float)
                                else X[:, t] if isinstance(t, int)
                                else t for t in apply_stack[-1][1:]]
                    intermediate_result = function(*terminals)
                    if len(apply_stack) != 1:
                        apply_stack.pop()
                        apply_stack[-1].append(intermediate_result)
                    else:
                        return intermediate_result

            # We should never get here
            return None

    def get_all_indices(self, n_samples=None, max_samples=None,
                        random_state=None):
        """Get the indices on which to evaluate the fitness of a program.

        Parameters
        ----------
        n_samples : int
            The number of samples.

        max_samples : int
            The maximum number of samples to use.

        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        indices : array-like, shape = [n_samples]
            The in-sample indices.

        not_indices : array-like, shape = [n_samples]
            The out-of-sample indices.

        """
        if self._indices_state is None and random_state is None:
            raise ValueError('The program has not been evaluated for fitness '
                             'yet, indices not available.')

        if n_samples is not None and self._n_samples is None:
            self._n_samples = n_samples
        if max_samples is not None and self._max_samples is None:
            self._max_samples = max_samples
        if random_state is not None and self._indices_state is None:
            self._indices_state = random_state.get_state()

        indices_state = check_random_state(None)
        indices_state.set_state(self._indices_state)

        not_indices = sample_without_replacement(
            self._n_samples,
            self._n_samples - self._max_samples,
            random_state=indices_state)
        sample_counts = np.bincount(not_indices, minlength=self._n_samples)
        indices = np.where(sample_counts == 0)[0]

        return indices, not_indices

    def _indices(self):
        """Get the indices used to measure the program's fitness."""
        return self.get_all_indices()[0]

    def any_shapes(self):
        for node in self.program:
            if isinstance(node, _Function):
                if node.name == 'shape':
                    return True
        return False

    def any_categorical_variables(self, categorical_variables):
        for node in self.program:
            if isinstance(node, int):
                if node in categorical_variables:
                    return True
        return False

    def raw_fitness(self, X, y, sample_weight, ohe_matrices={}):
        """Evaluate the raw fitness of the program according to X, y.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        sample_weight : array-like, shape = [n_samples]
            Weights applied to individual samples.

        Returns
        -------
        raw_fitness : float
            The raw fitness of the program.

        """

        # Check if the file checkpoints/{self.timestamp}/dictionary.csv exists
        if os.path.isfile(f"checkpoints/{self.timestamp}/dictionary.csv"):
            dictionary = pd.read_csv(f"checkpoints/{self.timestamp}/dictionary.csv",index_col=False)
            new_id = dictionary['id'].max() + 1
        else:
            # Create a directory for the checkpoints
            os.makedirs(f"checkpoints/{self.timestamp}")
            dictionary = pd.DataFrame(columns=['id','equation','raw_fitness','r2'])
            new_id = 0
        

        if not self.is_fitting_necessary(ohe_matrices.keys()):
            y_pred = self.execute(X)
        else: # You need to do training
            
            # model = Model(self,self.optim_dict,seed=0)

            # model.train(X,y,ohe_matrices,device=self.optim_dict['device'])
            # # t1 = time.time()
            # y_pred = model.predict(X,ohe_matrices,device=self.optim_dict['device']).cpu().detach().numpy()
            # # t2 = time.time()
            # # print(f"Whole predicting: {t2-t1}")

            self.keys = sorted(ohe_matrices.keys())
            self.categorical_variables_dict = {k:ohe_matrices[k].shape[1] for k in self.keys}

            model = LitModel(self,seed=0)

            dataset = torch.utils.data.TensorDataset(X,*[ohe_matrices[k] for k in self.keys],y)

            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size

            gen = torch.Generator()
            gen.manual_seed(self.optim_dict['seed'])

            # Use random_split to divide the dataset
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=gen)

           

            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.optim_dict['batch_size'], shuffle=True, num_workers=self.optim_dict['num_workers_dataloader'],generator=gen)
            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=self.optim_dict['batch_size'], shuffle=False, num_workers=self.optim_dict['num_workers_dataloader'],generator=gen)
            
            accelerator = "gpu" if self.optim_dict['device'] == 'cuda' else 'cpu'

            # torch.set_float32_matmul_precision("medium")
            early_stopping = pl.callbacks.EarlyStopping('val_loss',patience=10,min_delta=self.optim_dict['tol'])
            lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
            
           

            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                                monitor='val_loss',
                                dirpath=f'checkpoints/{self.timestamp}',
                                filename=f'{new_id}-best_val_loss',
                                save_top_k=1,
                                mode='min',
                                auto_insert_metric_name=True)


            logger = pl.loggers.TensorBoardLogger("tb_logs", name=f"{self.timestamp}/{new_id}")
            
            trainer = pl.Trainer(default_root_dir='./lightning_logs',logger=logger,deterministic=True,devices=1,check_val_every_n_epoch=10,callbacks=[early_stopping,lr_monitor,checkpoint_callback],auto_lr_find=True,enable_model_summary = False,enable_progress_bar=True,log_every_n_steps=10,auto_scale_batch_size=False,accelerator=accelerator,max_epochs=self.optim_dict['max_n_epochs'])
            
            trainer.tune(model,train_dataloaders=train_dataloader)
            
            trainer.fit(model=model,train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

            # Load the best model
            model = LitModel.load_from_checkpoint(f"checkpoints/{self.timestamp}/{new_id}-best_val_loss.ckpt", program=self)

            self.model = model

            # val_loss = trainer.callback_metrics['val_loss'].item()
            # print(f"val loss: {val_loss}")

            pred_dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.optim_dict['batch_size'], shuffle=False, num_workers=self.optim_dict['num_workers_dataloader'])

            y_pred = torch.concat(trainer.predict(model, pred_dataloader)).cpu().numpy()

            if 'keep_models' in self.optim_dict.keys():
                keep_the_model = self.optim_dict['keep_models']
            else:
                keep_the_model = False
            if not keep_the_model:
                # Delete the model
                os.remove(f"checkpoints/{self.timestamp}/{new_id}-best_val_loss.ckpt")
      
            
            # return val_loss

        if self.transformer:
            y_pred = self.transformer(y_pred)

        y_numpy = y.cpu().numpy()
        raw_fitness = self.metric(y_numpy, y_pred, sample_weight)
        if self.optim_dict['task'] == 'regression':
            r2 = r2_score(y_numpy, y_pred)
        else:
            logits = _sigmoid(y_pred)
            r2 = roc_auc_score(y_numpy,logits)
        print(f"{self} | raw_fitness: {raw_fitness}")

        new_row = pd.DataFrame({"id":[new_id],"equation":[str(self)],"raw_fitness":[raw_fitness],"r2":[r2]})
        dictionary = pd.concat([dictionary,new_row],ignore_index=True)
        dictionary.to_csv(f"checkpoints/{self.timestamp}/dictionary.csv",index=False)

        return raw_fitness

    def fitness(self, parsimony_coefficient=None):
        """Evaluate the penalized fitness of the program according to X, y.

        Parameters
        ----------
        parsimony_coefficient : float, optional
            If automatic parsimony is being used, the computed value according
            to the population. Otherwise the initialized value is used.

        Returns
        -------
        fitness : float
            The penalized fitness of the program.

        """
        if parsimony_coefficient is None:
            parsimony_coefficient = self.parsimony_coefficient
        penalty = parsimony_coefficient * len(self.program) * self.metric.sign
        return self.raw_fitness_ - penalty

    def get_possible_subtree_roots(self, variables, program=None):
        if program is None:
            program = self.program

        # deal with a program with a single node
        node = program[0]
        if isinstance(node, int):
            if node in variables:
                return [(0,'leaf')]
            else:
                return []

        # each element of stack is a tuple (a,b,c) where a is the index in program, b is the arity, c is a list of sets of active variables      stack = []
        
        stack = []
        subtrees = []

        for index, node in enumerate(program):

            if isinstance(node, _Function):
                stack.append((index, node.arity, []))
            else: # it's a variable
                stack[-1][2].append({node})
                if node in variables:
                    subtrees.append((index, 'leaf'))

            while len(stack[-1][2]) == stack[-1][1]:
                active_variables = reduce(lambda x, y: x.union(y), stack[-1][2])
                if active_variables.issubset(variables):
                        start = stack[-1][0]
                        if stack[-1][1] == 1:
                            function_type = 'single'
                        else:
                            function_type = 'operator'
                        subtrees.append((start,function_type))
                
                if len(stack) != 1:
                    stack.pop()
                    stack[-1][2].append(active_variables)
                else:
                    return subtrees
        
        # We should never get here
        return None 


    def get_subtree(self, random_state, program=None, variables=None):
        """Get a random subtree from the program.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        program : list, optional (default=None)
            The flattened tree representation of the program. If None, the
            embedded tree in the object will be used.

        Returns
        -------
        start, end : tuple of two ints
            The indices of the start and end of the random subtree.

        """
        if program is None:
            program = self.program

        if variables == None:
            # Choice of crossover points follows Koza's (1992) widely used approach
            # of choosing functions 90% of the time and leaves 10% of the time.
            probs = np.array([0.9 if isinstance(node, _Function) else 0.1
                            for node in program])
            probs = np.cumsum(probs / probs.sum())
            start = np.searchsorted(probs, random_state.uniform())
        else:
            possible_roots = self.get_possible_subtree_roots(variables,program=program)
            if len(possible_roots) == 0:
                return None
            probs = np.array([0.1 if root[1] == 'leaf' else 0.9
                            for root in possible_roots])
            probs = np.cumsum(probs / probs.sum())
            start_raw = np.searchsorted(probs, random_state.uniform())
            start = possible_roots[start_raw][0]

        stack = 1
        end = start
        while stack > end - start:
            node = program[end]
            if isinstance(node, _Function):
                stack += node.arity
            end += 1

        return start, end

    def reproduce(self):
        """Return a copy of the embedded program."""
        return copy(self.program)

    def crossover(self, donor, random_state):
        """Perform the crossover genetic operation on the program.

        Crossover selects a random subtree from the embedded program to be
        replaced. A donor also has a subtree selected at random and this is
        inserted into the original parent to form an offspring.

        Parameters
        ----------
        donor : list
            The flattened tree representation of the donor program.

        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        # Get a subtree to replace
        start, end = self.get_subtree(random_state)
        removed = range(start, end)
        active_variables_whole = self.find_active_variables()
        active_variables_removed = self.find_active_variables(self.program[start:end])
        all_possible_variables = set(range(self.n_features))
        active_variables_left = (active_variables_whole - active_variables_removed)
        active_variables_possible = all_possible_variables - active_variables_left
        # Get a subtree to donate
        result = self.get_subtree(random_state, donor, active_variables_possible)
        if result is not None:
            donor_start, donor_end = result
        else:
            return self.program, [], []
    
        # Check if there is a redundant shape function
        if start != 0:
            if self.is_shape_function(self.program[start-1]) and self.is_shape_function(donor[donor_start]):
                donor_start += 1
        
        donor_removed = list(set(range(len(donor))) -
                             set(range(donor_start, donor_end)))
        # Insert genetic material from donor
        return (self.program[:start] +
                donor[donor_start:donor_end] +
                self.program[end:]), removed, donor_removed

    def subtree_mutation(self, random_state):
        """Perform the subtree mutation operation on the program.

        Subtree mutation selects a random subtree from the embedded program to
        be replaced. A donor subtree is generated at random and this is
        inserted into the original parent to form an offspring. This
        implementation uses the "headless chicken" method where the donor
        subtree is grown using the initialization methods and a subtree of it
        is selected to be donated to the parent.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        start, end = self.get_subtree(random_state)
        removed = range(start, end)
        active_variables_whole = self.find_active_variables()
        active_variables_removed = self.find_active_variables(self.program[start:end])
        all_possible_variables = set(range(self.n_features))
        active_variables_left = (active_variables_whole - active_variables_removed)
        active_variables_possible = all_possible_variables - active_variables_left

        num_of_variables = self.random_skewed_integer(random_state,1,len(active_variables_possible)+1)
        chosen_active_variables = set(random_state.choice(list(active_variables_possible),size=num_of_variables,replace=False))

        # Build a new naive program
        chicken = self.build_program(random_state, variables=chosen_active_variables)
        
        donor_start = 0
        if start != 0:
            if self.is_shape_function(chicken[0]) and self.is_shape_function(self.program[start-1]):
                donor_start = 1
        
        return (self.program[:start] + chicken[donor_start:] + self.program[end:]), removed, range(donor_start,len(chicken))

        # # Do subtree mutation via the headless chicken method!
        # return self.crossover(chicken, random_state)

    def hoist_mutation(self, random_state):
        """Perform the hoist mutation operation on the program.

        Hoist mutation selects a random subtree from the embedded program to
        be replaced. A random subtree of that subtree is then selected and this
        is 'hoisted' into the original subtrees location to form an offspring.
        This method helps to control bloat.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        # Get a subtree to replace
        start, end = self.get_subtree(random_state)
        subtree = self.program[start:end]
        # Get a subtree of the subtree to hoist
        sub_start, sub_end = self.get_subtree(random_state, subtree)
        hoist = subtree[sub_start:sub_end]
        # Determine which nodes were removed for plotting
        removed = list(set(range(start, end)) -
                       set(range(start + sub_start, start + sub_end)))
        return self.program[:start] + hoist + self.program[end:], removed

    def point_mutation(self, random_state):
        """Perform the point mutation operation on the program.

        Point mutation selects random nodes from the embedded program to be
        replaced. Terminals are replaced by other terminals and functions are
        replaced by other functions that require the same number of arguments
        as the original node. The resulting tree forms an offspring.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        program = copy(self.program)



        # Get the nodes to modify
        mutate = np.where(random_state.uniform(size=len(program)) <
                          self.p_point_replace)[0]

        all_possible_variables = set([int(i) for i in range(self.n_features)])
        active_variables = self.find_active_variables()
        not_active_variables = all_possible_variables - active_variables
       

        # Add variables to be mutated
        for node in mutate:
            if isinstance(program[node], int):
                not_active_variables.add(program[node])
        

        for node in mutate:
            if isinstance(program[node], _Function):
                arity = program[node].arity
                # Find a valid replacement with same arity
                replacement = len(self.arities[arity])
                replacement = random_state.randint(replacement)
                replacement = self.arities[arity][replacement]
                program[node] = replacement
            else:
                terminal = int(random_state.choice(list(not_active_variables)))
                program[node] = terminal
                not_active_variables.remove(terminal)

    

        # for node in mutate:
        #     if isinstance(program[node], _Function):
        #         arity = program[node].arity
        #         # Find a valid replacement with same arity
        #         replacement = len(self.arities[arity])
        #         replacement = random_state.randint(replacement)
        #         replacement = self.arities[arity][replacement]
        #         program[node] = replacement
        #     else:
        #         # We've got a terminal, add a const or variable
        #         if self.const_range is not None:
        #             terminal = random_state.randint(self.n_features + 1)
        #         else:
        #             terminal = random_state.randint(self.n_features)
        #         if terminal == self.n_features:
        #             terminal = random_state.uniform(*self.const_range)
        #             if self.const_range is None:
        #                 # We should never get here
        #                 raise ValueError('A constant was produced with '
        #                                  'const_range=None.')
        #         program[node] = terminal

        return program, list(mutate)

    depth_ = property(_depth)
    length_ = property(_length)
    indices_ = property(_indices)
