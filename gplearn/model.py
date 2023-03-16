import torch

from .functions import _Function, _function_map

import numpy as np

import time

import pytorch_lightning as pl

MAX_FLOAT = 10e9

class ShapeFunction():

    def __init__(self):
        pass

    def __call__(self):
        pass

class ShapeNN(torch.nn.Module,ShapeFunction):

    def __init__(self, n_hidden_layers, width, activation_name='ReLU'):
        super(ShapeNN, self).__init__()
        self.n_hidden_layers = n_hidden_layers
        self.width = width
        if activation_name == 'ReLU':
            activation = torch.nn.ReLU()
        elif activation_name == 'Sigmoid':
            activation = torch.nn.Sigmoid()
        elif activation_name == 'ELU':
            activation = torch.nn.ELU()
        self.batch_norm = torch.nn.BatchNorm1d(1)
        self.input_layer = torch.nn.Linear(1,self.width)
        self.input_activation = activation
        self.hidden_layers = []
        for i in range(self.n_hidden_layers):
            self.hidden_layers.append(torch.nn.Linear(self.width,self.width))
            self.hidden_layers.append(activation)
        self.output_layer = torch.nn.Linear(self.width,1)
        self.nn = torch.nn.Sequential(self.batch_norm,self.input_layer, 
                                        self.input_activation,
                                        *self.hidden_layers,
                                        self.output_layer)
    def forward(self, x):
        return self.nn(x.unsqueeze(1))

def torch_safe_division(x,y):
    mask = torch.abs(y) > 1e-3
    result = torch.zeros_like(x)
    result[mask] = torch.div(x[mask],y[mask]).float()
    return result

torch_functions = {
    'add':torch.add,
    'sub':torch.sub,
    'mul':torch.mul,
    # 'div':lambda x,y: torch.where(torch.abs(y) < 1e-3, 1000.0 * x, torch.div(x,y)),
    'div': torch_safe_division,
    'sin':torch.sin,
    'cos':torch.cos
}

class LitModel(pl.LightningModule):

    def __init__(self, program, seed=0):
        super().__init__()
        print(program)
        self.seed = seed
        self.program = program
        self.program_list = program.program
        self.optim_dict = self.program.optim_dict
        self.shape_class = self.optim_dict['shape_class']
        self.constructor_dict = self.optim_dict['constructor_dict']
        self.categorical_variables_dict = self.program.categorical_variables_dict
        self.categorical_variables = self.program.keys
        self.lr = self.optim_dict['lr']

        torch.manual_seed(self.seed)

        # Choose the loss function
        if self.optim_dict['task'] == 'regression':
            self.loss_fn = torch.nn.MSELoss()
        elif self.optim_dict['task'] == 'classification':
            self.loss_fn = torch.nn.BCEWithLogitsLoss()

        # Add shape functions in front of categorical variables if there are none
        self._add_categorical_functions(self.categorical_variables)

        # Add shape function in pytorch
        self.shape_functions = []
        self.cat_shape_functions = torch.nn.ParameterDict()
        for i, node in enumerate(self.program_list):
            if isinstance(node, _Function):
                if node.name == 'shape':
                    next_node = self.program_list[i+1] 
                    if isinstance(next_node, int):
                        if next_node in self.categorical_variables:
                            continue
                    self.shape_functions.append(self.shape_class(**self.constructor_dict))
        self.shape_functions = torch.nn.ModuleList(self.shape_functions)
        for categorical_variable, n_categories in self.categorical_variables_dict.items():
            self.cat_shape_functions[str(categorical_variable)] = torch.nn.Parameter(torch.randn(n_categories),requires_grad=True)


    def _add_categorical_functions(self, categorical_variables):

        shape_function = _function_map['shape']

        cat_indices = []

        for i, node in enumerate(self.program_list):
            if isinstance(node,int):
                if node in categorical_variables:
                    if i > 0:
                        prev_node = self.program_list[i-1]
                        if not isinstance(prev_node, _Function):
                            cat_indices.append(i)
                        else:
                            if prev_node.name != 'shape':
                                cat_indices.append(i)
                    else:
                        cat_indices.append(i)
        
        shift = 0
        for ind in cat_indices:
            self.program_list.insert(ind+shift,shape_function)
            shift += 1       


    def training_step(self, batch, batch_idx):
     
        batch_X_and_ohe = batch[:-1]
        batch_y = batch[-1]
              
        pred = self(batch_X_and_ohe)
               
        loss = self.loss_fn(pred, batch_y)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_X_and_ohe = batch[:-1]
        batch_y = batch[-1]
              
        pred = self(batch_X_and_ohe)
               
        loss = self.loss_fn(pred, batch_y)
        self.log('val_loss', loss)
        return loss
      

    def test_step(self, batch, batch_idx):
        batch_X_and_ohe = batch[:-1]
        batch_y = batch[-1]
              
        pred = self(batch_X_and_ohe)
               
        loss = self.loss_fn(pred, batch_y)
        self.log('test_loss', loss)
        return loss
      

    def configure_optimizers(self):

        if self.optim_dict['alg'] == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optim_dict['alg'] == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.parameters(),lr=self.lr)

        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100, factor=0.5)

    #     return {
    #     "optimizer": optimizer,
    #     "lr_scheduler": {
    #         "scheduler": lr_scheduler,
    #         "monitor": "train_loss",
    #         # "frequency": 5
    #         # If "monitor" references validation metrics, then "frequency" should be set to a
    #         # multiple of "trainer.check_val_every_n_epoch".
    #     },
    # }
       
        return optimizer

    def forward(self, batch):
        batch_X = batch[0]
        batch_ohe = batch[1:]
        return self.evaluate_equation(batch_X, batch_ohe)

    # def predict_step(self, batch, batch_idx, dataloader_idx=0):
    #     batch_X = batch[0]
    #     batch_ohe = batch[1:]
    #     return self.evaluate_equation(batch_X, batch_ohe)

    def evaluate_equation(self, X, ohe_matrices):

        program = self.program_list # already has "padded" shapes

        shape_counter = 0

        node = program[0]
        if isinstance(node, int):
            return X[:,node]

        ohe_matrices = {k:ohe_matrices[i] for i, k in enumerate(self.categorical_variables)}
        
        apply_stack = []
        # apply_stack is a stack. Each element is a list where the first element is function tuple and the rest are arguments
        # the function tuple consists of the index of shape and the node itself

        for i, node in enumerate(program):
            # print(node)
            # print(type(node))
            if isinstance(node, _Function):
                apply_stack.append([(shape_counter,node)])
                if node.name == 'shape':
                    next_node = self.program_list[i+1] 
                    if isinstance(next_node, int):
                        if next_node not in self.categorical_variables:
                            shape_counter += 1
                    else:
                        shape_counter += 1
                    
            else:
                apply_stack[-1].append(node)
            
            while len(apply_stack[-1]) == apply_stack[-1][0][1].arity + 1:
                
                function = apply_stack[-1][0][1]
                index = apply_stack[-1][0][0]
                terminals = [X[:,t] if isinstance(t,int) 
                             else t for t in apply_stack[-1][1:]]
                if function.name == 'shape':
                    # Check if the argument is a categorical variable
                    raw_arg = apply_stack[-1][1] # this is the index of the variable
                    if isinstance(raw_arg, int):
                        if raw_arg in self.categorical_variables:
                            # Categorical
                            intermediate_result = torch.matmul(ohe_matrices[raw_arg],self.cat_shape_functions[str(raw_arg)])
                        else:
                            # Not categorical
                            intermediate_result = self.shape_functions[index](terminals[0]).flatten().float()
                    else:
                        # Not categorical
                        intermediate_result = self.shape_functions[index](terminals[0]).flatten().float()
                else:
                    function = torch_functions[function.name]
                    intermediate_result = function(*terminals).float()
                
                if len(apply_stack) != 1:
                    apply_stack.pop()
                    apply_stack[-1].append(intermediate_result)
                else:
                    return intermediate_result




class Model(torch.nn.Module):

    def __init__(self, program, optim_dict, seed=0):
        super(Model, self).__init__()
        print(program)
        self.seed = seed
        self.program = program
        self.program_list = program.program
        self.shape_class = optim_dict['shape_class']
        self.constructor_dict = optim_dict['constructor_dict']
        self.seed = seed
        self.optim_dict = optim_dict

    def _add_categorical_functions(self, categorical_variables):

        shape_function = _function_map['shape']

        cat_indices = []

        for i, node in enumerate(self.program_list):
            if isinstance(node,int):
                if node in categorical_variables:
                    if i > 0:
                        prev_node = self.program_list[i-1]
                        if not isinstance(prev_node, _Function):
                            cat_indices.append(i)
                        else:
                            if prev_node.name != 'shape':
                                cat_indices.append(i)
                    else:
                        cat_indices.append(i)
        
        shift = 0
        for ind in cat_indices:
            self.program_list.insert(ind+shift,shape_function)
            shift += 1       
        # for node in self.program:
        #     if isinstance(node,_Function):
        #         print(node.name)
        #     else:
        #         print(node)
       
    def forward(self,X, ohe_matrices):

        # print(X.size())

        program = self.program_list

        keys = self.keys

        shape_counter = 0
        # cat_shape_counter = 0

        node = program[0]
        if isinstance(node, int):
            return X[:,node]

        ohe_matrices = {k:ohe_matrices[i] for i, k in enumerate(keys)}
        
        apply_stack = []

        for node in program:

            if isinstance(node, _Function):
                apply_stack.append([node])
            else:
                apply_stack[-1].append(node)
            
            while len(apply_stack[-1]) == apply_stack[-1][0].arity + 1:
                
                function = apply_stack[-1][0]
                terminals = [X[:,t] if isinstance(t,int) 
                             else t for t in apply_stack[-1][1:]]
                if function.name == 'shape':
                    # Check if the argument is a categorical variable
                    raw_arg = apply_stack[-1][1]
                    if isinstance(raw_arg, int):
                        if raw_arg in ohe_matrices.keys():
                            # Categorical
                            # print(ohe_matrices[raw_arg].device)
                            # print(self.cat_shape_functions[str(raw_arg)].device)
                            intermediate_result = torch.matmul(ohe_matrices[raw_arg],self.cat_shape_functions[str(raw_arg)])
                            # print(intermediate_result.size())
                            # cat_shape_counter += 1
                        else:
                            # Not categorical
                            intermediate_result = self.shape_functions[shape_counter](terminals[0]).flatten().float()
                            shape_counter += 1
                    else:
                        # Not categorical
                        intermediate_result = self.shape_functions[shape_counter](terminals[0]).flatten().float()
                        shape_counter += 1
                else:
                    function = torch_functions[function.name]
                    intermediate_result = function(*terminals).float()
                
                if len(apply_stack) != 1:
                    apply_stack.pop()
                    apply_stack[-1].append(intermediate_result)
                else:
                    return intermediate_result
    
    def predict(self, X, ohe_matrices={}, device='cpu'):
        if (self.shape_functions is None) or (self.cat_shape_functions is None):
            raise ValueError("You cannot predict before the model is trained")

        device = torch.device(device)

        self.to(device)

        keys = sorted(ohe_matrices.keys())
        self.keys = keys

        if isinstance(X, torch.Tensor):
            data = torch.utils.data.TensorDataset(X,*[ohe_matrices[k] for k in keys])
        else:
            data = torch.utils.data.TensorDataset(torch.tensor(np.array(X), dtype=torch.float32, device=device),*[ohe_matrices[k] for k in keys])

        # Create a dataloader to iterate over the dataset in batches
        dataloader = torch.utils.data.DataLoader(data, batch_size=self.optim_dict['batch_size'], shuffle=False)
        
        predictions = []
        for batch_data in dataloader:
            batch_X = batch_data[0]
            batch_ohe = batch_data[1:]
            # Forward pass: compute predicted y by passing x to the model.
            y_pred = self(batch_X,batch_ohe)

            # Store the predicted and true labels
            predictions.append(y_pred)

        # Concatenate the predicted and true labels into single tensors
        predictions = torch.cat(predictions)

        return predictions
    
    def train(self, X, y, ohe_matrices={}, device='cpu'):

        # print(ohe_matrices)

        torch.manual_seed(self.seed)

        device = torch.device(device)

        # Add shape functions in front of categorical variables if there are none
        self._add_categorical_functions(list(ohe_matrices.keys()))
        self.shape_functions = []
        self.cat_shape_functions = torch.nn.ParameterDict()

        for i, node in enumerate(self.program_list):
            if isinstance(node, _Function):
                if node.name == 'shape':
                    next_node = self.program_list[i+1] 
                    if isinstance(next_node, int):
                        if next_node in ohe_matrices.keys():
                            continue
                    self.shape_functions.append(self.shape_class(**self.constructor_dict))
        self.shape_functions = torch.nn.ModuleList(self.shape_functions)

        keys = sorted(ohe_matrices.keys())


        for categorical_variable in ohe_matrices.keys():
            n_categories = ohe_matrices[categorical_variable].shape[1]
            self.cat_shape_functions[str(categorical_variable)] = torch.nn.Parameter(torch.randn(n_categories),requires_grad=True)

        self.keys = keys

        # if len(self.shape_functions) == 0:
        #     self.shape_functions.append(torch.nn.Linear(1,1))

        # if len(keys) == 0:
        #     self.cat_shape_functions["-1"] = torch.nn.Parameter(torch.randn(1),requires_grad=True)

            # keys.append(-1)
            # ohe_matrices[-1] = torch.nn.Parameter(torch.ones(X.size(0),1,requires_grad=True))

        if self.optim_dict['alg'] == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.optim_dict['lr'])
        elif self.optim_dict['alg'] == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.parameters(),lr=self.optim_dict['lr'])
        task = self.optim_dict['task']
        
        # X = torch.from_numpy(np.array(X)).float()
        # y = torch.from_numpy(np.array(y)).float()
        epochs_no_imp = 0
        prev_loss = 10e9

        self.to(device)

        # for key in ohe_matrices:
        #     ohe_matrices[key] = ohe_matrices[key].to(device)

        
        # t1 = time.time()
        # Convert the data to tensors and wrap them in a dataset
        data = torch.utils.data.TensorDataset(X,y,*[ohe_matrices[k] for k in keys])
        # t2 = time.time()
        # print(f"Creating data object: {t2-t1}")

        # Create a dataloader to iterate over the dataset in batches
        # t1 = time.time()
        dataloader = torch.utils.data.DataLoader(data, batch_size=self.optim_dict['batch_size'], shuffle=True)
        # t2 = time.time()
        # print(f"Creating dataloader object: {t2-t1}")

        if task == 'regression':
            loss_fn = torch.nn.MSELoss()
        elif task == 'classification':
            loss_fn = torch.nn.BCEWithLogitsLoss()
        
        
        t1 = time.time()
        for i in range(self.optim_dict['max_n_epochs']):

            combined_loss = 0

            for batch in dataloader:
                batch_X = batch[0]
                batch_y = batch[1]
                batch_ohe = batch[2:]
                # if traced_model is None:
                #     with torch.no_grad():
                #         traced_model = torch.jit.trace(self,(batch_X,batch_ohe))
                #         # print(traced_model.code)
                #         # print(loss_fn(traced_model(batch_X,batch_ohe), batch_y))
                #         # print(loss_fn(self(batch_X,batch_ohe),batch_y))

                # t21 = time.time()
                # pred = traced_model(batch_X,batch_ohe)
                # t22 = time.time()

                # t31 = time.time()
                pred = self(batch_X,batch_ohe)
                # print(batch_X.shape)
                # print(pred.shape,batch_y.shape)
                # t32 = time.time()

                # print(f"Traced - python: {(t22-t21) - (t32-t31)}")
                loss = loss_fn(pred, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                combined_loss += loss.item() * batch_X.shape[0]
            
            combined_loss /= X.shape[0]

            if prev_loss - combined_loss <= self.optim_dict['tol']:
                epochs_no_imp += 1
            else:
                epochs_no_imp = 0
            if epochs_no_imp == self.optim_dict['n_iter_no_change']:
                # print(f"Stopped at iteration: {i+1} | Loss {combined_loss}")
                break
            prev_loss = combined_loss
            # print(prev_loss)
            if i == self.optim_dict['max_n_epochs'] - 1:
                print(f"Did not converge in {self.optim_dict['max_n_epochs']}. Loss {combined_loss}")
            if (i > 10) and (prev_loss > MAX_FLOAT):
                # print(f"Optimization terminated at iteration {i+1}. Loss exceeded {MAX_FLOAT}. Loss {self.curr_loss}")
                break
        t2 = time.time()
        print(f"Training: {t2-t1}")
        self.loss = combined_loss










