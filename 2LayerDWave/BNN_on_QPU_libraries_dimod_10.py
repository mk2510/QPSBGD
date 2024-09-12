#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
sys.executable

import numpy as np

from libsvm.svmutil import *

import matplotlib.pylab as plt
import numpy as np

import sys
import struct
import networkx as nx
import dwave_networkx as dnx
import matplotlib.pyplot as plt
import numpy as np
import dimod
import collections
import math
import pickle
import json
from dwave.system import DWaveSampler, EmbeddingComposite, FixedEmbeddingComposite
from dwave.cloud import Client
from dwave.system import DWaveSampler
from dwave.embedding import EmbeddedStructure
import dwave.inspector
import dwave.embedding
from minorminer import find_embedding

from qubovert.sim import anneal_pubo, anneal_qubo

import itertools

import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Function
import math

from qubovert import *

import gurobipy as grb

import time


#global USE_OLD_XNOR
USE_OLD_XNOR = True
USE_OLD_GREATER_ZERO = True

# In[ ]:


#! wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a
#! wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a.t


# In[ ]:


Leap_API_token = '' ### from the leap account dashboard at https://cloud.dwavesys.com/


DATA = './datafolder/'


# In[ ]:


client = Client.from_config(token=Leap_API_token)

solvers = client.get_solvers(num_qubits__gt=3000)
solvers

solver = solvers[0]
solver


# In[ ]:


G = nx.Graph()
G.add_edges_from( solver.edges )


# In[ ]:


np.random.seed(2)


# In[ ]:


##https://pypi.org/project/libsvm/
##https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#a1a


# In[ ]:


def data_to_matrix(x):
    x_data = np.zeros((len(x),123))
    exists = np.zeros(123)
    for i,x_i in enumerate( x ):
        for j in range(123):
            if j in x_i.keys():
                assert x_i[j] == 1
                x_data[i,j] = 1.
                exists[j] = 1.
    return x_data, exists


# In[ ]:


def set_up_data(sel=3):
    
    ## selects a subset of features from the a1a dataset
    
    #global x
    global y
    #global x_test
    global y_test
    global select
    global x_data_reduced
    global x_test_data_reduced
    y, x = svm_read_problem(DATA +'/a1a')
    y_test, x_test = svm_read_problem(DATA +'/a1a.t')
    np.random.seed(2)
    train_indexes = list(range(len(y)))
    np.random.shuffle(train_indexes)
    y = [y[i] for i in train_indexes]
    x = [x[i] for i in train_indexes]
    x_data, exists = data_to_matrix(x)
    x_test_data, exists_test = data_to_matrix(x_test)

    if sel== 0.3:
        select = np.mean(x_data,0) > .3
        np.array(range(123))[select]
    elif sel== 7:     ### selects 7 features
        select = (((np.mean(x_data,0) > .3) + 0.) + (np.mean(x_data,0) < .65)) == 2.
        np.array(range(123))[select]
    elif sel== 9:      ### selects 9 features
        select = (((np.mean(x_data,0) > .3) + 0.) + (np.mean(x_data,0) < .7)) == 2.
        np.array(range(123))[select]
        
    elif sel== 15:      ### selects 15 features
        select = (((np.mean(x_data,0) > .21) + 0.) + (np.mean(x_data,0) < .79)) == 2.
        np.array(range(123))[select]
        
    elif sel == 3:      ### selects 3 features
        select = np.array([False for i in range(123)])
        select[6] = True
        select[22] = True
        select[36] = True
        select
    elif sel== 'all':
        select = np.array([True for i in range(123)])
    else:
        assert False

    x_data_reduced = x_data[:,select]
    x_test_data_reduced = x_test_data[:,select]
    
    
    y = 0+(np.array(y) > 0.)
    y_test = 0+(np.array(y_test) > 0.)

#set_up_data(3)


# In[ ]:




# In[ ]:


# BNN code adapted from:
# github.com/Akashmathwani/Binarized-Neural-networks-using-pytorch
# 


class BinarizeF(Function):

    @staticmethod
    def forward(cxt, input):
        output = input.new(input.size())
        output[input >= 0] = 1
        output[input < 0] = -1
        return output

    @staticmethod
    def backward(cxt, grad_output):
        grad_input = grad_output.clone()
        return grad_input

# aliases
binarize = BinarizeF.apply

class BinaryTanh(nn.Module):
    def __init__(self):
        super(BinaryTanh, self).__init__()
        self.hardtanh = nn.Hardtanh()

    def forward(self, input):
        output = self.hardtanh(input)
        output = binarize(output)
        
        if num_classes == 2:
            output += 1
            output /= 2
            output = output.squeeze()
        return output
        

class BinaryLinear(nn.Linear):

    def forward(self, input):
        binary_weight = binarize(self.weight)
        if self.bias is None:
            return F.linear(input, binary_weight)
        else:
            return F.linear(input, binary_weight, self.bias)

    def reset_parameters(self):
        # Glorot initialization
        in_features, out_features = self.weight.size()
        stdv = math.sqrt(1.5 / (in_features + out_features))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

        self.weight.lr_scale = 1. / stdv



# In[ ]:



## Define the NN architecture
class BinaryNet(nn.Module):
    def __init__(self):
        super(BinaryNet, self).__init__()
        self.fc_list = nn.ModuleList()
        if num_classes == 2:
            output_net_size = 1
        else:
            output_net_size = num_classes
        input_layer_size = input_size
        output_layer_size = hidden_n
        
        for i in range(n_layers):
            if i == n_layers -1:
                output_layer_size = output_net_size
            self.fc_list.append( BinaryLinear(input_layer_size, output_layer_size, bias=False) )
            input_layer_size = hidden_n
            
        self.dropout = nn.Dropout(0.2)
        self.activation = BinaryTanh()

    def forward(self, x):
        # flatten image input
        x = x.view(-1, input_size)
        # add hidden layer, with relu activation function
        for fc_layer in self.fc_list:
            x = self.activation(fc_layer(x))
        return x

# # initialize the NN
# model = BinaryNet()
# print(model)


# In[ ]:




# In[ ]:


def set_up_qubo_model(layers_to_optimize=[1,2]):
    global qubo_vars
    global H
    global image_index
    global H_layers_to_optim
    H_layers_to_optim = layers_to_optimize
    qubo_vars = {} ## dict of variables
    for j_layer in layers_to_optimize:   ## second and third layer
        for i in range( getattr(model, 'fc_list')[j_layer].weight.shape[0] ):
            for k in range( getattr(model, 'fc_list')[j_layer].weight.shape[1] ):
                qubo_vars[f'weight_{j_layer}_{i}_{k}'] = boolean_var( f'weight_{j_layer}_{i}_{k}' )
    H = PCBO()
    image_index = 0


# In[ ]:


def add_constrains_qubo( input_x, gt, Lambda=1):
    #target_index_i = np.zeros(num_classes)
    #target_index_i[int(gt)] = 1.
    global qubo_vars
    global H
    global image_index
    for k_layer_i in range(len(H_layers_to_optim)):
        k_layer = H_layers_to_optim[k_layer_i]
        if k_layer_i == 0:
            fc_in = input_x
            
        ### loop over output dimension of layer
        for j in range(  getattr(model, 'fc_list')[k_layer].weight.shape[0]  ):
            partial = 0.
            count = 0.
            ### loop over input dimension of layer
            for i in range( getattr(model, 'fc_list')[k_layer].weight.shape[1] ):   ## loop over input dimension of layer
                #print( k_layer_i, j, i, image_index)
                qubo_vars[f'partial_matrix_product_{k_layer}_{i}_{j}_{image_index}'] = boolean_var(f'partial_matrix_product_{k_layer}_{i}_{j}_{image_index}')
                if k_layer_i == 0:
                    ### the first layer does not require xnor
                    if fc_in[i] == 1:
                        #print('fc_in 1', k_layer_i, j, i, image_index)
                        #print('fc_in', fc_in[i])
                        H.add_constraint_eq_BUFFER(
                                    qubo_vars[f'partial_matrix_product_{k_layer}_{i}_{j}_{image_index}'],
                                    qubo_vars[f'weight_{k_layer}_{j}_{i}'],
                                    #lam=Lambda/getattr(model, 'fc_list')[k_layer].weight.shape[0]
                                    lam=Lambda
                                                  )
                    if fc_in[i] == 0:
                        #print('fc_in 0', k_layer_i, j, i, image_index)
                        #print('fc_in', fc_in[i])
                        H.add_constraint_eq_NOT( 
                                     qubo_vars[f'partial_matrix_product_{k_layer}_{i}_{j}_{image_index}'],
                                     qubo_vars[f'weight_{k_layer}_{j}_{i}'],
                                     lam=Lambda
                                               )
                else:
                    if USE_OLD_XNOR: 
                        H.add_constraint_eq_XNOR(
                                    qubo_vars[f'partial_matrix_product_{k_layer}_{i}_{j}_{image_index}'],
                                    qubo_vars[f'weight_{k_layer}_{j}_{i}'],
                                    fc_in[i] ,
                                    lam=Lambda
                                            )
                    else:
                        add_constraint_XNOR(
                                    qubo_vars[f'partial_matrix_product_{k_layer}_{i}_{j}_{image_index}'],
                                    qubo_vars[f'weight_{k_layer}_{j}_{i}'],
                                    fc_in[i] ,
                                    lam=Lambda,
                                    k_layer=k_layer_i,
                                    j=j,
                                    image_index=image_index,
                                    input_dim_i=i
                                            )
                partial += qubo_vars[f'partial_matrix_product_{k_layer}_{i}_{j}_{image_index}']
                count +=1.
            qubo_vars[f'matrix_product_{k_layer}_{j}_{image_index}'] = boolean_var(f'matrix_product_{k_layer}_{j}_{image_index}')
            #print('partial', partial)
            ## the activation function is a threshold
            if USE_OLD_GREATER_ZERO:
                add_constraint_greater_zero( count,
                                        partial,
                                        output_bool=qubo_vars[f'matrix_product_{k_layer}_{j}_{image_index}'],
                                        lam=Lambda,
                                        k_layer=k_layer,
                                        j=j,
                                        image_index=image_index,
                                       )
            else:
                #print(count)
                add_constraint_greater_zero_any_number_of_neurons( count,
                                        partial,
                                        output_bool=qubo_vars[f'matrix_product_{k_layer}_{j}_{image_index}'],
                                        lam=Lambda,
                                        k_layer=k_layer,
                                        j=j,
                                        image_index=image_index,
                                       )

        fc_in = [ qubo_vars[f'matrix_product_{k_layer}_{j}_{image_index}'] for j in range( getattr(model, 'fc_list')[k_layer].weight.shape[0] ) ]
    #print(fc_in)
    partial = 0.
    count = 0.
    k_layer = 'loss'
    if num_classes == 2:
        
        H += (gt - fc_in[0])*(gt - fc_in[0])
        
        
    else:
        ## not implemented
        assert False
    image_index += 1
        




def evaluate_solution(qubo, x):
    result = 0
    for (i,j) in qubo:
        result += qubo[(i,j)]*x[i]*x[j]
    return result
#evaluate_solution(target_qubo[0], target_qubo_solution)


# 


def feed_forward_qubo_nn(H_solution, test_data):
    
    #H_solution needs to be a dictionary like:
    # {'partial_matrix_product_0_0_0_0': 1.0,
    # 'weight_0_0_0': 0.0,
    # 'weight_0_0_1': 1.0,
    # 'partial_matrix_product_0_1_0_0': 0.0,
    #  ...
    # 
    
    plot_images = False
    
    data = test_data
    #target = y_test_data
    
    test_images = data.shape[0]
    if num_classes == 2:
        output_net_size = 1
    else:
        output_net_size = num_classes

    input_layer_size = input_size
    output_layer_size = hidden_n
    w_list = []
    for layer in range(n_layers):
        if layer == n_layers-1:
            output_layer_size = output_net_size
        w_list.append(
                     np.array([[H_solution[f'weight_{layer}_{j}_{i}'] for i in range(input_layer_size)] for j in range(output_layer_size)])
                     )
        input_layer_size = hidden_n    
    for layer in range(n_layers):
        w_list[layer] *=2
        w_list[layer] -=1
    if plot_images:
        for layer in range(n_layers):
            plt.figure()
            plt.imshow(w_list[layer], cmap='gray')    
    out = []
    for index_i in range(test_images):
        in_x = data[index_i].flatten()
        in_x = (in_x > 0.5)*2 -1
        for layer in range(n_layers):
            in_x = w_list[layer] @ in_x      
            in_x = (np.array(in_x) > 0)*2 -1  ## binarize
        out.append( in_x )
    out = np.array(out)
    results = (out > 0.5)*1.
    results = results.reshape((test_data.shape[0]) )
    return results


# In[ ]:


# grb_train = []
def mycallback_time(model, where):
    if where == grb.GRB.Callback.MIP:
        _time = model.cbGet(grb.GRB.Callback.RUNTIME)
        best = model.cbGet(grb.GRB.Callback.MIP_OBJBST)
        cur_bd = model.cbGet(grb.GRB.Callback.MIP_OBJBND)

        grb_train.append((_time,best, cur_bd))
        if _time > grb_time and best < grb.GRB.INFINITY:
            model.terminate()
            print(_time, best, cur_bd)

def fix_zero(x):
    outx=[]
    for i in x:
        if i == 0:
            outx.append(-1)
        else:
            outx.append(i)
    return outx
def mycallback_plot(model, where):
    if where == grb.GRB.Callback.MIP:
        time = model.cbGet(grb.GRB.Callback.RUNTIME)
        best = model.cbGet(grb.GRB.Callback.MIP_OBJBST)
        cur_bd = model.cbGet(grb.GRB.Callback.MIP_OBJBND)
        time_ = int(time)
        global print_time_
        if print_time_ != time_:
            print(time, best, cur_bd)
            print_time_ = time_
        grb_train.append((time,best, cur_bd))
        if time > grb_time:
            model.terminate()
            
    if where == grb.GRB.Callback.MIPSOL:
        # MIP solution callback
        solution = model.cbGetSolution(model._Model__vars)
        #plt.figure()
        #plt.imshow(np.array([fix_zero(solution)]), cmap='gray')
print_time_ = -1


# In[ ]:


def set_up_grb_model_qubo(target_qubo_0, const=0):

    global grb_model
    global gurobi_vars
    global grb_loss
    grb_model = grb.Model()
    grb_model.setParam('OutputFlag', False)
    grb_model.setParam('Threads', 1)

    # First add the input variables as Gurobi variables.
    gurobi_vars = []
    v = grb_model.addVar( vtype=grb.GRB.BINARY, name=f'x')
    grb_model.update()
    gurobi_vars.append(v)


    grb_loss = 0
    grb_model.update()

    for (i,j) in target_qubo_0:
        v = grb_model.getVarByName(f"{i}")
        if v is None:
            v = grb_model.addVar( vtype=grb.GRB.BINARY, name=f'{i}')
            gurobi_vars.append(v)
        grb_model.update()
        w = grb_model.getVarByName(f"{j}")
        if w is None:
            w = grb_model.addVar( vtype=grb.GRB.BINARY, name=f'{j}')
            gurobi_vars.append(w)
        grb_model.update()
        grb_loss += v*w*target_qubo_0[(i,j)]
        grb_model.update()
    grb_loss += const
    grb_model.update()


# In[ ]:


def set_up_gurobi_model_hard_c(layers_to_optimize=[0,1]):
    global grb_model_hard_c
    global gurobi_vars_hard_c
    global grb_loss_hard_c
    global eps
    global layers_to_optim
    layers_to_optim = layers_to_optimize
    grb_model_hard_c = grb.Model()
    grb_model_hard_c.setParam('OutputFlag', False)
    grb_model_hard_c.setParam('Threads', 1)

    grb_loss_hard_c = 0
    
    # First add the input variables as Gurobi variables.
    gurobi_vars_hard_c = []
    #binary_vars = []
    for j_layer in layers_to_optimize:   ## second and third layer
        for i in range( np.prod( getattr(model, 'fc_list')[j_layer].weight.shape ) ):
            v = grb_model_hard_c.addVar( vtype=grb.GRB.BINARY,
                                          name=f'N_{j_layer}_{i}')
            c = grb_model_hard_c.addVar(vtype=grb.GRB.CONTINUOUS, lb=-1, name= f'spin_{j_layer}_{i}')
            grb_model_hard_c.addConstr( c == v*2 -1 )
            gurobi_vars_hard_c.append(c)

            grb_loss_hard_c += c*weights_biases_hard_c[f'spin_{j_layer}_{i}'] ## bias term for online training

    grb_model_hard_c.update()
    
    eps = 1e-5

def add_constrains_hard_c( input_x, gt , batch_index, approximate_constr=False):
    if approximate_constr != False:
        Lambda = approximate_constr
        
    target_index_i = [gt*2-1]

    num_classes = 1

    
    global gurobi_vars_hard_c
    layer_start = 0
    for k_layer_i in range(len(layers_to_optim)):
        k_layer = layers_to_optim[k_layer_i]
        if k_layer_i == 0:
            fc_in = (input_x>0)*2-1
        else:
            layer_start += np.prod(getattr(model, 'fc_list')[layers_to_optim[k_layer_i-1]].weight.shape)
            fc_in =  gurobi_vars_hard_c[-getattr(model, 'fc_list')[k_layer].weight.shape[1] : ]
        for i in range(  getattr(model, 'fc_list')[k_layer].weight.shape[0]  ):
            v = grb_model_hard_c.addVar( vtype=grb.GRB.BINARY,
                                          name=f'N_{batch_index}_{k_layer}_{i}')
            c = grb_model_hard_c.addVar(vtype=grb.GRB.CONTINUOUS, lb=-1, name= f'out_{batch_index}_{k_layer}_{i}')
            grb_model_hard_c.addConstr( c == v*2 -1 )
            gurobi_vars_hard_c.append(c)
        grb_model_hard_c.update()
        global grb_loss_hard_c
        ### loop over output dimension of layer
        for j in range(  getattr(model, 'fc_list')[k_layer].weight.shape[0]  ):
            partial_matrix_product = []
            ### loop over input dimension of layer
            for i in range( getattr(model, 'fc_list')[k_layer].weight.shape[1] ):   ## loop over input dimension of layer
                b = grb_model_hard_c.addVar(vtype=grb.GRB.CONTINUOUS, lb=-1, name= f'partial_matrix_product_{batch_index}_{k_layer}_{i}_{j}')
                grb_model_hard_c.update()
                grb_model_hard_c.addConstr( b == gurobi_vars_hard_c[layer_start +i+ getattr(model, 'fc_list')[k_layer].weight.shape[1]*j] * fc_in[i] )
                grb_model_hard_c.update()
                partial_matrix_product.append(b)
                
            grb_model_hard_c.addConstr( eps <=  (sum(partial_matrix_product) +0.5) * gurobi_vars_hard_c[-getattr(model, 'fc_list')[k_layer].weight.shape[0] + j] )
        grb_model_hard_c.update()

    check_vrb_i_abs = grb_model_hard_c.addVar(vtype=grb.GRB.CONTINUOUS, lb=0, name= f'loss_abs_{batch_index}')
    check_vrb_i = grb_model_hard_c.addVar(vtype=grb.GRB.CONTINUOUS, lb=-1, name= f'loss_{batch_index}')
    grb_model_hard_c.update()
    
    if target_index_i[0] != None:
        #print('############################ target_index_i[0] != None')
        grb_model_hard_c.addConstr( check_vrb_i ==  (gurobi_vars_hard_c[-1] - target_index_i[0] )/2.)
        grb_model_hard_c.update()
        grb_model_hard_c.addConstr( check_vrb_i_abs == grb.abs_( check_vrb_i ) )
        grb_model_hard_c.update()
        grb_loss_hard_c += check_vrb_i_abs
        grb_model_hard_c.update()


# In[ ]:


def set_up_weights_biases(layers_to_optimize=[1,2]):
    global weights_biases
    
    weights_biases = {} ## dict of variables
    for j_layer in layers_to_optimize:   ## second and third layer
        for i in range( getattr(model, 'fc_list')[j_layer].weight.shape[0] ):
            for k in range( getattr(model, 'fc_list')[j_layer].weight.shape[1] ):
                weights_biases[f'weight_{j_layer}_{i}_{k}'] = 0.

def set_up_weights_biases_hard_c(layers_to_optimize=[1,2]):
    global weights_biases_hard_c
    
    weights_biases_hard_c = {} ## dict of variables
    for j_layer in layers_to_optimize:   ## second and third layer
        for i in range( getattr(model, 'fc_list')[j_layer].weight.shape[0] ):
            for k in range( getattr(model, 'fc_list')[j_layer].weight.shape[1] ):
                #weights_biases_hard_c[f'spin_{j_layer}_{i*sum(select) + k}'] = 0.
                weights_biases_hard_c[f'spin_{j_layer}_{i*x_data_reduced.shape[1] + k}'] = 0.
                


# In[ ]:

        
def add_constraint_greater_zero_any_number_of_neurons( count, partial_poly, output_bool, lam, k_layer, j, image_index):
    global H
    ### this should be done in a general way
    
    if count < 2:
        delta = (2-count -count%2 )/2.
        aux = 1
        #aux_1 =  boolean_var(f'aux_matrix_product_{k_layer}_{j}_{image_index}_{aux}')
        aux_1 = output_bool
        H.add_constraint_eq_zero( -aux_1 + partial_poly, lam=lam)
    elif count < 4:
        delta = (4-count -count%2 )/2.
        aux = 1
        aux_1 =  boolean_var(f'aux_matrix_product_{k_layer}_{j}_{image_index}_{aux}')
        aux_2 = output_bool
        H.add_constraint_eq_zero( -aux_1 -2*aux_2 + partial_poly, lam=lam)
    elif count < 8:
        delta = (8-count -count%2 )/2.
        aux = 1
        aux_1 =  boolean_var(f'aux_matrix_product_{k_layer}_{j}_{image_index}_{aux}')
        aux = 2
        aux_2 =  boolean_var(f'aux_matrix_product_{k_layer}_{j}_{image_index}_{aux}')
        aux_4 = output_bool
        H.add_constraint_eq_zero( -aux_1 -2*aux_2 -4*aux_4 + partial_poly + delta, lam=lam)
    elif count < 16:
        delta = (16-count -count%2 )/2.
        aux = 1
        aux_1 =  boolean_var(f'aux_matrix_product_{k_layer}_{j}_{image_index}_{aux}')
        aux = 2
        aux_2 =  boolean_var(f'aux_matrix_product_{k_layer}_{j}_{image_index}_{aux}')
        aux = 4
        aux_4 =  boolean_var(f'aux_matrix_product_{k_layer}_{j}_{image_index}_{aux}')
        aux_8 = output_bool
        H.add_constraint_eq_zero( -aux_1 -2*aux_2 -4*aux_4 -8*aux_8 + partial_poly, lam=lam)
    elif count < 32:
        assert False
        

        
######    

def add_constraint_XNOR( s1, s2, z, lam, k_layer, j, image_index, input_dim_i):

    '''
    It enforces z = xnor(s1,s2)
    '''

    global H
    aux=0
    a0 =  boolean_var(f'aux_product_{k_layer}_{j}_{image_index}_{input_dim_i}_{aux}')
    aux=1
    a1 =  boolean_var(f'aux_product_{k_layer}_{j}_{image_index}_{input_dim_i}_{aux}')
    #H += (lam/2.)*( 4 -s1 +a0*(1-s1-s2-z) +a1*(1-s1+s2+z) +s2*z )
    H += (lam/2.)*( 4 -( 2*s1 -1 ) +(2*a0-1)*(1-(2*s1-1)-(2*s2-1)-(2*z-1)) +(2*a1-1)*(1-(2*s1-1)+(2*s2-1)+(2*z-1)) +(2*s2-1)*(2*z-1) )

    
# In[ ]:


#%%time
grb_train = []
def train_models(layers_to_optimize=[0,1]):
    global list_H_solutions
    global H_solution
    global weights_biases
    global weights_biases_hard_c
    global dwave_qubo
    global grb_dwave_qubo_solution
    global grb_model
    global grb_loss
    global grb_time
    global grb_train
    list_H_solutions = []
    global list_results
    list_results = []

    set_up_weights_biases(layers_to_optimize)
    set_up_weights_biases_hard_c(layers_to_optimize)

    for batch_n in range(num_batches_to_run):
    #for batch_n in range(int(np.floor(x_data_reduced.shape[0]/selected_to_train))):

        #print(batch_n)
        set_up_qubo_model(layers_to_optimize)
        set_up_gurobi_model_hard_c(layers_to_optimize) ### direct gurobi model with hard constraints
        for index_i in range(selected_to_train):
            in_x = x_data_reduced[batch_n*selected_to_train+index_i]
            in_x = 0+(in_x > 0.)
            add_constrains_qubo( in_x, y[batch_n*selected_to_train+index_i], Lambda=Lambda_qubo)
            add_constrains_hard_c( in_x, y[batch_n*selected_to_train+index_i], batch_index=index_i) ### direct gurobi model with hard constraints
        for key in weights_biases.keys():
            H[(key,)] += weights_biases[key] ## correct biases with running average of past solutions
            ## 

        #print(len(H.variables))
        #print(len(qubo_vars))                  

        # convert to qubo 
        H_qubo = H.to_qubo()
        dwave_qubo = H_qubo.Q   
        
        if 'embedded_qubo' in optimization_type:
            # find dwave embedding of qubo
            emb = find_embedding(dwave_qubo, G.edges, random_seed=10, threads = 8) 
            bqm = dimod.AdjVectorBQM(dwave_qubo, "BINARY")
            target_bqm = dwave.embedding.embed_bqm(bqm, emb, G.adj) 
            #print(target_bqm.num_variables)
            target_qubo = target_bqm.to_qubo()

        runtime = time.time()

        if optimization_type == 'grb_hard_c':
            ### direct gurobi model with hard constraints
            grb_model_hard_c.setObjective(grb_loss_hard_c, grb.GRB.MINIMIZE)
            grb_train = []
            grb_time=np.inf
            #grb_model_hard_c.optimize(mycallback_time)
            grb_model_hard_c.optimize(mycallback_plot)
            
            assert grb_model_hard_c.status != 3 ## check that the model is not unfeasible
            
            #plt.figure()
            #plt.plot([x_[0] for x_ in grb_train][1:],[x_[1] for x_ in grb_train][1:], '-')
            #plt.plot([x_[0] for x_ in grb_train][10:],[x_[2] for x_ in grb_train][10:], '-')
            H_solution={}
            for ii in grb_model_hard_c.getVars():
                if 'spin' in ii.VarName:
                    #print(ii.VarName)
                    ii.VarName.split('_')
                    layer_i = int(ii.VarName.split('_')[1])
                    if layer_i == 0:
                        _x = int(ii.VarName.split('_')[2]) % x_data_reduced.shape[1] ##sum(select) 
                        _y = int(ii.VarName.split('_')[2]) // x_data_reduced.shape[1] ## sum(select)
                    else:
                        _x = int(ii.VarName.split('_')[2]) % hidden_n  
                        _y = int(ii.VarName.split('_')[2]) // hidden_n
                        
                    H_solution[f'weight_{layer_i}_{_y}_{_x}'] = (ii.x+1)/2.
                    weights_biases_hard_c[ii.VarName] += -alpha* ii.x

        elif optimization_type == 'sym_anneal_soft_qubo':
            # simulated annealing on pubo:
            res = anneal_pubo(H, num_anneals=sym_anneal_num_anneals)
            H_solution = res.best.state
            print("Model value:", res.best.value)
            print("Constraints satisfied?", H.is_solution_valid(H_solution))
            
        elif optimization_type == 'grb_soft_qubo':
            ## gurobi optimization
            set_up_grb_model_qubo(dwave_qubo, const=H_qubo[()]) ## set up gurobi model on qubo
            
            grb_model.setObjective(grb_loss, grb.GRB.MINIMIZE)
            grb_train = []
            grb_time=np.inf
            #grb_model.optimize(mycallback_time)
            grb_model.optimize(mycallback_plot)
            #plt.figure()
            #plt.plot([x[0] for x in grb_train][:],[x[1] for x in grb_train][:], '-')
            #plt.plot([x[0] for x in grb_train][:],[x[2] for x in grb_train][:], '-')
            #plt.axes().set_yscale('log')
            grb_model.getVars()
            grb_dwave_qubo_solution={}
            for i in grb_model.getVars():
                if i.VarName != 'x':
                    grb_dwave_qubo_solution[int(i.VarName)] = i.x
            H_solution = H.convert_solution(grb_dwave_qubo_solution)
            print("Constraints satisfied?", H.is_solution_valid(H_solution))


        #else:
        #    assert False

        if optimization_type == 'grb_soft_embedded_qubo':
            ## gurobi optimization
            set_up_grb_model_qubo(target_qubo[0]) ## set up gurobi model on qubo
            grb_model.setObjective(grb_loss, grb.GRB.MINIMIZE)
            grb_train = []
            grb_time=np.inf
            grb_model.optimize(mycallback_plot)
            #plt.figure()
            #plt.plot([x[0] for x in grb_train][:],[x[1] for x in grb_train][:], '-')
            #plt.plot([x[0] for x in grb_train][:],[x[2] for x in grb_train][:], '-')
            #plt.axes().set_yscale('log')
            grb_model.getVars()
            grb_target_qubo_solution={}
            for i in grb_model.getVars():
                if i.VarName != 'x':
                    grb_target_qubo_solution[int(i.VarName)] = i.x
            unembedded_results_from_grb_target_qubo_solution = {}
            for i in range(bqm.num_variables):
                _results_ = []
                for j in emb[i]:
                    _results_.append(grb_target_qubo_solution[j])
                unembedded_results_from_grb_target_qubo_solution[i] = np.mean(_results_)
                assert  np.mean(_results_) == 0 or np.mean(_results_) == 1
                #unembedded_results_from_target_qubo_solution[i] = np.random.bit_generator.randbits(1)
            #unembedded_results_from_target_qubo_solution
            H_solution = H.convert_solution(unembedded_results_from_grb_target_qubo_solution)

        if optimization_type == 'sym_anneal_soft_embedded_qubo':
            # simulated annealing on embedded qubo
            target_qubo_res = anneal_qubo(target_qubo[0], num_anneals=sym_anneal_num_anneals)
            target_qubo_solution = target_qubo_res.best.state
            print("Model value:", target_qubo_res.best.value)
            unembedded_results_from_target_qubo_solution = {}
            for i in range(bqm.num_variables):
                _results_ = []
                for j in emb[i]:
                    _results_.append(target_qubo_solution[j])
                unembedded_results_from_target_qubo_solution[i] = np.mean(_results_)
                #unembedded_results_from_target_qubo_solution[i] = np.random.bit_generator.randbits(1)
            H_solution = H.convert_solution(unembedded_results_from_target_qubo_solution)
            #unembedded_results_from_target_qubo_solution

        qpu_response = 0
        if optimization_type == 'quantum_anneal_soft_embedded_qubo':

            ## QPU sampler
            n_reads_qpu = 100
            n_reads = n_reads_qpu
            response = solver.sample_bqm(target_bqm, num_reads=n_reads) #, chain_strength=2*L) 
            response_result = response.result()
            unembedded_results = {}
            for i in range(bqm.num_variables):
                _results_ = []
                for j in emb[i]:
                    _results_.append(response_result['solutions'][0][j])
                unembedded_results[i] = np.mean(_results_)
            #unembedded_results
            H_solution = H.convert_solution(unembedded_results)
            
            qpu_response = [ dict(response_result), emb]

        runtime = time.time() - runtime

        for key in weights_biases.keys():
            weights_biases[key] +=  -alpha*(2*H_solution[key]-1)
        list_H_solutions.append(H_solution)
        
        #print((feed_forward_qubo_nn( H_solution, x_data_reduced[batch_n*selected_to_train:(batch_n+1)*selected_to_train] ) > 0.5) == y[batch_n*selected_to_train:(batch_n+1)*selected_to_train])
        #print(feed_forward_qubo_nn( H_solution, x_data_reduced[batch_n*selected_to_train:(batch_n+1)*selected_to_train] ))
        #print(y[batch_n*selected_to_train:(batch_n+1)*selected_to_train])
        wrong_on_train = (sum((feed_forward_qubo_nn( H_solution, x_data_reduced[batch_n*selected_to_train:(batch_n+1)*selected_to_train] ) > 0.5) != y[batch_n*selected_to_train:(batch_n+1)*selected_to_train]))
        #print(wrong_on_train)
        list_results.append( ( batch_n, wrong_on_train, runtime, qpu_response )    )


# In[ ]:


def create_plots():
    plt.figure()
    plt.title('time'+' '+optimization_type)
    plt.hist( [ list_results[i][2]-list_results_grb_hard_c[i][2]     for i in range(num_batches_to_run)])
    plt.figure()
    plt.title('distance from optimum'+' '+optimization_type)
    plt.hist( [ list_results[i][1]-list_results_grb_hard_c[i][1]     for i in range(num_batches_to_run)])
    plt.figure()
    plt.title('distance from optimum vs time'+' '+optimization_type)
    plt.plot( [ list_results[i][2]-list_results_grb_hard_c[i][2]     for i in range(num_batches_to_run)],
              [ list_results[i][1]-list_results_grb_hard_c[i][1]     for i in range(num_batches_to_run)],
              '.'
            )


