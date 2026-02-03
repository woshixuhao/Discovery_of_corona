'''
graph regression for any expression with unknown form and coefficient---More variables
formal version
by HaoXu
'''

import os
import math
import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy
from torch_geometric.data import Data
import torch
from scipy.optimize import minimize,curve_fit
from tqdm import tqdm
import time
import heapq
from sympy import symbols, sin, cos, tan, log, ln, sqrt, exp, csc, sec, cot, sinh, tanh, cosh, atan, asin, acos, atanh, \
    asinh, acosh, sympify,pi, lambdify,E,I
from copy import deepcopy
import warnings
import sympy as sp
import pickle
from concurrent.futures import ProcessPoolExecutor
import utils
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore", category=RuntimeWarning)
NODE_FEATURE_MAP = {
    "1":1,
    'add': 2,
    'mul': 3,
    'exp': 4,
    'div': 5,
    "log": 6,
    "ln": 7,
    "sqrt": 8,
    "abs": 9,
    "sub": 10,
    "sin": 11,
    "cos": 12,
    "tan":13,
    "sinh":14,
    "tanh":15,
    "cosh":16,
    "atan":17,
    "asin":18,
    "acos":19,
    "atanh":20,
    "asinh":21,
    "acosh":22,
    'x': 23,  # Variable
    'y': 24,
    'z':25,
    'E':26,
    'pi':27,
    'x1':28, 'x2':29,'x3':30
}
Binary_Operator=['add','mul']
Unary_Operator_ln=["log"]
Unary_Operator_exp=['exp']
Triangle_Operator=["sin", "cos","tan"]
Hyperbolic_Operator=["sinh", "tanh","cosh"]
Variables=['x1','x2','x3']
Constant=['1','pi','E']
x1, x2,x3,x4,C ,C1,C2,C3,A,B = symbols('x1 x2 x3 x4 C C1 C2 C3 A B')


def set_random_seeds(rand_seed=525, np_rand_seed=314):
    random.seed(rand_seed)
    np.random.seed(np_rand_seed)

set_random_seeds()

def convert_graph_to_pyG(graph):
    graph_nodes = graph['nodes']
    graph_edges = graph['edges']
    graph_edge_attr = graph['edge_attr']
    x=[NODE_FEATURE_MAP[node] for node in graph_nodes]
    x=torch.from_numpy(np.array(x).astype(np.int64))
    edge_index=torch.from_numpy(np.array(graph_edges).astype(np.int64)).T
    edge_attr=torch.from_numpy(np.array(graph_edge_attr).astype(np.float32))
    pyG_graph=Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return pyG_graph



class Random_graph_for_expr():
    def __init__(self):
        super().__init__()  # 调用父类的构造函数
        self.max_add_term = 5
    def concate_subgraph_to_node(self, graph, subgraph, concate_node_index, concate_node,set_maximum_node_num=-1,with_node_indice=[]):
        '''
        :param graph: a tuple of [nodes, edges, edge_attr] for the existing graph
        :param concate_node: concate the graph to which node
        :param subgraph: a tuple of [nodes, edges, edge_attr] for the generated subgraph
        :return: new graph
        '''
        if set_maximum_node_num==-1:
            maximun_node_num = len(graph['nodes'])
        else:
            maximun_node_num =set_maximum_node_num
        subgraph_nodes = subgraph['nodes']
        subgraph_edges = subgraph['edges']
        subgraph_edge_attr = subgraph['edge_attr']

        graph_nodes = graph['nodes']
        graph_edges = graph['edges']
        graph_edge_attr = graph['edge_attr']

        for sublist in subgraph_edges:
            for i in range(len(sublist)):
                sublist[i] += maximun_node_num
        graph_nodes += subgraph_nodes
        graph_edges.append([concate_node_index, maximun_node_num])
        graph_edges += subgraph_edges
        if concate_node =='exp':
            graph_edge_attr.append(random.choices([-2,-1,-0.5,0.5,2],[0.1,0.25,0.1,0.1,0.1],k=1)[0]) #no 1
        elif concate_node=='add':
            graph_edge_attr.append(random.choices([1,-1],[0.45,0.45],k=1)[0])
        elif concate_node=='mul':
            graph_edge_attr.append(random.choices([1, -1], [0.3, 0.3], k=1)[0])
        elif concate_node=='E_exp':
            graph_edge_attr.append(random.choices([1, -1], [0.3, 0.3], k=1)[0])
        else:
            graph_edge_attr.append(1)
        graph_edge_attr += subgraph_edge_attr
        graph = {'nodes': graph_nodes, 'edges': graph_edges, 'edge_attr': graph_edge_attr}

        if len(with_node_indice)==0:
            return graph
        else:
            with_node_indice=[maximun_node_num+i for i in range(len(subgraph_nodes))]
            return graph,with_node_indice

    def generate_single_vars(self):
        select_var=random.choice(Variables)
        subgraph={'nodes': [select_var], 'edges': [], 'edge_attr': []}
        return subgraph

    def generate_log_template(self):
            '''
               The function to generate subgraph for the node 'log' and 'ln'
               define Template:
               log(A/B)
               log(A+B)
               log(A*B)
               :return:
               '''

            Template = ['log(A)', 'ln(A)']
            use_template = random.choice(Template)
            if use_template == 'log(A)':
                initial_graph = {'nodes': ['log'], 'edges': [], 'edge_attr': []}
                A_graph = self.generate_single_vars()
                graph = self.concate_subgraph_to_node(initial_graph, A_graph, 0, 'log')
                graph['edge_attr'][0]=10

            elif use_template == 'ln(A)':
                initial_graph = {'nodes': ['log'], 'edges': [], 'edge_attr': []}
                A_graph = self.generate_single_vars()
                graph = self.concate_subgraph_to_node(initial_graph, A_graph, 0, 'log')
                graph['edge_attr'][0] = math.e

            return graph

    def generate_multiple_vars(self,max_multiple_term=3):
        choices = list(range(1, max_multiple_term + 1))
        weights = [1.0 / k for k in choices]
        n_terms = random.choices(choices, weights=weights, k=1)[0]
        add_log=random.choices([True, False], weights=[0.2,1], k=1)[0]

        if n_terms == 1:
            subgraph = self.generate_single_vars()
        else:
            subgraph = {'nodes': ['mul'], 'edges': [], 'edge_attr': []}
            if add_log == True:
                for i in range(n_terms-1):
                    subgraph = self.concate_subgraph_to_node(subgraph, self.generate_single_vars(), 0, 'mul')
                log_template = self.generate_log_template()
                subgraph = self.concate_subgraph_to_node(subgraph,log_template,0, 'mul')
            if add_log == False:
                for i in range(n_terms):
                    subgraph = self.concate_subgraph_to_node(subgraph, self.generate_single_vars(), 0, 'mul')


        return subgraph

    def generate_rational_template(self,max_additive_term=2):
        '''
        A/B
        :return:
        '''


        choices = list(range(1, max_additive_term + 1))
        weights = [1.0 / k for k in choices]
        n_terms = random.choices(choices, weights=weights, k=1)[0]
        B_graph = {'nodes': ['add'], 'edges': [], 'edge_attr': []}
        for i in range(n_terms):
            B_graph = self.concate_subgraph_to_node(B_graph, self.generate_multiple_vars(), 0, 'mul')


        is_A_constant = random.choices([True,False], weights=[0.4,1], k=1)[0]
        if is_A_constant:
            A_graph =  {'nodes': ['1'], 'edges': [], 'edge_attr': []}
        else:
            A_graph=self.generate_multiple_vars()

        inv_B = {'nodes': ['exp'], 'edges': [], 'edge_attr': []}
        inv_B= self.concate_subgraph_to_node( inv_B, B_graph, 0, 'exp')
        inv_B['edge_attr'][0] = -1

        subgraph = {'nodes': ['mul'], 'edges': [], 'edge_attr': []}
        subgraph = self.concate_subgraph_to_node(subgraph, A_graph, 0, 'mul')
        subgraph = self.concate_subgraph_to_node(subgraph, inv_B, 0, 'mul')
        return subgraph
    def generate_graph_template(self,max_add_term=5):
        n_terms = random.randint(1,max_add_term)
        template=['poly','rational','log']
        graph = {'nodes': ['add'], 'edges': [], 'edge_attr': []}
        add_1 = random.choices([True, False], weights=[0.2, 1], k=1)[0]
        for i in range(n_terms):
            select_template = random.choices(template, weights=[1, 0.5,0.5])[0]
            if select_template=='poly':
                 graph = self.concate_subgraph_to_node(graph, self.generate_multiple_vars(), 0, 'add')
            elif select_template=='rational':
                graph = self.concate_subgraph_to_node(graph, self.generate_rational_template(), 0, 'add')
            elif select_template == 'log':
                graph = self.concate_subgraph_to_node(graph, self.generate_log_template(), 0, 'add')
        if add_1==True and n_terms<max_add_term:
            graph = self.concate_subgraph_to_node(graph, {'nodes': ['1'], 'edges': [], 'edge_attr': []} , 0, 'add')
        return graph

    def generate_random_graph(self,max_add_term=5):
        graph=self.generate_graph_template(max_add_term=max_add_term)
        return graph

class Graph_to_sympy():

    def get_nodes_to_subgraphs(self,graph):
        operator_map = {
            'add': lambda a, b: a + b,
            'mul': lambda a, b: a * b,
            'div': lambda a, b: a / b,
            'log': lambda a: log(a, 10),
            'ln': lambda a: ln(a),
            'sqrt': lambda a: sqrt(a),
            'exp': lambda a: a,  # exp(x) -> x^C, where C is an unknown coefficient
            'sin': lambda a: sin(a),
            'cos': lambda a: cos(a),
            'tan': lambda a: tan(a),
            'sinh': lambda a: sinh(a),
            'tanh': lambda a: tanh(a),
            'cosh': lambda a: cosh(a),
            'atan': lambda a: atan(a),
            'asin': lambda a: asin(a),
            'acos': lambda a: acos(a),
            'atanh': lambda a: atanh(a),
            'asinh': lambda a: asinh(a),
            'acosh': lambda a: acosh(a),
            'E_exp': lambda a: E ** a
        }

        nodes, edges, edge_attr = graph['nodes'], graph['edges'], graph['edge_attr']
        # Create a dictionary to hold the expressions for each node
        expressions = {}
        visited = set()  # To track nodes that are currently being processed
        symbol_index = 0  # To track how many unknown coefficients we've generated

        # Assuming new symbols are generated for edge_attr = -1
        unknown_symbol_dict = {i: symbols(f'C{i}') for i in range(len(edges) + 5)}  #

        # Function to evaluate the expression for a given node
        def evaluate_node(node_index):
            # If the node has already been evaluated, return its expression
            if node_index in expressions:
                return expressions[node_index]

            # If the node is currently being processed, we've detected a cycle
            if node_index in visited:
                raise RecursionError(f"Cyclic dependency detected at node {node_index}")

            # Mark the current node as being processed
            visited.add(node_index)

            # Get the current node's operation or variable
            node_value = nodes[node_index]
            # If the node is a constant (e.g., pi or 1)
            if node_value == 'pi':
                expressions[node_index] = pi
            elif node_value == 'x1':
                expressions[node_index] = x1
            elif node_value == 'x2':
                expressions[node_index] = x2
            elif node_value == 'x3':
                expressions[node_index] = x3
            elif node_value == 'x4':
                expressions[node_index] = x4
            elif node_value == '1':
                expressions[node_index] = 1
            # If the node is a unary operator
            elif node_value in Unary_Operator_ln + Unary_Operator_exp + Triangle_Operator + Hyperbolic_Operator + [
                'E_exp']:
                # For unary operators, only one edge should point to it
                child_node_index = [edges[i][1] for i in range(len(edges)) if edges[i][0] == node_index][
                    0]  # Find the parent node
                edge_index = [i for i in range(len(edges)) if edges[i][0] == node_index][0]
                if node_value == 'log':
                    if evaluate_node(child_node_index) in [0, -1]:
                        expressions[node_index] = 1
                    elif (type(evaluate_node(child_node_index)).__name__.lower() in ['int', 'float', 'integer',
                                                                                     'rational']) and (
                            float(evaluate_node(child_node_index)) < 0):
                        expressions[node_index] = 1
                    else:
                        if abs(edge_attr[edge_index] - math.e) < 1e-5:
                            expressions[node_index] = operator_map['ln'](evaluate_node(child_node_index))
                        else:
                            expressions[node_index] = operator_map[node_value](evaluate_node(child_node_index))

                elif node_value in Triangle_Operator + Hyperbolic_Operator:
                    if edge_attr[edge_index] == -1e8:
                        expressions[node_index] = operator_map[node_value](
                            unknown_symbol_dict[edge_index] * evaluate_node(child_node_index))
                    else:
                        expressions[node_index] = operator_map[node_value](
                            edge_attr[edge_index] * evaluate_node(child_node_index))
                else:
                    expressions[node_index] = operator_map[node_value](evaluate_node(child_node_index))
            # If the node is a binary operator (add, mul)
            elif node_value in Binary_Operator:
                # Collect the child nodes of this node
                child_nodes = [edges[i][1] for i in range(len(edges)) if edges[i][0] == node_index]
                child_egdes = [i for i in range(len(edges)) if edges[i][0] == node_index]

                # Apply the appropriate operation (add, mul, div)
                if node_value == 'add':
                    addition = 0
                    for iter, child in enumerate(zip(child_nodes, child_egdes)):
                        if edge_attr[child[1]] == -1e8:
                            addition += evaluate_node(child[0]) * unknown_symbol_dict[child[1]]
                        else:
                            addition += evaluate_node(child[0]) * edge_attr[child[1]]
                    expressions[node_index] = addition
                elif node_value == 'mul':
                    product = 1
                    unknown_flag = 0
                    for iter, child in enumerate(zip(child_nodes, child_egdes)):
                        if edge_attr[child[1]] == -1e8:
                            unknown_flag += 1
                            product *= (evaluate_node(child[0]))
                        else:
                            product *= (evaluate_node(child[0]) * edge_attr[child[1]])
                    if unknown_flag == 0:
                        expressions[node_index] = product
                    else:
                        expressions[node_index] = product * unknown_symbol_dict[child[1]]

            # If the edge_attr is -1, we introduce an unknown coefficient (C)
            if nodes[node_index] == 'exp':  # For non-exponentiation nodes
                child_egdes = [i for i in range(len(edges)) if edges[i][0] == node_index]

                if edge_attr[child_egdes[0]] != -1e8:
                    if expressions[node_index] == 0 and edge_attr[child_egdes[0]] <= 0:
                        expressions[node_index] = 1
                    elif expressions[node_index] == -1 and edge_attr[child_egdes[0]] in [0.5, -0.5]:
                        expressions[node_index] = 1
                    elif type(expressions[node_index]).__name__.lower() in ['int', 'float', 'integer',
                                                                            'rational'] and float(
                            evaluate_node(child_node_index)) < 0 and edge_attr[child_egdes[0]] in [0.5, -0.5]:
                        expressions[node_index] = 1
                    else:
                        expressions[node_index] = expressions[node_index] ** edge_attr[child_egdes[0]]
                else:
                    if expressions[node_index] == 0:
                        expressions[node_index] = 1
                    else:
                        expressions[node_index] = expressions[node_index] ** unknown_symbol_dict[child_egdes[0]]

            # Remove the node from the visited set as we're done processing it
            visited.remove(node_index)

            # Return the evaluated expression for the node
            return expressions[node_index]

        # Start the evaluation from the root nodes (ones that do not have incoming edges)
        for node_index in range(len(nodes)):
            if node_index not in [edge[1] for edge in edges]:
                evaluate_node(node_index)

        if type(expressions[0]).__name__.lower() not in ['int', 'float', 'integer', 'rational']:
            if expressions[0].has(I) == True:
                expressions[0] = 1
        # The root node will be the last evaluated expression
        return expressions
    def graph_to_sympy(self,graph):
        expressions=self.get_nodes_to_subgraphs(graph)
        # The root node will be the last evaluated expression
        return expressions[0]#expressions[0]*A+B

        # Example input (nodes, edges, edge_attr as described)

class Genetic_algorithm(Random_graph_for_expr,Graph_to_sympy):
    def __init__(self,x_data,y_data):
        super().__init__()
        self.x_data=x_data
        self.y_data=y_data
        self.size_pop=300
        self.generation_num=100
        self.distinction_epoch=5
        self.max_terms=5
        self.use_parallel_computing=False
        self.seek_best_initial=True
        self.use_monotonicity=True
        self.epi=0.2
        self.mono_penalty=0.1


    def renumber_subgraph(self,graph,node_indice):
        """
        Renumber the nodes of a subgraph starting from 0, and adjust the edges accordingly.

        Args:
        nodes (list): A list of nodes in the subgraph.
        edges (list): A list of edges, where each edge is represented by a pair of node indices.

        Returns:
        tuple: A tuple containing:
            - The renumbered nodes (list)
            - The renumbered edges (list)
        """
        nodes=node_indice
        edges=graph['edges']
        # Step 1: Create a mapping from the original nodes to the new renumbered nodes.
        node_mapping = {node: i for i, node in enumerate(sorted(nodes))}
        # Step 2: Renumber the nodes in the edges based on the new node_mapping
        renumbered_edges = [[node_mapping[edge[0]], node_mapping[edge[1]]] for edge in edges]

        # Step 3: Return the renumbered nodes (sorted and starting from 0) and edges.
        renumbered_nodes = list(node_mapping.values())  # New nodes are just the renumbered indices

        graph['edges']=renumbered_edges
        return graph

    def extract_subgraph(self,graph, root):
        nodes, edges, edge_attr = graph['nodes'], graph['edges'], graph['edge_attr']
        subgraph_nodes = []
        subgraph_edges = []
        subgraph_edge_attr = []

        queue = [root]
        visited = set()

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            subgraph_nodes.append(current)

            for edge, attr in zip(edges, edge_attr):
                if edge[0] == current:
                    subgraph_edges.append(edge)
                    subgraph_edge_attr.append(attr)
                    queue.append(edge[1])
        subgraph_nodes=sorted(subgraph_nodes)
        return subgraph_nodes, subgraph_edges, subgraph_edge_attr

    def delete_subgraph_from_node(self, graph, node_index):
        subgraph_nodes, subgraph_edges, subgraph_edge_attr = self.extract_subgraph(graph, node_index)
        graph_node_index = [i for i in range(len(graph['nodes']))]
        graph['nodes'] = [element for index, element in enumerate(graph['nodes']) if index not in subgraph_nodes]
        graph_node_index = [element for index, element in enumerate(graph_node_index) if index not in subgraph_nodes]
        new_edges = []
        new_edge_attr = []
        for index, edge_info in enumerate(zip(graph['edges'], graph['edge_attr'])):
            if (edge_info[0] not in subgraph_edges) and (edge_info[0][1] not in subgraph_nodes):
                new_edges.append(edge_info[0])
                new_edge_attr.append(edge_info[1])
        graph['edges'] = new_edges
        graph['edge_attr'] = new_edge_attr
        return graph,graph_node_index

    def cross_over(self,graph1, graph2):
        """
        Perform crossover by exchanging subgraphs between two graphs.
        """
        graph1 = deepcopy(graph1)
        graph2 = deepcopy(graph2)

        selected_nodes_index_1=[graph1['edges'][i][1] for i in range(len(graph1['edges'])) if graph1['edges'][i][0] == 0]
        selected_nodes_index_2 = [graph2['edges'][i][1] for i in range(len(graph2['edges'])) if
                                  graph2['edges'][i][0] == 0]

        #print('original graphs','\n',self.graph_to_sympy(graph1),'\n',self.graph_to_sympy(graph2))

        # Select a random node in graph1 as the root of the subgraph to replace
        node1 = random.choice(selected_nodes_index_1)
        max_node1_num=len(graph1['nodes'])
        parent_node_index= [graph1['edges'][i][0] for i in range(len(graph1['edges'])) if graph1['edges'][i][1] == node1][0]
        parent_node_attr= [graph1['edge_attr'][i] for i in range(len(graph1['edges'])) if graph1['edges'][i][1] == node1][0]
        # Select a random node in graph2 as the root of the subgraph to use
        node2 = random.choice(selected_nodes_index_2)
        sub_nodes_2, sub_edges_2, sub_edge_attr_2 = self.extract_subgraph(graph2, node2)
        sub_nodes_1, sub_edges_1, sub_edge_attr_1 = self.extract_subgraph(graph1, node1)

        #Delete the subgraph
        graph1_node_index = [i for i in range(len(graph1['nodes']))]
        graph1['nodes'] = [element for index, element in enumerate(graph1['nodes']) if index not in sub_nodes_1]
        graph1_node_index=[element for index, element in enumerate(graph1_node_index) if index not in sub_nodes_1]
        new_edges_1=[]
        new_edge_attr_1=[]
        for index,edge_info in enumerate(zip(graph1['edges'],graph1['edge_attr'])):
            if edge_info[0] not in sub_edges_1:
                if edge_info[0][1] not in sub_nodes_1:
                    new_edges_1.append(edge_info[0])
                    new_edge_attr_1.append(edge_info[1])
        graph1['edges']=new_edges_1
        graph1['edge_attr']=new_edge_attr_1
        # print(graph1['nodes'], graph1['edges'])
        # print(sub_nodes_2, sub_edges_2, sub_edge_attr_2)
        #print('cross_over nodes:',node1)
        if node1==0:
            graph1_nodes=sub_nodes_2
            graph1['nodes']=[graph2['nodes'][i] for i in graph1_nodes]
            graph1['edges']=sub_edges_2
            graph1['edge_attr']=sub_edge_attr_2
            graph1=self.renumber_subgraph(graph1,graph1_nodes)

        else:

            if len(sub_edges_2)!=0:
                sub_edges_2=(np.array(sub_edges_2)+max_node1_num).tolist()
                nodes2_min=np.min(np.array(sub_edges_2))
                graph1['edges'].append([parent_node_index,nodes2_min])
                graph1_node_index.extend((np.array(sub_nodes_2) + max_node1_num).tolist())
            else:
                graph1['edges'].append([parent_node_index, parent_node_index+max_node1_num])
                graph1_node_index.extend([parent_node_index+max_node1_num])

            graph1['edge_attr'].append(parent_node_attr)

            graph1['nodes'].extend([graph2['nodes'][i] for i in sub_nodes_2])
            graph1['edges'].extend(sub_edges_2)
            graph1['edge_attr'].extend(sub_edge_attr_2)

            #print(graph1_node_index,graph1['edges'])
            graph1=self.renumber_subgraph(graph1,graph1_node_index)
        #print('after cross over:',graph1)
        # pyG_graph = convert_graph_to_pyG(graph1)
        # plot_graph_with_features(pyG_graph)
        # plt.show()

        return graph1

    def mutate(self,graph, node_mutation_rate=0.2,graph_mutation_rate=0.5,graph_delete_graph=0.2,graph_add_graph=0.2,mutate_edge_attr_prob=0.5):
        """
        Perform mutation by modifying nodes and edges randomly.
        """
        graph = deepcopy(graph)
        num_nodes = len(graph['nodes'])
        edges=graph['edges']

        # Mutate nodes
        for i in range(num_nodes):
            if random.random() < node_mutation_rate:
                if graph['nodes'][i]=='log':
                    edge_index=[j for j in range(len(edges)) if edges[j][0] == i][0]
                    graph['edge_attr'][edge_index]= random.choice([10,math.e])
                if graph['nodes'][i] in Triangle_Operator:
                    graph['nodes'][i]=random.choice(Triangle_Operator)
                if graph['nodes'][i] in Hyperbolic_Operator:
                    graph['nodes'][i] = random.choice(Hyperbolic_Operator)
                if graph['nodes'][i] in Variables:
                    graph['nodes'][i] = random.choice(Variables)

        # Mutate subgraphs--search more complex forms:
        if (random.random()<graph_mutation_rate) and (len(graph['nodes']))>1:
            selected_nodes_index = [graph['edges'][i][1] for i in range(len(graph['edges'])) if
                                      graph['edges'][i][0] == 0]
            mutate_node = random.choice(selected_nodes_index)
            parent_node=0
            graph, node_indices = self.delete_subgraph_from_node(graph, mutate_node)

            template = ['poly', 'rational','log']
            select_template = random.choices(template, weights=[1,0.5,0.5])[0]
            if select_template == 'poly':
                graph, concate_node_indices = self.concate_subgraph_to_node(graph, self.generate_multiple_vars(),
                                                                            parent_node,
                                                                            graph['nodes'][parent_node],
                                                                            set_maximum_node_num=num_nodes,
                                                                            with_node_indice=node_indices)
            elif select_template == 'rational':
                graph, concate_node_indices = self.concate_subgraph_to_node(graph, self.generate_rational_template(),
                                                                            parent_node,
                                                                            graph['nodes'][parent_node],
                                                                            set_maximum_node_num=num_nodes,
                                                                            with_node_indice=node_indices)
            elif select_template == 'log':
                graph, concate_node_indices = self.concate_subgraph_to_node(graph, self.generate_log_template(),
                                                                            parent_node,
                                                                            graph['nodes'][parent_node],
                                                                            set_maximum_node_num=num_nodes,
                                                                            with_node_indice=node_indices)
            node_indices = node_indices + concate_node_indices
            graph = self.renumber_subgraph(graph, node_indices)



        #delete edges
        if random.random()<graph_delete_graph:
            #print('delete edges')
            if len(graph['nodes'])>1:
                mutate_node= random.randint(1, len(graph['nodes']) - 1)
                parent_node_index = \
                [graph['edges'][i][0] for i in range(len(graph['edges'])) if graph['edges'][i][1] == mutate_node][0]
                parent_node_attr = \
                [graph['edge_attr'][i] for i in range(len(graph['edges'])) if graph['edges'][i][1] == mutate_node][0]
                mutate_node_value=graph['nodes'][mutate_node]
                max_node_num=len(graph['nodes'])
                subgraph_nodes, subgraph_edges, subgraph_edge_attr=self.extract_subgraph(graph,mutate_node)
                graph_node_index = [i for i in range(len(graph['nodes']))]
                graph['nodes'] = [element for index, element in enumerate(graph['nodes']) if index not in subgraph_nodes]
                graph_node_index = [element for index, element in enumerate(graph_node_index) if index not in subgraph_nodes]
                new_edges_1 = []
                new_edge_attr_1 = []
                for index, edge_info in enumerate(zip(graph['edges'], graph['edge_attr'])):
                    if edge_info[0] not in subgraph_edges:
                        if edge_info[0][1] not in subgraph_nodes:
                            new_edges_1.append(edge_info[0])
                            new_edge_attr_1.append(edge_info[1])
                graph['edges'] = new_edges_1
                graph['edge_attr'] = new_edge_attr_1
                graph['nodes'].append('1')
                graph_node_index.append(max_node_num+1)
                graph['edges'].append([parent_node_index,max_node_num+1])
                graph['edge_attr'].append(1)
                graph=self.renumber_subgraph(graph,graph_node_index)

        #add edges
        if random.random() < graph_add_graph:
            template = ['poly', 'rational', 'log']
            select_template = random.choices(template, weights=[1, 0.5,0.5])[0]
            mutate_node=0
            num_nodes = len(graph['nodes'])
            node_indices = [i for i in range(len(graph['nodes']))]
            if select_template == 'poly':
                graph, concate_node_indices = self.concate_subgraph_to_node(graph,
                                                                            self.generate_multiple_vars(),
                                                                            mutate_node,
                                                                            graph['nodes'][mutate_node],
                                                                            set_maximum_node_num=num_nodes,
                                                                            with_node_indice=node_indices)
            elif select_template ==  'rational':
                graph, concate_node_indices = self.concate_subgraph_to_node(graph,
                                                                            self.generate_rational_template(),
                                                                            mutate_node,
                                                                            graph['nodes'][mutate_node],
                                                                            set_maximum_node_num=num_nodes,
                                                                            with_node_indice=node_indices)
            elif select_template ==  'log':
                graph, concate_node_indices = self.concate_subgraph_to_node(graph,
                                                                            self.generate_log_template(),
                                                                            mutate_node,
                                                                            graph['nodes'][mutate_node],
                                                                            set_maximum_node_num=num_nodes,
                                                                            with_node_indice=node_indices)



        #mutate edge_attr
        for mutate_edge_attr_index in range(len(graph['edge_attr'])):
            if random.random()<mutate_edge_attr_prob:
                #print('mutate edge_attr')
                mutate_edge=graph['edges'][mutate_edge_attr_index]
                begin_node=graph['nodes'][mutate_edge[0]]
                if begin_node=='add':
                    graph['edge_attr'][mutate_edge_attr_index]=random.choices([1,-1],[0.45,0.45],k=1)[0]
                elif begin_node=='exp':
                    graph['edge_attr'][mutate_edge_attr_index] =random.choices([-2,-1,-0.5,0.5,2],[0.1,0.15,0.1,0.1,0.1],k=1)[0]
                elif begin_node=='log':
                    graph['edge_attr'][mutate_edge_attr_index] = random.choice([10,math.e])
                elif begin_node in Triangle_Operator:
                    graph['edge_attr'][mutate_edge_attr_index] = random.choice([1,2,math.pi,2*math.pi])
                elif begin_node=='mul':
                    graph['edge_attr'][mutate_edge_attr_index] = random.choice([1, -1])
                elif begin_node=='E_exp':
                    graph['edge_attr'][mutate_edge_attr_index] = random.choice([1, -1])
                elif begin_node in Hyperbolic_Operator:
                    random.choices([1, 2], [1, 0.2])[0]

        return graph

    def construct_regressed_matrix_from_expr(self,expr,x_data):
        terms = sp.Add.make_args(expr)

        if Variables==['x1','x2','x3','x4']:
            symbols = [x1, x2, x3, x4]
            data_dict = {x1:x_data[0][0],x2:x_data[0][1],x3:x_data[0][2],x4:x_data[0][3]}
        elif Variables == ['x1', 'x2', 'x3']:
            symbols = [x1, x2, x3]
            data_dict = {x1: x_data[0][0], x2: x_data[0][1], x3: x_data[0][2]}
        # 对每个子项构造 lambdified 函数
        funcs = [sp.lambdify(symbols, term, 'numpy') for term in terms]

        # 样本数量
        N =len(x_data[0][0])

        # 构造 A
        A = np.zeros((N, len(funcs)))
        for i, func in enumerate(funcs):
            # 每个 func 接收变量向量，返回 (N,) 数组
            A[:, i] = func(*[data_dict[sym] for sym in symbols])
        return A, terms

    def get_fitness_from_graph(self,graph):
        expr = self.graph_to_sympy(graph)
        A, terms = self.construct_regressed_matrix_from_expr(expr, self.x_data)
        f_vals = self.y_data[0]
        if np.isfinite(A).all()==False:
            return 1e8, 0,terms
        else:
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A, f_vals, rcond=None)
            except np.linalg.LinAlgError:
                # print('Error: LSTSQ do not work')
                # print(expr)
                # print('terms:',terms)
                # print(A)
                # print(f_vals)
                return 1e8, 0
            f_pred = A @ coeffs
            residuals = f_vals - f_pred
            mse = np.mean(residuals ** 2)
            r2 = r2_score(f_vals, f_pred)
            fitness = 1 - r2

        return  fitness,coeffs,terms


    def get_fitness_from_expr(self,expr,x_data,y_data):
        '''
        Used for high-performance parallel computing
        since graph_to_sympy cannot be parallelled
        '''
        A, terms = self.construct_regressed_matrix_from_expr(expr,x_data)
        f_vals = y_data[0]
        if np.isfinite(A).all() == False:
            return 1e8, 0, terms
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, f_vals, rcond=None)
        except np.linalg.LinAlgError:
            # print('Error: LSTSQ do not work')
            # print(expr)
            # print('terms:', terms)
            # print(A)
            # print(f_vals)
            return 1e8,0,terms
        f_pred = A @ coeffs
        residuals = f_vals - f_pred
        mse = np.mean(residuals ** 2)
        r2= r2_score(f_vals,f_pred)
        fitness=1-r2

        return fitness,coeffs,terms


    def sorted(self,graph_list,fitness_list):
        combined = list(zip(graph_list, fitness_list))

        combined_sorted = sorted(combined, key=lambda x: x[1])

        graph_sorted, fitness_sorted = zip(*combined_sorted)
        graph_sorted = list(graph_sorted)
        fitness_sorted = list(fitness_sorted)
        return graph_sorted,fitness_sorted

    def distinction(self,graphs):
        for i in range(1,len(graphs)):
            graphs[i]=self.generate_random_graph(self.max_terms)
        return graphs

    def elimiate_terms(self,graph):

        selected_nodes_index = [graph['edges'][i][1] for i in range(len(graph['edges'])) if
                                  graph['edges'][i][0] == 0]

        if len(selected_nodes_index)<self.max_terms:
            return graph
        else:
            delete_node_index = random.sample(
                selected_nodes_index,
                len(selected_nodes_index) - self.max_terms
            )
            graph_node_index = [i for i in range(len(graph['nodes']))]
            for node_index in delete_node_index:
                subgraph_nodes, subgraph_edges, subgraph_edge_attr = self.extract_subgraph(graph, node_index)
                graph_node_index = [index for index in graph_node_index if
                                    index not in subgraph_nodes]



            for node_index in delete_node_index:
                subgraph_nodes, subgraph_edges, subgraph_edge_attr = self.extract_subgraph(graph, node_index)
                new_edges_1 = []
                new_edge_attr_1 = []
                for index, edge_info in enumerate(zip(graph['edges'], graph['edge_attr'])):
                    if edge_info[0] not in subgraph_edges:
                        if edge_info[0][1] not in subgraph_nodes:
                            new_edges_1.append(edge_info[0])
                            new_edge_attr_1.append(edge_info[1])
                graph['edges'] = new_edges_1
                graph['edge_attr'] = new_edge_attr_1
            graph['nodes'] = [graph['nodes'][i] for i in graph_node_index]
            graph = self.renumber_subgraph(graph, graph_node_index)

        return graph

    def parallel_get_fitness(self,exprs,x_data,y_data, get_fitness_from_expr):
        """
        Compute fitness for all graphs in parallel using multiprocessing.
        x_data,y_data should be amplified
        """
        max_workers = max(os.cpu_count() - 1, 1)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(get_fitness_from_expr, exprs, x_data, y_data))
        return results

    def get_fitness_from_expr_monotone_penalty(self, expr, x_data, y_data,
                                               x1_range=(2, 50),
                                               x2_range=(10, 50),
                                               x3_range=(2, 12),
                                               tol=1e-9,  # numerical tolerance for “non-decreasing”
                                               ):
        A, terms = self.construct_regressed_matrix_from_expr(expr, x_data)
        f_vals = y_data[0]  # keep your original convention: first dependent variable

        # -- If A contains NaN/Inf, bail out as before --
        if not np.isfinite(A).all():
            return 1e8, 0, terms

        # -- Initial coefficients via unconstrained least squares (warm start) --
        try:
            c0, _, _, _ = np.linalg.lstsq(A, f_vals, rcond=None)
        except np.linalg.LinAlgError:
            return 1e8, 0, terms
        regressed_expr = utils.build_regressed_expr(terms, c0, ndigits=6)
        x1, x2, x3, x4 = sp.symbols('x1 x2 x3 x4', real=True)
        f_num = sp.lambdify((x1, x2, x3, x4), regressed_expr, "numpy")
        # -- Build collocation grids for monotonicity intervals (x1, x2, x3) --

        # sweeps
        sweep_x1 = np.arange(x1_range[0], x1_range[1], 2)
        sweep_x2 = np.arange(x2_range[0], x2_range[1], 2)
        sweep_x3 = np.arange(x3_range[0], x3_range[1], 2)

        self.anchors = utils.generate_anchors(x1_range, x2_range, x3_range)

        def _mono_error(values):
            values = np.atleast_1d(values).ravel()
            if values.size < 2:
                return 0.0
            diffs = np.diff(values)
            viol = np.maximum(0.0, -(diffs) - tol)
            return np.sum(viol ** 2) / (np.sum(np.abs(diffs)) ** 2 + 1e-12)

        total_err = 0.0

        # ---- x1 monotonicity ----
        for anchor in self.anchors["x1"]:
            x2_val, x3_val, x4_val = anchor[1], anchor[2], anchor[3]
            pts = np.column_stack([
                sweep_x1,
                np.full_like(sweep_x1, x2_val),
                np.full_like(sweep_x1, x3_val),
                np.full_like(sweep_x1, x4_val),
            ])
            y = f_num(pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3])

            if np.all(np.isfinite(y)):
                total_err += _mono_error(y)
            else:
                total_err += 1.0  # heavy penalty for invalid eval

        # ---- x2 monotonicity ----
        for anchor in self.anchors["x2"]:
            x1_val, x3_val, x4_val = anchor[0], anchor[2], anchor[3]
            pts = np.column_stack([
                np.full_like(sweep_x2, x1_val),
                sweep_x2,
                np.full_like(sweep_x2, x3_val),
                np.full_like(sweep_x2, x4_val),
            ])
            y = f_num(pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3])
            if np.all(np.isfinite(y)):
                total_err += _mono_error(y)
            else:
                total_err += 1.0

        # ---- x3 monotonicity ----
        for anchor in self.anchors["x3"]:
            x1_val, x2_val, x4_val = anchor[0], anchor[1], anchor[3]
            pts = np.column_stack([
                np.full_like(sweep_x3, x1_val),
                np.full_like(sweep_x3, x2_val),
                sweep_x3,
                np.full_like(sweep_x3, x4_val),
            ])

            y = f_num(pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3])
            if np.all(np.isfinite(y)):
                total_err += _mono_error(y)
            else:
                total_err += 1.0

        # -- 5) Base fit metrics on training data using LS coefficients --
        f_pred = A @ c0
        r2 = r2_score(f_vals, f_pred)
        base_fitness = 1.0 - r2
        # Final fitness with soft monotonicity penalty (larger when violations occur)
        fitness = base_fitness + self.mono_penalty * total_err
        return fitness, c0, terms

    def evolution(self,save_dir='default'):
        self.graphs=[]
        self.fitnesses=[]
        self.best_graphs_record=[]
        self.best_fitness_record=[]
        self.exprs=[]
        all_x_data=[]
        all_y_data=[]

        if self.use_parallel_computing==True:
            for i in range(self.size_pop):
                graph=self.generate_random_graph(self.max_terms)
                graph = self.elimiate_terms(graph)
                self.graphs.append(graph)
                expr = self.graph_to_sympy(graph)
                self.exprs.append(expr)
                all_x_data.append(self.x_data)
                all_y_data.append(self.y_data)
            results = self.parallel_get_fitness(self.exprs, all_x_data, all_y_data, self.get_fitness_from_expr)
            self.fitnesses = [1e18 if pd.isna(f[0]) else f[0] for f in results]

        if self.use_parallel_computing==False:
            for i in range(self.size_pop):
                graph = self.generate_random_graph(self.max_terms)
                graph = self.elimiate_terms(graph)
                self.graphs.append(graph)
                expr=self.graph_to_sympy(graph)
                if self.use_monotonicity==True:
                    fitness = self.get_fitness_from_expr_monotone_penalty(expr,self.x_data,self.y_data)[0]
                else:
                    fitness = self.get_fitness_from_expr(expr,self.x_data,self.y_data)[0]
                if pd.isna(fitness) == True:
                    fitness = 1e18
                self.fitnesses.append(fitness)

        self.graphs,self.fitnesses=self.sorted(self.graphs,self.fitnesses)
        print('fitness:  ', self.fitnesses[0:5])
        print('exprs:  ',[self.graph_to_sympy(graph) for graph in self.graphs[0:5]])

        # These variables are used to contain the best ones, do not add other terms!!!
        best_graph = {0: self.graphs[0], 1: self.graphs[1], 2: self.graphs[2], 3: self.graphs[3], 4: self.graphs[4]}
        best_fitness = {0: self.fitnesses[0], 1: self.fitnesses[1], 2: self.fitnesses[2], 3: self.fitnesses[3],
                        4: self.fitnesses[4]}

        self.best_graphs_record.append(best_graph)
        self.best_fitness_record.append(best_fitness)


        distinction_flag=0
        for iter_num in tqdm(range(self.generation_num)):
            new_graphs=list(best_graph.values())
            new_fitness_list=list(best_fitness.values())

            for i in range(self.size_pop):
                parent1=self.graphs[i]
                parent2=self.graphs[random.randint(0,self.size_pop-1)]

                # Perform crossover
                offspring = self.cross_over(parent1, parent2)
                # Perform mutation
                offspring = self.mutate(offspring)
                offspring=self.elimiate_terms(offspring)

                if self.use_parallel_computing==False:
                    expr = self.graph_to_sympy(offspring)
                    if self.use_monotonicity==True:
                        fitness = self.get_fitness_from_expr_monotone_penalty(expr, self.x_data, self.y_data)[0]
                    else:
                        fitness = self.get_fitness_from_expr(expr, self.x_data, self.y_data)[0]

                    if pd.isna(fitness)==True:
                        fitness=1e8
                    new_fitness_list.append(fitness)

                new_graphs.append(offspring)

            if self.use_parallel_computing==True:
                new_exprs=[self.graph_to_sympy(graph) for graph in new_graphs[10:]]
                results = self.parallel_get_fitness(new_exprs, all_x_data, all_y_data,self.frac_orders,self.get_fitness_from_expr)
                new_fitness_list += [1e18 if pd.isna(f[0]) else f[0] for f in results]


            #sort
            re1 = list(map(new_fitness_list.index, heapq.nsmallest(int(self.size_pop / 2), new_fitness_list)))

            sorted_graph=[]
            sorted_fitness= []
            for index in re1:
                if new_fitness_list[index] not in sorted_fitness:
                    sorted_graph.append(new_graphs[index])
                    sorted_fitness.append(new_fitness_list[index])
            for index in range(self.size_pop-len(sorted_fitness)):
                sorted_graph.append(self.generate_random_graph(self.max_terms))
            self.graphs=sorted_graph
            self.fitnesses=sorted_fitness
            print('fitness:  ',self.fitnesses[0:5])
            print('exprs:  ',[self.graph_to_sympy(graph) for graph in self.graphs[0:5]])
            print('best graph:',self.graphs[0])
            if self.fitnesses[0]==best_fitness[0]:
                distinction_flag+=1
            else:
                distinction_flag=0

            best_graph={0:self.graphs[0],1:self.graphs[1],2:self.graphs[2],3:self.graphs[3],4:self.graphs[4]}
            best_fitness={0:self.fitnesses[0],1:self.fitnesses[1],2:self.fitnesses[2],3:self.fitnesses[3],4:self.fitnesses[4]}

            self.best_graphs_record.append(best_graph)
            self.best_fitness_record.append(best_fitness)
            if distinction_flag==self.distinction_epoch:
                distinction_flag=0
                self.graphs=self.distinction(self.graphs)

            try:
                os.makedirs(f'result_save/{save_dir}/')
            except OSError:
                pass
            if iter_num==0:
                params = {
                    "size_pop": self.size_pop,
                    "generation_num": self.generation_num,
                    "distinction_epoch": self.distinction_epoch,
                    "max_terms": self.max_terms,
                    "use_parallel_computing": self.use_parallel_computing,
                    "seek_best_initial": self.seek_best_initial,
                    "epi": self.epi,

                }

                with open(f'result_save/{save_dir}/params.txt', "w") as f:
                    for key, value in params.items():
                        f.write(f"{key}: {value}\n")
                f.close()
            if (iter_num+1)%10==0:
                pickle.dump(self.best_graphs_record,open(f'result_save/{save_dir}/best_graphs.pkl', 'wb'))
                pickle.dump(self.best_fitness_record,open(f'result_save/{save_dir}/best_fitness.pkl', 'wb'))
