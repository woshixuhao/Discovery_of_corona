import GSR_multi as GSR
from plot import *
import pandas as pd
import re
import pickle
from sympy import symbols, sympify
import  copy

'''
In this work, x1: E, x2:d, x3:n
'''


def process_data(file_path):
    df = pd.read_excel(file_path, sheet_name='Sheet1')
    pattern = re.compile(r"d([\d\.]+)_N(\d+)_I(\d+)", re.IGNORECASE)
    all_cases = []

    for col in df.columns:
        if col.startswith("X_"):
            y_col = "Y" + col[1:]
            if y_col not in df.columns:
                continue

            m = pattern.search(col)
            if not m:
                continue
            d_val = float(m.group(1))
            N_val = int(m.group(2))
            I_val = int(m.group(3))


            X_data = df[col].values
            y_data = df[y_col].values


            case_df = pd.DataFrame({
                "X": X_data,  #E
                "d": d_val/10,   #d
                "N": N_val,   #N
                "I": I_val,   #I
                "y": y_data
            })
            case_array = case_df.dropna().to_numpy()

            all_cases.append(case_array)


    final_array = np.vstack(all_cases)
    return final_array,all_cases


if __name__ == '__main__':
    MODE='Compare_RIEF'
    Target='RIEF'
    x1, x2, x3, x4 = symbols('x1 x2 x3 x4')
    if Target=='AN':
        file_path='dataset/combined_data_AN.xlsx'
    if Target == 'RIEF':
        file_path = 'dataset/combined_data_RIEF.xlsx'

    data_matrix,all_cases = process_data(file_path)
    x_data=[[data_matrix[:,i] for i in range(data_matrix.shape[1]-1)]]
    y_data=[data_matrix[:,-1]]

    Graph_generation = GSR.Random_graph_for_expr()
    Graph_sympy = GSR.Graph_to_sympy()
    Optimizer = GSR.Genetic_algorithm(x_data=x_data, y_data=y_data)


    if MODE=='Train':
        Optimizer.max_terms=2
        Optimizer.generation_num=200
        Optimizer.size_pop=500
        print(f'==============Begin Optimization  {Target}     max_terms={Optimizer.max_terms}   ============')
        Optimizer.evolution(f'{Target}_{Optimizer.max_terms}_mono')

    if MODE=='Valid':
        best_graph_AN_5_mono= {'nodes': ['add', 'mul', 'x3', 'exp', 'add', 'mul', 'x2', 'x2', 'mul', 'x1', 'log', 'x1', 'log', 'x3', 'mul', 'x2', 'exp', 'add', 'mul', 'x1', 'x2', 'mul', 'x3', 'x2', 'x3', 'x2'], 'edges': [[0, 1], [1, 2], [1, 3], [3, 4], [4, 5], [5, 6], [5, 7], [4, 8], [8, 9], [8, 10], [10, 11], [0, 12], [12, 13], [0, 14], [14, 15], [14, 16], [16, 17], [17, 18], [18, 19], [18, 20], [0, 21], [21, 22], [21, 23], [21, 24], [0, 25]], 'edge_attr': [-1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 2.718281828459045, -1, 2.718281828459045, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1]}
        best_expr_AN_5 = Graph_sympy.graph_to_sympy(best_graph_AN_5_mono)

        best_graph_AN_4_mono ={'nodes': ['add', 'mul', '1', 'exp', 'add', 'x1', 'log', 'x2', 'mul', 'x3', 'x1', 'mul', '1', 'exp', 'add', 'mul', 'x2', 'x1', 'x2'], 'edges': [[0, 1], [1, 2], [1, 3], [3, 4], [4, 5], [0, 6], [6, 7], [0, 8], [8, 9], [8, 10], [0, 11], [11, 12], [11, 13], [13, 14], [14, 15], [15, 16], [15, 17], [15, 18]], 'edge_attr': [1, -1, -1, -2, 1, -1, 10, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, -1]}
        best_expr_AN_4 = Graph_sympy.graph_to_sympy(best_graph_AN_4_mono)

        best_graph_AN_3_mono = {'nodes': ['add', 'mul', '1', 'exp', 'add', 'mul', 'x1', 'log', 'x1', 'log', 'x2', 'mul', 'x3', 'x1'], 'edges': [[0, 1], [1, 2], [1, 3], [3, 4], [4, 5], [5, 6], [5, 7], [7, 8], [0, 9], [9, 10], [0, 11], [11, 12], [11, 13]], 'edge_attr': [1, 1, -1, -1, 1, -1, 1, 10, 1, 2.718281828459045, 1, -1, 1]}
        best_expr_AN_3 = Graph_sympy.graph_to_sympy(best_graph_AN_3_mono)


        #================RIEF====================================
        best_graph_RIEF_3_mono ={'nodes': ['add', 'mul', 'x3', 'x2', 'x2', 'mul', '1', 'exp', 'add', 'x2', 'mul', 'x2', 'x1', 'log', 'x1'], 'edges': [[0, 1], [1, 2], [1, 3], [1, 4], [0, 5], [5, 6], [5, 7], [7, 8], [8, 9], [8, 10], [10, 11], [10, 12], [0, 13], [13, 14]], 'edge_attr': [1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, 10]}
        best_expr_RIEF_3 = Graph_sympy.graph_to_sympy(best_graph_RIEF_3_mono)

        best_graph_RIEF_4_mono =  {'nodes': ['add', 'mul', 'x3', 'exp', 'add', 'mul', 'x3', 'x3', 'x2', 'x2', 'mul', '1', 'exp', 'add', 'x1', '1', 'mul', 'x3', 'exp', 'add', 'mul', 'x3', 'x2', 'x2', 'x1'], 'edges': [[0, 1], [1, 2], [1, 3], [3, 4], [4, 5], [4, 9], [5, 6], [5, 7], [5, 8], [0, 10], [10, 11], [10, 12], [12, 13], [13, 14], [0, 15], [0, 16], [16, 17], [16, 18], [18, 19], [19, 20], [19, 24], [20, 21], [20, 22], [20, 23]], 'edge_attr': [1, 1, -1, -1, -1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, 1, -1, 1]}
        best_expr_RIEF_4 = Graph_sympy.graph_to_sympy(best_graph_RIEF_4_mono)

        best_graph_RIEF_5_mono =  {'nodes': ['add', 'mul', 'x3', 'exp', 'add', 'mul', 'x3', 'x1', '1', 'mul', 'mul', 'x1', 'x2', 'x3', 'exp', 'add', 'mul', 'x3', 'x2', 'x3', 'x2', 'mul', '1', 'exp', 'add', 'mul', 'x1', 'x2'], 'edges': [[0, 1], [1, 2], [1, 3], [3, 4], [4, 5], [5, 6], [5, 7], [0, 8], [0, 9], [9, 10], [10, 11], [10, 12], [10, 13], [9, 14], [14, 15], [15, 16], [16, 17], [16, 18], [16, 19], [0, 20], [0, 21], [21, 22], [21, 23], [23, 24], [24, 25], [25, 26], [25, 27]], 'edge_attr': [-1, -1, 1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -2, 1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 1, -1]}
        best_expr_RIEF_5 = Graph_sympy.graph_to_sympy(best_graph_RIEF_5_mono)


        best_expr=best_expr_AN_3
        fitness, coef, terms = Optimizer.get_fitness_from_expr_monotone_penalty(best_expr, x_data, y_data)
        regressed_expr=build_regressed_expr(terms,coef,ndigits=4)
        expr_to_human(regressed_expr)
        predict_fn = make_predict_fn(regressed_expr, [x1, x2, x3,x4])
        cal_error(data_matrix, predict_fn)

    if MODE=='Valid_discovered':
        best_graph = pickle.load(open(f'result_save/RIEF_4_mono/best_graphs.pkl', 'rb'))
        best_fitnesses = pickle.load(open(f'result_save/RIEF_4_mono/best_fitness.pkl', 'rb'))
        best_expr = []
        for graphs in best_graph:
            exprs = []
            for i in range(5):
                exprs.append(Graph_sympy.graph_to_sympy(graphs[i]))
            best_expr.append(exprs)

        for j in range(150,200):
            print(f'The {j} epoch')
            for i in range(5):
                print(f'The #{i} best expr:, The #{i} best fitness:', best_expr[j][i], best_fitnesses[j][i])
                print(f'The #{i} best graph:', best_graph[j][i])
            print('=================================')


