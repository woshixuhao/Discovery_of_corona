import numpy as np

import GSR_multi as GSR
from utils import *
import pandas as pd
import re
import pickle
import os
from collections import Counter, defaultdict
from sympy import symbols, sympify
import  copy
'''
x1: E
x2:d
x3:n
'''

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_SAVE_DIR = os.path.join(SCRIPT_DIR, "result_save")





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
                "X": X_data,  #表面场强E
                "d": d_val/10,   #导线线径d
                "N": N_val,   #导线根数N
                "I": I_val,   #导线
                "y": y_data
            })
            case_array = case_df.dropna().to_numpy()

            all_cases.append(case_array)


    final_array = np.vstack(all_cases)
    return final_array,all_cases


def filter_by_I(data_matrix, n_to_I=None):
    # data_matrix columns: [E(col0), d(col1), N(col2), I(col3), y(col4)]
    if n_to_I is None:
        n_to_I = {6: 375, 8: 400, 4: 500}
    N_col = data_matrix[:, 2]
    I_col = data_matrix[:, 3]
    mask = np.zeros(len(data_matrix), dtype=bool)
    for n_val, i_val in n_to_I.items():
        mask |= (np.isclose(N_col, n_val) & np.isclose(I_col, i_val))
    filtered = data_matrix[mask]
    if len(filtered) == 0:
        raise ValueError(f"No rows matched n_to_I={n_to_I}. Check N and I values in data.")
    return filtered


def print_configurations(data_matrix):
    # data_matrix columns: [E(col0), d(col1), N(col2), I(col3), y(col4)]
    rows = data_matrix[:, [2, 1, 3]]  # (N, d, I)
    unique_rows = np.unique(rows, axis=0)
    print(f"{'N':>5}  {'d':>8}  {'I':>6}  {'#points':>8}")
    print("-" * 35)
    for N, d, I in unique_rows:
        mask = (
            np.isclose(data_matrix[:, 2], N) &
            np.isclose(data_matrix[:, 1], d) &
            np.isclose(data_matrix[:, 3], I)
        )
        n_pts = mask.sum()
        print(f"{int(N):>5}  {d:>8.4f}  {int(I):>6}  {n_pts:>8}")
    print("-" * 35)
    print(f"Total: {len(unique_rows)} configurations, {len(data_matrix)} points")


def split_data(data_matrix,
               val_configs=None,
               test_configs=None):
    # data_matrix columns: [E(col0), d(col1), N(col2), I(col3), y(col4)]
    # test_configs: list of (n, d) tuples identifying test configurations.
    # val_configs is kept only for backward-compatible call sites and is not used.
    if test_configs is None:
        test_configs = []

    data_matrix = np.asarray(data_matrix, dtype=float)
    N_col = data_matrix[:, 2]
    d_col = data_matrix[:, 1]
    n_x_cols = data_matrix.shape[1] - 1

    test_mask = np.zeros(len(data_matrix), dtype=bool)
    for n_val, d_val in test_configs:
        test_mask |= np.isclose(N_col, n_val) & np.isclose(d_col, d_val)

    train_mat = data_matrix[~test_mask]
    test_mat  = data_matrix[test_mask]

    if len(test_configs) > 0 and len(test_mat) == 0:
        raise ValueError(f"No test samples matched test_configs={test_configs}. "
                         f"Check N/d values against: {np.unique(np.stack([N_col, d_col], axis=1), axis=0)}")
    if len(train_mat) == 0:
        raise ValueError("No training samples remain after removing test configurations.")

    def _pack_x(mat):
        return [[mat[:, i] for i in range(n_x_cols)]]

    def _pack_y(mat):
        return [mat[:, -1]]

    print(f"Split: {len(train_mat)} train points, {len(test_mat)} test points "
          f"({len(test_configs)} test configurations)")
    return (
        _pack_x(train_mat), _pack_x(test_mat),
        _pack_y(train_mat), _pack_y(test_mat)
    )


    # data_matrix columns: [E, d, N, I, y]; configs are (N, d).
    data_matrix = np.asarray(data_matrix, dtype=float)
    exclude_configs = exclude_configs or []
    rows = data_matrix[:, [2, 1]]
    unique_configs = np.unique(rows, axis=0)
    include_configs = []
    for n_val, d_val in unique_configs:
        excluded = any(
            np.isclose(n_val, ex_n) and np.isclose(d_val, ex_d)
            for ex_n, ex_d in exclude_configs
        )
        if not excluded:
            include_configs.append((n_val, d_val))
    return get_case_matrices_by_configs(data_matrix, include_configs)

def regression_metrics(data_matrix, predict_fn):
    data_matrix = np.asarray(data_matrix, dtype=float)
    X = data_matrix[:, 0:4]
    y_true = data_matrix[:, -1]
    y_pred = np.asarray(predict_fn(X), dtype=float).ravel()
    if y_pred.size == 1 and y_true.size > 1:
        y_pred = np.full_like(y_true, y_pred.item(), dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(valid):
        raise ValueError("No finite prediction/observation pairs are available for metrics.")

    y_true = y_true[valid]
    y_pred = y_pred[valid]
    residual = y_true - y_pred
    mse = np.mean(residual ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residual))
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - np.sum(residual ** 2) / sst if sst > 0 else np.nan
    eps = 1e-12
    nonzero = np.abs(y_true) > eps
    mape = np.mean(np.abs(residual[nonzero] / y_true[nonzero])) * 100.0 if np.any(nonzero) else np.nan
    return {
        "N": len(y_true),
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "MAPE": mape,
    }

def print_metrics(name, metrics):
    print(
        f"{name}: N={metrics['N']}, "
        f"MSE={metrics['MSE']:.6g}, RMSE={metrics['RMSE']:.6g}, "
        f"MAE={metrics['MAE']:.6g}, R2={metrics['R2']:.6g}, "
        f"MAPE={metrics['MAPE']:.3f}%"
    )

if __name__ == '__main__':
    MODE='Train'
    Target='AN'
    test_configs = []
    if Target=='AN':
        test_configs = [(8, 2.68), (6, 2.68),(6, 3.36)]
    if Target=='RIEF':
        test_configs = [(6, 3), (8, 2.68),(8,3.36)]
        
    x1, x2, x3, x4 = symbols('x1 x2 x3 x4')
    if Target=='AN':
        file_path='E:/python_project/Electorinc_discovery_kang/revised_version/data/combined_data_AN.xlsx'
    if Target == 'RIEF':
        file_path = 'E:/python_project/Electorinc_discovery_kang/revised_version/data/combined_data_RIEF.xlsx'
    data_matrix,all_cases = process_data(file_path)

    data_matrix = filter_by_I(data_matrix) 
    
    #print_configurations(data_matrix)

  
    x_tr, x_te, y_tr, y_te = split_data(
        data_matrix,
        test_configs=test_configs
    )

    # x_data=[[data_matrix[:,i] for i in range(data_matrix.shape[1]-1)]]
    # y_data=[data_matrix[:,-1]]
    Graph_generation = GSR.Random_graph_for_expr()
    Graph_sympy = GSR.Graph_to_sympy()
    Optimizer = GSR.Genetic_algorithm(x_data=x_tr, y_data=y_tr)
    Optimizer.max_terms=4
    Optimizer.generation_num=100
    Optimizer.mono_penalty=1
    Optimizer.size_pop=200
    Optimizer.use_parallel_computing=False

    random_seed=2
    GSR.set_random_seeds(rand_seed=random_seed, np_rand_seed=random_seed)
    if hasattr(GSR, "torch"):
        GSR.torch.manual_seed(random_seed)
        if GSR.torch.cuda.is_available():
            GSR.torch.cuda.manual_seed_all(random_seed)

    if MODE=='Train':
        print(f'==============Begin Optimization  {Target}     max_terms={Optimizer.max_terms}   seed={random_seed}============')
        train_save_dir = f'{Target}_{Optimizer.max_terms}_{Optimizer.mono_penalty}_{Optimizer.use_monotonicity}_seed{random_seed}'
        Optimizer.evolution(os.path.join(RESULT_SAVE_DIR, train_save_dir))

    if MODE=='Valid_single':
        if Target=='AN':
            best_graph_AN_3_mono={'nodes': ['add', 'log', 'add', 'mul', 'x1', 'x3', 'mul', 'x1', 'x2', 'mul', '1', 'exp', 'add', 'x3', 'exp', 'add', 'x1', 'mul', 'x1', 'x2'], 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [3, 5], [2, 6], [6, 7], [6, 8], [0, 9], [9, 10], [9, 11], [11, 12], [12, 13], [0, 14], [14, 15], [15, 16], [15, 17], [17, 18], [17, 19]], 'edge_attr': [-1, 2.718281828459045, 1, -1, -1, 1, 1, 1, -1, -1, 1, -2, -1, 1, -1, 1, 1, 1, 1]}
            Optimizer.max_terms=3
            best_expr = Graph_sympy.graph_to_sympy(best_graph_AN_3_mono)
        if Target=='RIEF':
            best_graph_RIEF_4_mono={'nodes': ['add', '1', 'exp', 'add', 'x1', 'x2', 'mul', 'mul', 'x1', 'x2', 'exp', 'add', 'x3', 'log', 'x2'], 'edges': [[0, 1], [0, 2], [2, 3], [3, 4], [3, 5], [0, 6], [6, 7], [6, 10], [7, 8], [7, 9], [10, 11], [11, 12], [0, 13], [13, 14]], 'edge_attr': [1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 1, 2.718281828459045]}
            Optimizer.max_terms=4
            best_expr = Graph_sympy.graph_to_sympy(best_graph_RIEF_4_mono)
        fit_result = Optimizer.get_fitness_from_expr_monotone_penalty(best_expr, x_tr, y_tr)
        fitness, coef, terms = fit_result[:3]
        violation_rate = fit_result[3] if len(fit_result) > 3 else np.nan
        base_fitness = fit_result[4] if len(fit_result) > 4 else np.nan
        mono_loss = fit_result[5] if len(fit_result) > 5 else np.nan

        print("Best expr:", best_expr)
        print(f"Train monotone fitness={fitness:.6g}, base fitness={base_fitness:.6g}, "
              f"monotone loss={mono_loss:.6g}, violation rate={violation_rate:.6g}")
        print("Coefficients:", coef)
        print("Terms:", terms)

        regressed_expr=build_regressed_expr(terms,coef,ndigits=4)
        expr_to_human(regressed_expr)

        predict_fn = make_predict_fn(regressed_expr, [x1, x2, x3,x4])
        train_matrix = np.column_stack([x_tr[0][i] for i in range(len(x_tr[0]))] + [y_tr[0]])
        test_matrix = np.column_stack([x_te[0][i] for i in range(len(x_te[0]))] + [y_te[0]]) if len(y_te[0]) > 0 else None
        train_metrics = regression_metrics(train_matrix, predict_fn)
        print_metrics("Train", train_metrics)
        if test_matrix is not None:
            test_metrics = regression_metrics(test_matrix, predict_fn)
            print_metrics("Test", test_metrics)

    if MODE=='Valid_discovered':
        if Target=='AN':
            discovered_dir = os.path.join(RESULT_SAVE_DIR, 'AN_3_mono')
        if Target=='RIEF':
            discovered_dir = os.path.join(RESULT_SAVE_DIR, 'RIEF_4_mono')
        with open(os.path.join(discovered_dir, 'best_graphs.pkl'), 'rb') as f:
            best_graph = pickle.load(f)
        with open(os.path.join(discovered_dir, 'best_fitness.pkl'), 'rb') as f:
            best_fitnesses = pickle.load(f)
        best_expr = []
        for graphs in best_graph:
            exprs = []
            for i in range(5):
                exprs.append(Graph_sympy.graph_to_sympy(graphs[i]))
            best_expr.append(exprs)

        for j in range(Optimizer.generation_num-5,Optimizer.generation_num):
            print(f'The {j} epoch')
            for i in range(5):
                print(f'The #{i} best expr:, The #{i} best fitness:', best_expr[j][i], best_fitnesses[j][i])
                print(f'The #{i} best graph:', best_graph[j][i])
            print('=================================')
       
