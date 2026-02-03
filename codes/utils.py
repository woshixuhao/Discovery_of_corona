import numpy as np
import sympy as sp
from sympy import lambdify,symbols,integrate,solve
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.optimize import minimize,curve_fit
import torch
import itertools

def expr_to_human(expr):
    """
    Convert a sympy expression to human-readable math form,
    replacing variables as specified:
    x1 -> E, x2 -> d, x3 -> N, x4 -> s
    """
    # Define symbols
    x1, x2, x3, x4 = sp.symbols('x1 x2 x3 x4')
    E, d, N, s = sp.symbols('E d N s')

    # Substitute variables
    mapping = {x1: E, x2: d, x3: N, x4: s}
    expr_sub = expr.subs(mapping)

    # Convert to LaTeX string (for publication-style math)
    latex_str = sp.latex(expr_sub)

    # Or, for plain text pretty print:
    pretty_str = sp.pretty(expr_sub)
    print("LaTeX form:")
    print(latex_str)
    print("\nPretty form:")
    print(pretty_str)
    return latex_str, pretty_str

def build_regressed_expr(terms, coeffs,ndigits=3):
    """
    Combine terms and regression coefficients into a single sympy expression.

    Parameters
    ----------
    terms : list of sympy.Expr
        Basis functions (e.g., [x1, x2**2, log(x3)]).
    coeffs : array-like
        Regression coefficients corresponding to each term.

    Returns
    -------
    expr : sympy.Expr
        Combined expression, e.g. coeff1*term1 + coeff2*term2 + ...
    """
    coeffs = np.round(coeffs, ndigits)
    expr = sum(sp.Float(c, ndigits) * t for c, t in zip(coeffs, terms))
    return sp.simplify(expr)


def make_predict_fn(expr, variables):
    """
    Create a numerical function from a sympy expression.

    Parameters
    ----------
    expr : sympy.Expr
        The symbolic expression (e.g. 3*x1 + 2*log(x2)).
    variables : list of sympy.Symbol
        The variables in expr, e.g. [x1, x2, x3].

    Returns
    -------
    func : callable
        A function f(X) that takes a numpy array X of shape (n_samples, n_features)
        and returns predicted y.
    """
    f = sp.lambdify(variables, expr, "numpy")
    def func(X):
        return f(*[X[:, i] for i in range(X.shape[1])])
    return func


def generate_data_matrix(method='others'):
    """
    Generate a new data_matrix with shape (100, 4):
    - col1: linspace(12, 30, 100)
    - col2: constant 8
    - col3: constant 16.8
    - col4: constant 400
    """
    col1 = np.linspace(12, 20, 101)  # first column
    if method == 'others':
        col2 =np.full_like(col1, 33.6/10, dtype=float) # second column
    elif method == 'ours':
        col2 = np.full_like(col1, 33.6, dtype=float)  # second column
    col3 = np.full_like(col1, 8.0, dtype=float)    # third column
    col4 = np.full_like(col1, 400.0, dtype=float)  # fourth column

    data_matrix = np.column_stack([col1, col2, col3, col4])
    return data_matrix

def _build_collocation_grids(x1_range=(12.0, 30.0),
                             x2_range=(20.0, 40.0),
                             x3_range=(2.0, 12.0),
                             anchors=[12,26.8,8,400]):
    """
    Construct 1D collocation grids for each axis while holding other variables at anchor values.
    Anchors are chosen as component-wise medians of x_data (robust w.r.t. outliers).

    Returns
    -------
    grids : dict with keys 'x1','x2','x3'
        Each entry is an array of shape (n_per_axis, 4), rows are [x1, x2, x3, x4] points.
    anchors : np.ndarray shape (4,)
        Anchor point used for the non-swept coordinates.
    """


    # x1 sweep
    x1 = np.arange(x1_range[0], x1_range[1], 2)
    x2 = np.arange(x2_range[0], x2_range[1], 2)
    x3 = np.arange(x3_range[0], x3_range[1], 2)

    G1_pts = np.column_stack([
        x1,
        np.full_like(x1, anchors[1]),
        np.full_like(x1, anchors[2]),
        np.full_like(x1, anchors[3]),
    ])

    # x2 sweep
    G2_pts = np.column_stack([
        np.full_like(x2, anchors[0]),
        x2,
        np.full_like(x2, anchors[2]),
        np.full_like(x2, anchors[3]),
    ])

    # x3 sweep
    G3_pts = np.column_stack([
        np.full_like(x3, anchors[0]),
        np.full_like(x3, anchors[1]),
        x3,
        np.full_like(x3, anchors[3]),
    ])

    return {"x1": G1_pts, "x2": G2_pts, "x3": G3_pts}

def generate_anchors(x1_range,x2_range,x3_range):
    anchors_x1 = np.arange(x1_range[0], x1_range[1], 2)
    anchors_x2 = np.arange(x2_range[0], x2_range[1], 2)
    anchors_x3 = np.arange(x3_range[0], x3_range[1], 2)
    anchors_x4 = 400.0  # fixed

    # 为每个方向生成需要的锚点组合
    anchors = {
        "x1": np.array(list(itertools.product(anchors_x2, anchors_x3))),
        "x2": np.array(list(itertools.product(anchors_x1, anchors_x3))),
        "x3": np.array(list(itertools.product(anchors_x1, anchors_x2))),
    }

    # 每个方向返回 shape=(N,4) 的锚点矩阵，其中被 sweep 的维度占位 None
    anchors["x1"] = np.hstack([
        np.full((anchors["x1"].shape[0], 1), np.nan),   # x1 sweep later
        anchors["x1"][:, 0:1],                          # x2
        anchors["x1"][:, 1:2],                          # x3
        np.full((anchors["x1"].shape[0], 1), anchors_x4)
    ])

    anchors["x2"] = np.hstack([
        anchors["x2"][:, 0:1],                          # x1
        np.full((anchors["x2"].shape[0], 1), np.nan),   # x2 sweep later
        anchors["x2"][:, 1:2],                          # x3
        np.full((anchors["x2"].shape[0], 1), anchors_x4)
    ])

    anchors["x3"] = np.hstack([
        anchors["x3"][:, 0:1],                          # x1
        anchors["x3"][:, 1:2],                          # x2
        np.full((anchors["x3"].shape[0], 1), np.nan),   # x3 sweep later
        np.full((anchors["x3"].shape[0], 1), anchors_x4)
    ])

    return anchors


def _mono_error_on_grid(points,f_num,tol):
    """
    Given grid points P = [(x1_i,x2_i,x3_i,x4_i)]_{i=1..M}, compute monotonic error:
      Err = sum(viol^2) / (sum(|diffs|)^2 + δ), where viol = max(0, -(diff) - tol)
    """
    P = np.asarray(points, dtype=float)
    y = f_num(P[:, 0], P[:, 1], P[:, 2], P[:, 3])
    print(y)
    # handle invalid evaluations (log domain etc.) by heavy penalty
    if not np.all(np.isfinite(y)):
        return 1.0  # large normalized error
    diffs = np.diff(y)
    # strictly non-decreasing: diffs >= 0 (allow small negative within tol)
    viol = np.maximum(0.0, -(diffs) - tol)
    num = np.sum(viol ** 2)
    denom = (np.sum(np.abs(diffs)) ** 2) + 1e-12  # δ for numerical safety
    return num / denom

def select_cases_by_d_n(all_cases, targets):
    """
    从 all_cases 中按 (d, n) 条件提取 case。
    - all_cases: 可迭代，每个元素为二维数组/矩阵，列含 [x1, d, n, I, y] (至少 5 列)
    - targets: 需要提取的 (d, n) 元组列表，如 [(3, 10), (4, 12)]
    - 若某 (d, n) 在 all_cases 中有多个，只取首次出现的 case
    - 返回与 targets 等长的列表；若未找到则对应位置为 None
    """
    selected = []
    seen = set()

    for d, n in targets:
        if (d, n) in seen:
            # 已经取过，直接跳过
            selected.append(None)
            continue

        picked = None
        for case in all_cases:
            arr = np.asarray(case, dtype=float)
            if arr.ndim != 2 or arr.shape[1] < 3:
                continue
            d_val = arr[0, 1]
            n_val = arr[0, 2]
            if np.isclose(d_val, d) and np.isclose(n_val, n):
                picked = arr
                break

        selected.append(picked)
        if picked is not None:
            seen.add((d, n))

    return selected
def inverse_derive_params():
    x1, x2, x3, x4, x5 = sp.symbols("x1 x2 x3 x4 x5", real=True)

    # 你的两个表达式
    our_expr =  97.2 * sp.log(x1, 10) + 19.1 * sp.log(x3, 10) \
               + 41.7 * sp.log(x2, 10)-123 - 11.4 * sp.log(x5, 10) \
               - 5.8 + 10 * sp.log(3, 10) - 3.6

    BPA_expr = 120 * sp.log(x1, 10) + 26.4 * sp.log(x3, 10) \
               + 55 * sp.log(x2, 10) - 128.4 - 11.4 * sp.log(x5, 10) \
               - 5.8 + 10 * sp.log(3, 10)

    # 数值函数
    def make_predict_fn(expr, variables):
        f = sp.lambdify(variables, expr, "numpy")
        return lambda X: f(*[X[:, i] for i in range(X.shape[1])])

    predict_our = make_predict_fn(our_expr, [x1, x2, x3, x4, x5])
    predict_BPA = make_predict_fn(BPA_expr, [x1, x2, x3, x4, x5])

    # 固定参数 [G, d, N, I, Distance]
    Henan_params = np.array([21, 3.36, 8, 0.4, 15]).reshape(1, -1)

    target_our = 45.89
    target_bpa = 48.30

    solutions = []

    # 遍历搜索
    G_values = np.linspace(12, 20, 101)  # G: 15~30 kV/cm
    d_values = np.linspace(2.4, 4.0, 101)  # d: 2~5 cm

    for G in G_values:
        for d_val in d_values:
            BPA_params = Henan_params.copy()
            BPA_params[0, 0] = G
            BPA_params[0, 1] = d_val
            our_params = Henan_params.copy()
            our_params[0, 0] = G
            our_params[0, 1] = d_val*10

            val_our = predict_our(our_params)
            val_bpa = predict_BPA(BPA_params)


            #diff = abs(val_our - target_our) + abs(val_bpa - target_bpa)
            diff=abs(val_bpa - target_bpa)
            if diff < 0.005:  # 允许误差
                solutions.append((float(G), float(d_val), float(val_our), float(val_bpa), float(diff)))

    print("找到解的数量:", len(solutions))
    for s in solutions[:10]:  # 输出前10个解
        print(f"G={s[0]:.2f}, d={s[1]:.2f}, our={s[2]:.2f}, BPA={s[3]:.2f}, diff={s[4]:.4f}")