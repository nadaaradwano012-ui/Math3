import sympy as sp

def seidel(A, b, iterations):
    n = len(b)
    x = [0.0] * n

    max_error = 0
    iterations_done = 0
    is_exact = False

    for iteration in range(1, iterations + 1):
        x_old = x.copy()

        for i in range(n):
            if A[i][i] == 0:
                return None

            total = b[i]
            for j in range(n):
                if j != i:
                    total -= A[i][j] * x[j]

            x[i] = total / A[i][i]

        max_error = 0

        for i in range(n):
            if x[i] != 0:
                error = abs((x[i] - x_old[i]) / x[i]) * 100

                if error > max_error:
                    max_error = error

        iterations_done = iteration

        if max_error == 0:
            is_exact = True
            break

    return x, iterations_done, max_error, is_exact

#-------------------------------------------------

def bisection(equation, a, b, iterations):
    x = sp.symbols('x')
    expr = sp.sympify(equation)
    f = sp.lambdify(x, expr, "math")

    a = float(a)
    b = float(b)
    iterations = int(iterations)

    original_a, original_b = a, b

    if iterations <= 0:
        return None, None, None, None, 0

    if f(a) == 0:
        return a, sp.nsimplify(a), 0.0, 0.0 if a != 0 else None, 0

    if f(b) == 0:
        return b, sp.nsimplify(b), 0.0, 0.0 if b != 0 else None, 0

    if f(a) * f(b) > 0:
        return None, None, None, None, 0

    previous_c = None
    best_root = None
    min_error = None

    for iteration in range(1, iterations + 1):
        c = (a + b) / 2
        error = None

        if previous_c is not None and c != 0:
            error = abs((c - previous_c) / c) * 100

            if min_error is None or error < min_error:
                min_error = error
                best_root = c

        if f(c) == 0:
            return c, sp.nsimplify(c), error, 0.0 if c != 0 else None, iteration

        if f(a) * f(c) < 0:
            b = c
        else:
            a = c

        previous_c = c

    if best_root is None:
        best_root = c

    exact_root = None
    try:
        for root in sp.solve(expr, x):
            if root.is_real and original_a <= float(root) <= original_b:
                if exact_root is None or abs(float(root) - best_root) < abs(float(exact_root) - best_root):
                    exact_root = root
    except Exception:
        exact_root = None

    true_error = None
    if exact_root is not None and float(exact_root) != 0:
        true_error = abs((float(exact_root) - best_root) / float(exact_root)) * 100

    return best_root, exact_root, min_error, true_error, iterations
#--------------------------------------------------------------------------------

def modified_secant(f, x0, delta, max_iterations):

    xi = x0
    error = 100
    iteration = 0

    best_root = None
    best_error = None

    while iteration < max_iterations:

        f_xi = f(xi)
        f_perturbed = f(xi + delta * xi)

        denominator = f_perturbed - f_xi

        if denominator == 0:
            return None

        x_new = xi - (delta * xi * f_xi) / denominator

        if x_new == 0:
            error = abs(x_new - xi)
        else:
            error = abs((x_new - xi) / x_new) * 100

        if best_error is None or error < best_error:
            best_error = error
            best_root = x_new

        xi = x_new
        iteration = iteration + 1

    return xi, iteration, error, best_root, best_error
#--------------------------------------------------------------------
# Linear Regression Calculations

def linear_regression(x, y, new_x):

    n = len(x)

    sum_x = 0
    sum_y = 0
    sum_xy = 0
    sum_x2 = 0

    for i in range(n):
        sum_x = sum_x + x[i]
        sum_y = sum_y + y[i]
        sum_xy = sum_xy + x[i] * y[i]
        sum_x2 = sum_x2 + x[i] * x[i]

    mean_x = sum_x / n
    mean_y = sum_y / n

    denominator = sum_x2 - n * mean_x * mean_x

    if denominator == 0:
        return None

    a1 = (sum_xy - n * mean_x * mean_y) / denominator

    a0 = mean_y - a1 * mean_x

    predicted_y = a0 + a1 * new_x

    sum_error = 0
    sum_total = 0

    for i in range(n):
        predicted = a0 + a1 * x[i]

        sum_error = sum_error + (y[i] - predicted) ** 2
        sum_total = sum_total + (y[i] - mean_y) ** 2

    if sum_total == 0:
        r2 = 1
    else:
        r2 = 1 - (sum_error / sum_total)

    return a0, a1, r2, predicted_y
#---------------------------------------------------------
def lagrange_interpolation(x, y, value):

    n = len(x)
    result = 0

    for i in range(n):

        L = 1

        for j in range(n):

            if i != j:
                L = L * (value - x[j]) / (x[i] - x[j])

        result = result + y[i] * L

    return result