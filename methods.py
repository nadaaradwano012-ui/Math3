# ==========================================
# MATH 3 - NUMERICAL METHODS
# ==========================================


def relative_approx_error(new, old):
    """Approximate relative error in percent."""
    if new == 0:
        return 0.0 if old == 0 else float("inf")
    return abs((new - old) / new) * 100


def _bisection_run(f, a, b, tolerance=None, max_iterations=10000,
                   residual_limit=None):
    fa, fb = f(a), f(b)

    if fa == 0:
        return a, 0, 0.0, 0.0, a, 0.0, 0, 0.0
    if fb == 0:
        return b, 0, 0.0, 0.0, b, 0.0, 0, 0.0
    if fa * fb > 0:
        raise ValueError("Invalid interval: f(a) and f(b) must have opposite signs.")

    previous = None
    approx_error = None
    best_root, lowest_residual = None, float("inf")
    best_iteration, best_approx_error = 0, None

    for iteration in range(1, max_iterations + 1):
        c = (a + b) / 2
        fc = f(c)
        residual = abs(fc)

        if previous is not None:
            approx_error = relative_approx_error(c, previous)

        if residual < lowest_residual:
            best_root, lowest_residual = c, residual
            best_iteration, best_approx_error = iteration, approx_error

        if fc == 0:
            approx_error = 0.0
            if residual < lowest_residual:
                best_root, lowest_residual = c, residual
                best_iteration, best_approx_error = iteration, approx_error
            break
        if residual_limit is not None and residual <= residual_limit:
            break
        if tolerance is not None and approx_error is not None and approx_error <= tolerance:
            break

        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc

        previous = c

    return (
        c, iteration, approx_error, residual,
        best_root, lowest_residual, best_iteration, best_approx_error
    )


def bisection(f, a, b, tolerance, max_iterations):
    """Return the user-requested run and the best residual found."""
    a, b = float(a), float(b)
    if max_iterations < 1:
        raise ValueError("Maximum iterations must be greater than zero.")

    requested = _bisection_run(
        f, a, b, tolerance=tolerance, max_iterations=max_iterations
    )
    best = _bisection_run(
        f, a, b, max_iterations=10000, residual_limit=1e-12
    )

    requested_result = (requested[0], requested[1], requested[2], requested[3])
    best_result = (best[4], best[6], best[7], best[5])
    return requested_result, best_result


def _modified_secant_run(f, x0, delta, tolerance=None,
                         max_iterations=10000, residual_limit=None):
    if delta == 0:
        raise ValueError("Delta must not be zero.")

    x = float(x0)
    initial_residual = abs(f(x))
    if initial_residual == 0:
        return x, 0, 0.0, 0.0, x, 0.0, 0, 0.0

    best_root, lowest_residual = x, initial_residual
    best_iteration, best_approx_error = 0, 0.0
    approx_error = None

    for iteration in range(1, max_iterations + 1):
        fx = f(x)
        step = delta * x

        if step == 0:
            raise ValueError("Initial guess cannot be zero for Modified Secant.")

        denominator = f(x + step) - fx
        if abs(denominator) < 1e-15:
            raise ValueError("Division by zero. Try another initial guess or delta.")

        x_new = x - (step * fx) / denominator
        approx_error = relative_approx_error(x_new, x)
        residual = abs(f(x_new))

        if residual < lowest_residual:
            best_root, lowest_residual = x_new, residual
            best_iteration, best_approx_error = iteration, approx_error

        x = x_new

        if residual_limit is not None and residual <= residual_limit:
            break
        if tolerance is not None and approx_error <= tolerance:
            break

    return (
        x, iteration, approx_error, abs(f(x)),
        best_root, lowest_residual, best_iteration, best_approx_error
    )


def modified_secant(f, x0, delta, tolerance, max_iterations):
    if max_iterations < 1:
        raise ValueError("Maximum iterations must be greater than zero.")

    requested = _modified_secant_run(
        f, x0, delta, tolerance=tolerance, max_iterations=max_iterations
    )
    best = _modified_secant_run(
        f, x0, delta, max_iterations=10000, residual_limit=1e-12
    )

    requested_result = (requested[0], requested[1], requested[2], requested[3])
    best_result = (best[4], best[6], best[7], best[5])
    return requested_result, best_result


def _gauss_seidel_run(A, b, tolerance=None, max_iterations=10000):
    n = len(b)
    x = [0.0] * n
    best_solution = x.copy()
    lowest_error = float("inf")
    best_iteration = 0
    max_error = None

    for iteration in range(1, max_iterations + 1):
        old_x = x.copy()

        for i in range(n):
            if A[i][i] == 0:
                raise ValueError("Zero on the main diagonal (division by zero).")

            total = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x[i] = (b[i] - total) / A[i][i]

        errors = [relative_approx_error(x[i], old_x[i]) for i in range(n)]
        max_error = max(errors)

        if max_error < lowest_error:
            lowest_error = max_error
            best_solution = x.copy()
            best_iteration = iteration

        if tolerance is not None and max_error <= tolerance:
            break

        if any(abs(value) > 1e100 for value in x):
            raise ValueError("The solution is diverging. Check the matrix.")

    return x, iteration, max_error, best_solution, lowest_error, best_iteration


def gauss_seidel(A, b, tolerance, max_iterations):
    n = len(b)
    if n == 0 or len(A) != n or any(len(row) != n for row in A):
        raise ValueError("Matrix A must be square and match Vector b length.")
    if max_iterations < 1:
        raise ValueError("Maximum iterations must be greater than zero.")

    requested = _gauss_seidel_run(
        A, b, tolerance=tolerance, max_iterations=max_iterations
    )
    best = _gauss_seidel_run(
        A, b, tolerance=1e-12, max_iterations=10000
    )

    return (
        (requested[0], requested[1], requested[2]),
        (best[3], best[5], best[4])
    )


def linear_regression(x, y, new_x):
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("X and Y must have the same length (at least 2 values).")

    n = len(x)
    mean_x, mean_y = sum(x) / n, sum(y) / n
    denominator = sum(value ** 2 for value in x) - n * mean_x ** 2

    if abs(denominator) < 1e-15:
        raise ValueError("All X values cannot be identical.")

    a1 = (sum(x[i] * y[i] for i in range(n)) - n * mean_x * mean_y) / denominator
    a0 = mean_y - a1 * mean_x
    predicted_y = a0 + a1 * new_x

    sum_error = sum((y[i] - (a0 + a1 * x[i])) ** 2 for i in range(n))
    sum_total = sum((value - mean_y) ** 2 for value in y)
    r2 = 1 if sum_total == 0 and sum_error == 0 else (
        0 if sum_total == 0 else 1 - sum_error / sum_total
    )
    rmse = (sum_error / n) ** 0.5

    return a0, a1, r2, predicted_y, rmse


def lagrange_interpolation(x, y, value):
    if len(x) != len(y) or len(x) == 0:
        raise ValueError("X and Y must have the same non-zero number of values.")
    if len(set(x)) != len(x):
        raise ValueError("Duplicate X values are not allowed.")

    result = 0.0
    for i in range(len(x)):
        term = 1.0
        for j in range(len(x)):
            if i != j:
                term *= (value - x[j]) / (x[i] - x[j])
        result += y[i] * term

    return result


def true_error(approximate, exact):
    """Return absolute error and relative true error percentage."""
    absolute = abs(exact - approximate)
    if exact == 0:
        return absolute, None
    return absolute, (absolute / abs(exact)) * 100
