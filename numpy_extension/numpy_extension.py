import numpy as np


def gs_solve(a_matrix, b_matrix, tol=1e-6, max_iterations=2000, to_list=False):
    """Uses the Gauss-Seidel recursive method to solve systems of equations."""
    if not (isinstance(a_matrix, np.ndarray) and isinstance(a_matrix, np.ndarray)):
        a_matrix = np.array(a_matrix, dtype=float)
        b_matrix = np.array(b_matrix, dtype=float)
    if not is_square(a_matrix):
        raise np.linalg.LinAlgError("Invalid A matrix.")
    matrix_size = a_matrix.shape[0]
    lower = -np.tril(a_matrix)
    upper = -np.triu(a_matrix)
    for i in range(matrix_size):
        lower[i, i] = 0
        upper[i, i] = 0
    diagonal = a_matrix + lower + upper
    g = np.linalg.inv(diagonal - lower)
    c = g @ b_matrix
    g = g @ upper
    norms = get_norms(g)
    if not (norms[0] < 1 or norms[1] < 1):
        raise np.linalg.LinAlgError("Gauss-Seidel does not fulfill convergence criteria.")
    x_new = np.zeros(b_matrix.shape, dtype=float)
    for i in range(max_iterations):
        x = x_new
        x_new = g @ x + c
        if np.abs(x_new - x).max() < tol:
            if to_list:
                return x_new.tolist()
            else:
                return x_new
    else:
        raise np.linalg.LinAlgError("Gauss-Seidel method did not converge.")


def gs_does_converge(a_matrix):
    """Checks if the Gauss-Seidel recursive method converges."""
    if not isinstance(a_matrix, np.ndarray):
        a_matrix = np.array(a_matrix)
    matrix_size = a_matrix.shape[0]
    lower = -np.tril(a_matrix)
    upper = -np.triu(a_matrix)
    for i in range(matrix_size):
        lower[i, i] = 0
        upper[i, i] = 0
    diagonal = a_matrix + lower + upper
    g = np.linalg.inv(diagonal - lower)
    g = g @ upper
    norms = get_norms(g)
    if not (norms[0] < 1 or norms[1] < 1):
        return False
    else:
        return True


def jac_solve(a_matrix, b_matrix, tol=1e-6, max_iterations=2000, to_list=False):
    """Uses the Jacobi recursive method to solve systems of equations."""
    if not (isinstance(a_matrix, np.ndarray) and isinstance(a_matrix, np.ndarray)):
        a_matrix = np.array(a_matrix, dtype=float)
        b_matrix = np.array(b_matrix, dtype=float)
    if not is_square(a_matrix):
        raise np.linalg.LinAlgError("Invalid A matrix.")
    matrix_size = a_matrix.shape[0]
    lower = -np.tril(a_matrix)
    upper = -np.triu(a_matrix)
    for i in range(matrix_size):
        lower[i, i] = 0
        upper[i, i] = 0
    diagonal = a_matrix + lower + upper
    g = np.linalg.inv(diagonal)
    c = g @ b_matrix
    g = g @ (lower + upper)
    norms = get_norms(g)
    if not (norms[0] < 1 or norms[1] < 1):
        raise np.linalg.LinAlgError("Jacobi does not fulfill convergence criteria.")
    x_new = np.zeros(b_matrix.shape, dtype=float)
    for i in range(max_iterations):
        x = x_new
        x_new = g @ x + c
        if np.abs(x_new - x).max() < tol:
            if to_list:
                return x_new.tolist()
            else:
                return x_new
    else:
        raise np.linalg.LinAlgError("Jacobi method did not converge.")


def jac_does_converge(a_matrix):
    if not isinstance(a_matrix, np.ndarray):
        a_matrix = np.array(a_matrix, dtype=float)
    if not is_square(a_matrix):
        raise np.linalg.LinAlgError("Invalid A matrix.")
    matrix_size = a_matrix.shape[0]
    lower = -np.tril(a_matrix)
    upper = -np.triu(a_matrix)
    for i in range(matrix_size):
        lower[i, i] = 0
        upper[i, i] = 0
    diagonal = a_matrix + lower + upper
    g = np.linalg.inv(diagonal)
    g = g @ (lower + upper)
    norms = get_norms(g)
    if not (norms[0] < 1 or norms[1] < 1):
        return False
    else:
        return True


def is_square(matrix):
    """Checks if the matrix is square."""
    dimensions = matrix.shape
    if len(dimensions) != 2:
        return False
    else:
        if dimensions[0] == dimensions[1]:
            return True
        else:
            return False


def get_norms(matrix):
    """Returns the infinite and 0 norm."""
    return np.abs(matrix).max(), np.abs(matrix).max(1).max()


if __name__ == "__main__":
    systems = [[[[1., 4.], [2., -4]], [2., 0.]], [[[5., -2.], [5., -13]], [8., -67.]],
               [[[12., 4.], [-56., -12]], [-10., 50.]],
               [[[456., -413.], [-2., -1231]], [-10., 50.]],
               [[[876., -432.], [-218., -12334]], [[-10., 50.], [-1660., 534.]]]]
    for system in systems:
        ge = np.linalg.solve(*system)
        try:
            gs = gs_solve(*system, tol=1e-10, to_list=True)
            print("Gauss-Seidel: " + str(gs) + " Gauss elimination: " + str(ge.tolist()) + " Test results: "
                  + str(gs == ge))
        except np.linalg.LinAlgError as e:
            print(e)
            print("Gauss elimination: " + str(ge.tolist()))
            continue
        try:
            jac = jac_solve(*system, to_list=True, tol=1e-10)
        except np.linalg.LinAlgError as e:
            print(e)
            continue
        print("Gauss-Seidel: " + str(gs) + " Jacobi: " + str(jac) + " Gauss elimination: "
              + str(ge.tolist()) + " Test results: " + str(gs == ge))
