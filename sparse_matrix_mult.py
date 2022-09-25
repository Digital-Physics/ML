import time
import numpy as np


def sparse_matrix_mult(matrix_a: list[list[int]], matrix_b: list[list[int]]) -> list[list[int]]:
    """this function should speed up multiplication on sparsely filled matrices,
    but does it really run faster np.matmult()?"""
    if matrix_a and matrix_b:
        if len(matrix_a[0]) != len(matrix_b):
            return [[]]
        else:
            matrix_a_dict = get_matrix_hash(matrix_a, row_col_ind=True)
            matrix_b_dict = get_matrix_hash(matrix_b, row_col_ind=False)

            matrix_out = [[0 for _ in range(len(matrix_b[0]))] for _ in range(len(matrix_a))]

            for i in matrix_a_dict.keys():
                for j in matrix_b_dict.keys():
                    for a_col_idx in matrix_a_dict[i].keys():
                        for b_row_idx in matrix_b_dict[j].keys():
                            if a_col_idx == b_row_idx:
                                # entries are dot product of two vectors
                                matrix_out[i][j] += matrix_a_dict[i][a_col_idx]*matrix_b_dict[j][b_row_idx]

            return matrix_out
    else:
        return [[]]


def get_matrix_hash(matrix: list[list[int]], row_col_ind: bool) -> dict:
    """store non-zero matrix values in a nested dictionary"""
    dict_out = {}

    if row_col_ind:
        for i in range(len(matrix)):
            temp_row_dict = {}
            for j in range(len(matrix[0])):
                if matrix[i][j] != 0:
                    temp_row_dict[j] = matrix[i][j]
            dict_out[i] = temp_row_dict
    else:
        for j in range(len(matrix[0])):
            temp_col_dict = {}
            for i in range(len(matrix)):
                if matrix[i][j] != 0:
                    temp_col_dict[i] = matrix[i][j]
            dict_out[j] = temp_col_dict

    return dict_out


mat_a = [[1, 2, 3],
         [4, 5, 6]]

mat_b = [[1, 2, 3, 3],
         [4, 5, 6, 6],
         [7, 8, 9, 9]]

mat_c = [[0, 2, 0], [0, -3, 5]]
mat_d = [[0, 10, 0], [0, 0, 0], [0, 0, 4]]

print(sparse_matrix_mult(mat_a, mat_b))

mat_e = [[0 for _ in range(10000)] for _ in range(len(mat_a[0]))]

start = time.time()
print(sparse_matrix_mult(mat_a, mat_e))
end = time.time()

print("sparse matrix time:", end-start)

start2 = time.time()
print(np.matmul(mat_a, mat_e))
end2 = time.time()

print("np.matmul() time:", end2-start2)

print("ratio:", (end-start)/(end2-start2))
