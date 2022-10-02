from __future__ import annotations
import numpy as np

v1 = [1, 2, 3, 4, 43]
v2 = [6, 4, 2, 5, 5]
v3 = [1, 0, 1, 1, 0, 1, 1]
v4 = [1, 1, 1, 1, 0, 0, 0]
v5 = ["y", 2, "alice", 4, 43]
v6 = ["n", 4, "bob", 5, 5]


# def manhattan_distance(v_1: list[int], v_2: list[int]) -> int|float:
def manhattan_distance(v_1: list[int | float], v_2: list[int | float]) -> int | float:
    """think street block distance"""
    return sum(abs(v1_i - v2_i) for v1_i, v2_i in zip(v_1, v_2))


# def euclidean_distance(v_1: list[int|float], v_2: list[int|float]) -> float:
def euclidean_distance(v_1, v_2):
    """think flat geometry distance"""
    return sum(abs(v1_i - v2_i) ** 2 for v1_i, v2_i in zip(v_1, v_2)) ** (1 / 2)


def minkowski_distance(v_1, v_2, d):
    """distance in d dimensions; a generalization of euclidean and mahnattan"""
    return sum(abs(v1_i - v2_i) ** d for v1_i, v2_i in zip(v_1, v_2)) ** (1 / d)


def chebyshev_distance(v_1, v_2):
    """the max component distance; also equal to the minkowski with d=inf!?!
    think of a crane in a warehouse that moves on x, y, and z axes at the same rate"""
    return max(abs(v1_i - v2_i) for v1_i, v2_i in zip(v_1, v_2))


def hamming_distance(v_1, v_2):
    """bit difference distance"""
    return sum((0 if (v1_i - v2_i) == 0 else 1 for v1_i, v2_i in zip(v_1, v_2)))


def hamming_distance2(v_1, v_2):
    """use the xor to get the bit difference"""
    return sum((v1_i ^ v2_i for v1_i, v2_i in zip(v_1, v_2)))


def cosine_similarity(v_1, v_2):
    """a cosine angle between the vectors in high dimensional space.
    ranges between -1 and 1, just like Cos(angle)
    dot product of vectors/product of vector lengths"""
    dot_product = np.sum(np.array(v_1) * np.array(v_2))
    assert dot_product == np.dot(np.array(v_1), np.array(v_2))
    assert dot_product == np.matmul(np.array(v_1), np.array(v_2))
    v1_length = euclidean_distance(v_1, [0] * len(v_1))
    v2_length = euclidean_distance(v_2, [0] * len(v_2))
    return dot_product / (v1_length * v2_length)


def cosine_distance(v_1, v_2):
    """this doesn't actually follow the triangle inequality,
    so it isn't a true distance metric.
    note: it can be quick to work on sparse vectors. zero features can be ignored.
    good for directional similarity, not necessarily magnitude"""
    return 1 - cosine_similarity(v_1, v_2)


def jaccard_similarity(s_1, s_2):
    """similarity between two sets
    size of the intersection/size of the union
    think of how much two Venn diagram circles overlap"""
    size_intersection_of_sets = len(set(s_1).intersection(s_2))
    size_union_of_sets = len(set(s_1).union(s_2))
    return size_intersection_of_sets / size_union_of_sets


def jaccard_distance(s_1, s_2):
    return 1 - jaccard_similarity(s_1, s_2)


def kolmogorov_complexity_information_distance(v_1, v_2):
    """given a programming language, the length of the smallest program that transforms
    one vector into another. 1010101010101010101 is close to 0101010101010101010
    because there is a short program that bit-flips one into the other"""
    pass


def mixed_type_invented_distance(v_1, v_2):
    """made this metric up.
    future improvement: would be good if component differences ranged between 0 and 1"""
    diff_vector = [0] * len(v1)

    for i, (v1_i, v2_i) in enumerate(zip(v_1, v_2)):
        if not v1_i or not v2_i:
            diff_vector[i] = 1
        elif isinstance(v1_i, str):
            if v1_i != v2_i:
                diff_vector[i] = 1
        else:
            diff_vector[i] = v1_i - v2_i

    return euclidean_distance(diff_vector, [0] * len(diff_vector))


print("Manhattan distance:", manhattan_distance(v1, v2))
print("Euclidean distance:", euclidean_distance(v1, v2))
print("Minkowski distance (d=1):", minkowski_distance(v1, v2, 1))
print("Minkowski distance (d=2):", minkowski_distance(v1, v2, 2))
print("Minkowski distance (d=3):", minkowski_distance(v1, v2, 3))
print("Minkowski distance (d~=inf):", minkowski_distance(v1, v2, 100))
print("Chebyshev distance:", chebyshev_distance(v1, v2))
print("Hamming distance:", hamming_distance(v3, v4))
print("Hamming distance2:", hamming_distance2(v3, v4))
print("Cosine (angle) similarity:", cosine_similarity(v1, v2))
print("Cosine (angle) distance:", cosine_distance(v1, v2))
print("Jaccard (set) similarity:", jaccard_similarity(v1, v2))
print("Jaccard (set) distance:", jaccard_distance(v1, v2))
print("Made up distance metric:", mixed_type_invented_distance(v5, v6))
