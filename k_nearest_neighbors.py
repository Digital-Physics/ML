from __future__ import annotations
print("Some of this code is specific to a certain data structure as shown in the my_examples dictionary")
print("In general, we need a distance metric that can handle continuous & categorical (including ordinal and boolean) data")


def predict_label(examples: dict, features: list[float], k: int, label_key: str="is_intrusive") -> int:
    """we take the majority vote. let's assume they pass in an odd value for k."""
    nearest_neighbors = find_k_nearest_neighbors(examples, features, k)

    yes_count = 0
    no_count = 0

    for id in nearest_neighbors:
        if examples[id][label_key] == 1:
            yes_count += 1
        else:
            no_count += 1

    return 1 if yes_count > no_count else 0


def find_k_nearest_neighbors(examples: dict, features: list[float], k: int) -> list[str]:
    """use the Euclidean distance to find the k nearest neighbors"""
    closest = []
    new_closest = None

    for key, value in examples.items():
        dist = euclidean_distance(features, value["features"])

        new_closest_flag = False
        if not closest:
            closest.append((key, dist))
        else:
            for i, tup in enumerate(closest):
                if dist < tup[1]:
                    new_closest_flag = True
                    new_closest = closest[:i] + [(key, dist)] + closest[i:k-1]
                    break

            if new_closest_flag:
                closest = new_closest[:]
            elif len(closest) < k:
                closest.append((key, dist))

    return [closest[i][0] for i in range(k)]


def euclidean_distance(v1: list[float], v2: list[float]) -> float:
    """Euclidean Distance between two vectors"""
    return sum([(v1[i] - v2[i])**2 for i in range(len(v1))])**(1/2)


my_examples = {
    "pid_0": {
        "features": [1.32323, 2.323343, 4.234343],
        "is_intrusive": 0},
    "pid_1": {
        "features": [1.73, 2.8343, 2.234343],
        "is_intrusive": 0},
    "pid_3": {
        "features": [2.32323, 3.323343, 1.234343],
        "is_intrusive": 1},
    "pid_4": {
        "features": [2.73, 3.8343, 5.234343],
        "is_intrusive": 1}}


print(predict_label(my_examples, [2.2, 3.345, 4.23], 3))
