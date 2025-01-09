from typing import List

def vec_product(v1: List[float], v2: List[float]) -> List[float]:
    """
    Computes the element-wise product of two vectors.

    :param v1: First vector (list of floats).
    :param v2: Second vector (list of floats).
    :return: Element-wise product of v1 and v2 if sizes match; otherwise, an empty list.
    """
    if len(v1) != len(v2):
        return []  # Return empty list if sizes differ

    return [v1[i] * v2[i] for i in range(len(v1))]  # Element-wise product

def print_vector(v: List[float]) -> None:
    """
    Prints a vector (list of floats) to the console.

    :param v: The vector to print.
    """
    print(" ".join(map(str, v)))

if __name__ == "__main__":
    v1 = [1.0, 2.0, 3.0]
    v2 = [4.0, 5.0, 6.0]

    v3 = vec_product(v1, v2)
    print_vector(v3)  # Output: 4.0 10.0 18.0

    v4 = [42.0]
    print_vector(vec_product(v1, v4))  # Output: (empty line)
