import numpy as np
import matplotlib.pyplot as plt
import time


def manual_matrix_multiply(A, B):
# Get the dimensions of the input matrices
    m, n = len(A), len(A[0])
    n, p = len(B), len(B[0])

    # Initialize the result matrix with zeros
    result = [[0 for _ in range(p)] for _ in range(m)]

    # print matrix multiplication manually
    for i in range(m):
        for j in range(p):
            for k in range(n):
                result[i][j] += A[i][k] * B[k][j]

    # print(A)
    # print(B)
    # print(result)
    return result

def CreatingMatrix(n):
    # Generate random matrices A and B of size n x n
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    
    # Measure execution time for matrix multiplication
    start_time = time.perf_counter()
    result = manual_matrix_multiply(A, B)
    end_time = time.perf_counter()
    
    return end_time - start_time

def main():
    n_values = []  # List to store values of n
    execution_times = []  # List to store execution times

    max_n = 150  # Maximum value of n
    current_n = 30  # Start with n = 2

    while current_n <= max_n:
        n_values.append(current_n)
        execution_time = CreatingMatrix(current_n)
        execution_times.append(execution_time)

        print(f"Matrix size (n): {current_n}, Execution Time: {execution_time:.6f} seconds")
        
        current_n += 1  # Increase n by 1 for the next iteration

    # Plot the execution times
    plt.plot(n_values, execution_times, marker='o',markersize = 1)
    plt.title('Manual Matrix Multiplication Execution Time')
    plt.xlabel('Matrix Size (n)')
    plt.ylabel('Execution Time (seconds)')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()