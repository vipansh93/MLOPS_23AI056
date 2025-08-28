import autograd.numpy as np
from autograd import jacobian

def create_input_layer(image_2d):
    # Flatten 28x28 image into 1D array of 784 neurons
    if len(image_2d) != 28 or any(len(row) != 28 for row in image_2d):
        raise ValueError("Input must be a 28x28 image")
    flattened = [pixel for row in image_2d for pixel in row]
    # print("flattened",flattened) single row
    return flattened

def simple_network_function(input_vector, weights):
    # Example function: linear transformation of input
    # Weights is a matrix (e.g., 10x784 for 10 output classes)
    output = []
    for i in range(len(weights)):
        sum = 0.0
        for j in range(len(input_vector)):
            sum += weights[i][j] * input_vector[j]
        output.append(sum)
    return output

def compute_jacobian(input_vector, weights):
    # Wrap the network function for autograd
    def wrapped_function(x):
        return np.array(simple_network_function(x, weights))
    # Compute Jacobian using autograd
    jac = jacobian(wrapped_function)
    return jac(np.array(input_vector))

def main():
    # Simulate a 28x28 MNIST image (e.g., pixel values between 0 and 1)
    simulated_image = [[0.5 for _ in range(28)] for _ in range(28)]
    # print("simulated image",simulated_image[0]) #row
    # print("simulated image",simulated_image[1]) #row
    # print("simulated image outer",simulated_image) #complete matrix
    
    # Create input layer (784 neurons)
    input_layer = create_input_layer(simulated_image)
    print("Input layer (first 10 values):", input_layer[:10])
    print("Input layer length:", len(input_layer))
    
    # Simulate weights for a simple network (e.g., 10 output neurons)
    weights = [[0.1 for _ in range(784)] for _ in range(10)]
    
    # Compute output of the simple network
    output = simple_network_function(input_layer, weights)
    print("Network output:", output)
    
    # Compute Jacobian of the network function with respect to input
    jacobian_matrix = compute_jacobian(input_layer, weights)
    print("Jacobian matrix shape:", jacobian_matrix.shape)

if __name__ == "__main__":
    main()