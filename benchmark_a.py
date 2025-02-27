#Libraries
import numpy as np
import matplotlib.pyplot as plt

#Rosenbrock Function
def rosenbrock(x):
    return sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

#Contour 2D Plot
def plot_contour(G_best_history):
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[rosenbrock(np.array([i, j])) for i, j in zip(x_row, y_row)] for x_row, y_row in zip(X, Y)])
    
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, Z, levels=50, cmap='jet')
    plt.colorbar(label='Function Value')
    
    G_best_history = np.array(G_best_history)
    plt.plot(G_best_history[:, 0], G_best_history[:, 1], 'wo-', markersize=5, label='Best Position Path')
    plt.scatter(G_best_history[-1, 0], G_best_history[-1, 1], color='red', marker='x', s=100, label='Final Best')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('PSO Optimization on Rosenbrock Function')
    plt.legend()
    plt.show()

#PSO Algorithm
def pso(nParticles=60, maxIter=200, wmax=0.9, wmin=0.1, c1=1.5, c2=1.5, lb=-5, ub=5, tol=1e-2): #variables - we change them in benchmarks
    #tol is the stopping factor lb ub are graph space
    np.random.seed() #randmon initial values
    dim = 10  #variables (n)
    X = np.random.uniform(lb, ub, (nParticles, dim))  #position vector as an array
    V = np.zeros((nParticles, dim))  #initial velocity arrayu set to 0 - array in python is a vector
    P_best = X.copy()
    P_best_val = np.array([rosenbrock(x) for x in X])
    G_best = P_best[np.argmin(P_best_val)]
    G_best_val = np.min(P_best_val)
    G_best_history = [G_best.copy()]
    
    for iteration in range(maxIter):
        # Dynamic inertia weight
        w = wmax - (iteration / maxIter) * (wmax - wmin)
        
        # Update velocities
        r1, r2 = np.random.rand(nParticles, dim), np.random.rand(nParticles, dim)
        V = w * V + c1 * r1 * (P_best - X) + c2 * r2 * (G_best - X)
        
        # Update positions
        X = X + V
        #X = np.clip(X, lb, ub)  # Keep within bounds not restricting here to check the behavior
        
        # Evaluate fitness
        fitness = np.array([rosenbrock(x) for x in X])
        
        # Update personal best
        improved = fitness < P_best_val
        P_best[improved] = X[improved]
        P_best_val[improved] = fitness[improved]
        
        # Update global best
        if np.min(P_best_val) < G_best_val:
            G_best_val = np.min(P_best_val)
            G_best = P_best[np.argmin(P_best_val)]
            G_best_history.append(G_best.copy())
        
        # Check for convergence
        if G_best_val <= tol:
            plot_contour(G_best_history)
            return iteration + 1, G_best_val, G_best  # Return iterations number, best value, and best position
    
    plot_contour(G_best_history)
    return maxIter, G_best_val, G_best  # If not converged, return maxIter we use this to check if the code has converged or not

# Run the PSO algorithm 10 times and record iterations
num_runs = 10
results = [pso() for _ in range(num_runs)]

# Compute median iterations
iteration_results = [res[0] for res in results]
median_iterations = int(np.median(iteration_results))

# Print results
print("Results from 10 runs:")
for i, (iters, best_val, best_pos) in enumerate(results, 1):
    print(f"Run {i}: Optimization finished in {iters} iterations with best value {best_val:.5f} at {best_pos}")

print(f"\nMedian iterations required for convergence: {median_iterations}")
