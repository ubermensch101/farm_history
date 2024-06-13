import numpy as np

probs = model.predict_proba(X_test)[:, 1]
# Number of bootstrap iterations
n_iterations = 1000

# Function to calculate confidence intervals
def calculate_confidence_intervals(model, X, y, n_iterations):
    probabilities = []
    for _ in range(n_iterations):
        # Bootstrap sample
        X_boot, y_boot = resample(X, y)
        # Fit model on bootstrap sample
        model.fit(X_boot, y_boot)
        # Predict probabilities for original data 
        probabilities.append(model.predict_proba(X)[:, 1])
    
    # Calculate confidence intervals
    lower_bound = np.percentile(probabilities, 2.5, axis=0)
    upper_bound = np.percentile(probabilities, 97.5, axis=0)
    return lower_bound, upper_bound

# Calculate confidence intervals
lower_bound, upper_bound = calculate_confidence_intervals(model, X_test, Y_test, n_iterations)

for i in range(len(lower_bound)):
    print(f"Farmplot {i}: Prediction {probs[i]:.2f} \t {lower_bound[i]:.2f} - {upper_bound[i]:.2f}")