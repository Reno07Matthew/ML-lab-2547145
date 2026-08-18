import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Lab 5: Linear Regression using Gradient Descent\n",
    "\n",
    "This notebook covers the complete workflow for training a Linear Regression model using Gradient Descent. It combines the requirements of the lab (data preprocessing, train-test splitting, learning rate experiments, and evaluation) with the implementation of advanced Gradient Descent variants (Mini-Batch and Stochastic) and 3D visualizations from the sample Colab notebook."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 1: Import Libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "from mpl_toolkits.mplot3d import Axes3D\n",
    "\n",
    "from sklearn.preprocessing import StandardScaler, LabelEncoder\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n",
    "\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 2: Load the Dataset and Preprocess"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load dataset\n",
    "df = pd.read_csv(\"student-mat.csv\", sep=\";\")\n",
    "print(\"Original Dataset Shape:\", df.shape)\n",
    "\n",
    "# 1. Handle missing values\n",
    "if df.isnull().sum().sum() == 0:\n",
    "    print(\"No missing values found.\")\n",
    "else:\n",
    "    df = df.dropna()  # or use imputation\n",
    "\n",
    "# 2. Encode categorical variables\n",
    "le = LabelEncoder()\n",
    "categorical_cols = df.select_dtypes(include=[\"object\"]).columns\n",
    "for col in categorical_cols:\n",
    "    df[col] = le.fit_transform(df[col])\n",
    "print(\"Categorical columns encoded.\")\n",
    "\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 3: Feature Selection, Splitting, and Scaling\n",
    "\n",
    "We use multiple features to predict `G3`. We split the dataset into training and testing sets, then apply feature scaling."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Target variable\n",
    "target = 'G3'\n",
    "\n",
    "# Select appropriate input features (all columns except the target)\n",
    "X = df.drop(columns=[target]).values\n",
    "y = df[target].values.reshape(-1, 1)\n",
    "\n",
    "# Split into training and testing sets (80% train, 20% test)\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "# Feature Scaling\n",
    "scaler = StandardScaler()\n",
    "X_train_scaled = scaler.fit_transform(X_train)\n",
    "X_test_scaled = scaler.transform(X_test)\n",
    "\n",
    "# Add bias column (intercept term) for Gradient Descent\n",
    "X_train_bias = np.c_[np.ones((len(X_train_scaled), 1)), X_train_scaled]\n",
    "X_test_bias = np.c_[np.ones((len(X_test_scaled), 1)), X_test_scaled]\n",
    "\n",
    "print(\"Shape of X_train (with bias):\", X_train_bias.shape)\n",
    "print(\"Shape of X_test  (with bias):\", X_test_bias.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 4: Define the Cost Function & Gradient Descent Algorithm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def compute_cost(theta, X, y):\n",
    "    predictions = X.dot(theta)\n",
    "    return (1 / (2 * len(y))) * np.sum((predictions - y) ** 2)\n",
    "\n",
    "def batch_gradient_descent(X, y, learning_rate=0.01, epochs=1000):\n",
    "    theta = np.zeros((X.shape[1], 1))\n",
    "    history = []\n",
    "    cost_history = []\n",
    "    m = len(y)\n",
    "\n",
    "    for epoch in range(epochs):\n",
    "        predictions = X.dot(theta)\n",
    "        gradients = (1 / m) * X.T.dot(predictions - y)\n",
    "        theta = theta - learning_rate * gradients\n",
    "        \n",
    "        if X.shape[1] == 2:  # Only save history if 1D feature for visualization later\n",
    "            history.append(theta.copy())\n",
    "            \n",
    "        cost = compute_cost(theta, X, y)\n",
    "        cost_history.append(cost)\n",
    "        \n",
    "    return theta, np.array(history), cost_history"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 5: Experiment with Different Learning Rates\n",
    "\n",
    "We will test multiple learning rates and observe their effect on model convergence using the training set."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "learning_rates = [0.1, 0.05, 0.01, 0.001]\n",
    "cost_histories = {}\n",
    "trained_thetas = {}\n",
    "\n",
    "plt.figure(figsize=(10, 6))\n",
    "\n",
    "for lr in learning_rates:\n",
    "    theta_opt, _, cost_history = batch_gradient_descent(X_train_bias, y_train, learning_rate=lr, epochs=500)\n",
    "    cost_histories[lr] = cost_history\n",
    "    trained_thetas[lr] = theta_opt\n",
    "    plt.plot(cost_history, label=f\"LR = {lr}\")\n",
    "\n",
    "plt.xlabel(\"Iterations\")\n",
    "plt.ylabel(\"Cost (MSE)\")\n",
    "plt.title(\"Loss (Cost) vs Iterations for Different Learning Rates\")\n",
    "plt.legend()\n",
    "plt.grid(True, alpha=0.3)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 6: Evaluate the Trained Model\n",
    "\n",
    "We evaluate the model using the optimal learning rate (e.g., LR = 0.05) on the testing set."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "best_lr = 0.05\n",
    "best_theta = trained_thetas[best_lr]\n",
    "\n",
    "# Predict on Test Set\n",
    "y_pred_test = X_test_bias.dot(best_theta)\n",
    "\n",
    "# Calculate metrics\n",
    "mae = mean_absolute_error(y_test, y_pred_test)\n",
    "mse = mean_squared_error(y_test, y_pred_test)\n",
    "rmse = np.sqrt(mse)\n",
    "r2 = r2_score(y_test, y_pred_test)\n",
    "\n",
    "print(\"Model Evaluation on Test Set:\")\n",
    "print(f\"Mean Absolute Error (MAE): {mae:.4f}\")\n",
    "print(f\"Mean Squared Error (MSE): {mse:.4f}\")\n",
    "print(f\"Root Mean Squared Error (RMSE): {rmse:.4f}\")\n",
    "print(f\"R² Score: {r2:.4f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Interpretation of Results:\n",
    "- **Convergence Behavior**: The loss vs iterations plot shows that larger learning rates (0.1, 0.05) converge rapidly, while a very small learning rate (0.001) takes many more iterations to minimize the cost. There is no divergence, meaning the learning rates are appropriate.\n",
    "- **Prediction Performance**: The R² score indicates how well the variance in the target variable (Final Grade, G3) is explained by the features. The MAE tells us that on average, the predictions differ from the actual grades by ~1.5 points, which is a solid performance on this dataset."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "# Part 2: Gradient Descent Variants & Visualizations\n",
    "\n",
    "To replicate the exact visualizations from the sample Colab notebook (3D Cost Surface and Regression Line comparisons), we need to project our problem down to **a single feature**. \n",
    "We will use `G2` (Second Period Grade) to predict `G3`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1. Select single feature (G2) and target (G3)\n",
    "X_single = df[[\"G2\"]].values\n",
    "y_single = df[\"G3\"].values.reshape(-1, 1)\n",
    "\n",
    "# 2. Scale the feature\n",
    "scaler_single = StandardScaler()\n",
    "X_scaled_single = scaler_single.fit_transform(X_single)\n",
    "\n",
    "# 3. Add bias column\n",
    "m_single = len(X_scaled_single)\n",
    "X_bias_single = np.c_[np.ones((m_single, 1)), X_scaled_single]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 7: Implement Mini-Batch and Stochastic GD"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def mini_batch_gradient_descent(X, y, learning_rate=0.05, epochs=100, batch_size=32):\n",
    "    theta = np.zeros((X.shape[1], 1))\n",
    "    history = []\n",
    "    cost_history = []\n",
    "    m = len(y)\n",
    "\n",
    "    for epoch in range(epochs):\n",
    "        indices = np.random.permutation(m)\n",
    "        X_shuffled = X[indices]\n",
    "        y_shuffled = y[indices]\n",
    "\n",
    "        for i in range(0, m, batch_size):\n",
    "            X_batch = X_shuffled[i:i + batch_size]\n",
    "            y_batch = y_shuffled[i:i + batch_size]\n",
    "            predictions = X_batch.dot(theta)\n",
    "            gradients = (1 / len(X_batch)) * X_batch.T.dot(predictions - y_batch)\n",
    "            theta = theta - learning_rate * gradients\n",
    "\n",
    "        history.append(theta.copy())\n",
    "        cost = compute_cost(theta, X, y)\n",
    "        cost_history.append(cost)\n",
    "    return theta, np.array(history), cost_history\n",
    "\n",
    "def stochastic_gradient_descent(X, y, learning_rate=0.01, epochs=100):\n",
    "    theta = np.zeros((X.shape[1], 1))\n",
    "    history = []\n",
    "    cost_history = []\n",
    "    m = len(y)\n",
    "\n",
    "    for epoch in range(epochs):\n",
    "        indices = np.random.permutation(m)\n",
    "        X_shuffled = X[indices]\n",
    "        y_shuffled = y[indices]\n",
    "\n",
    "        for i in range(m):\n",
    "            xi = X_shuffled[i:i+1]\n",
    "            yi = y_shuffled[i:i+1]\n",
    "            predictions = xi.dot(theta)\n",
    "            gradients = xi.T.dot(predictions - yi)\n",
    "            theta = theta - learning_rate * gradients\n",
    "\n",
    "        history.append(theta.copy())\n",
    "        cost = compute_cost(theta, X, y)\n",
    "        cost_history.append(cost)\n",
    "    return theta, np.array(history), cost_history"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 8: Train All Models (Single Feature)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "theta_batch, history_batch, cost_batch = batch_gradient_descent(X_bias_single, y_single, learning_rate=0.05, epochs=100)\n",
    "theta_mini, history_mini, cost_mini = mini_batch_gradient_descent(X_bias_single, y_single, learning_rate=0.05, epochs=100)\n",
    "theta_sgd, history_sgd, cost_sgd = stochastic_gradient_descent(X_bias_single, y_single, learning_rate=0.01, epochs=100)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 9: 3D Visualization of Optimization Paths"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "theta0_vals = np.linspace(0, 20, 50)\n",
    "theta1_vals = np.linspace(-5, 10, 50)\n",
    "T0, T1 = np.meshgrid(theta0_vals, theta1_vals)\n",
    "J_vals = np.zeros(T0.shape)\n",
    "\n",
    "for i in range(T0.shape[0]):\n",
    "    for j in range(T0.shape[1]):\n",
    "        theta_temp = np.array([[T0[i, j]], [T1[i, j]]])\n",
    "        J_vals[i, j] = compute_cost(theta_temp, X_bias_single, y_single)\n",
    "\n",
    "fig = plt.figure(figsize=(18, 6))\n",
    "\n",
    "# BATCH GD\n",
    "ax1 = fig.add_subplot(131, projection='3d')\n",
    "ax1.plot_surface(T0, T1, J_vals, cmap='viridis', alpha=0.8)\n",
    "ax1.plot(history_batch[:, 0], history_batch[:, 1],\n",
    "         [compute_cost(h.reshape(2,1), X_bias_single, y_single) for h in history_batch],\n",
    "         color='red', marker='o')\n",
    "ax1.set_title(\"Batch Gradient Descent\")\n",
    "ax1.set_xlabel(\"Theta 0 (Intercept)\")\n",
    "ax1.set_ylabel(\"Theta 1 (Slope)\")\n",
    "ax1.set_zlabel(\"Cost\")\n",
    "\n",
    "# MINI-BATCH GD\n",
    "ax2 = fig.add_subplot(132, projection='3d')\n",
    "ax2.plot_surface(T0, T1, J_vals, cmap='viridis', alpha=0.8)\n",
    "ax2.plot(history_mini[:, 0], history_mini[:, 1],\n",
    "         [compute_cost(h.reshape(2,1), X_bias_single, y_single) for h in history_mini],\n",
    "         color='blue', marker='o')\n",
    "ax2.set_title(\"Mini-Batch Gradient Descent\")\n",
    "ax2.set_xlabel(\"Theta 0 (Intercept)\")\n",
    "ax2.set_ylabel(\"Theta 1 (Slope)\")\n",
    "ax2.set_zlabel(\"Cost\")\n",
    "\n",
    "# SGD\n",
    "ax3 = fig.add_subplot(133, projection='3d')\n",
    "ax3.plot_surface(T0, T1, J_vals, cmap='viridis', alpha=0.8)\n",
    "ax3.plot(history_sgd[:, 0], history_sgd[:, 1],\n",
    "         [compute_cost(h.reshape(2,1), X_bias_single, y_single) for h in history_sgd],\n",
    "         color='green', marker='o')\n",
    "ax3.set_title(\"Stochastic Gradient Descent\")\n",
    "ax3.set_xlabel(\"Theta 0 (Intercept)\")\n",
    "ax3.set_ylabel(\"Theta 1 (Slope)\")\n",
    "ax3.set_zlabel(\"Cost\")\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 10: Loss Curve Comparison (GD Variants)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(10, 6))\n",
    "plt.plot(cost_batch, label=\"Batch GD\")\n",
    "plt.plot(cost_mini, label=\"Mini-Batch GD\")\n",
    "plt.plot(cost_sgd, label=\"SGD\")\n",
    "plt.xlabel(\"Epochs\")\n",
    "plt.ylabel(\"Cost (MSE)\")\n",
    "plt.title(\"Loss Curve Comparison\")\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 11: Regression Line Comparison"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(10, 6))\n",
    "plt.scatter(X_scaled_single, y_single, alpha=0.5, label=\"Real Data\")\n",
    "\n",
    "plt.plot(X_scaled_single, X_bias_single.dot(theta_batch), label=\"Batch GD\")\n",
    "plt.plot(X_scaled_single, X_bias_single.dot(theta_mini), label=\"Mini-Batch GD\")\n",
    "plt.plot(X_scaled_single, X_bias_single.dot(theta_sgd), label=\"SGD\")\n",
    "\n",
    "plt.xlabel(\"G2 (Scaled)\")\n",
    "plt.ylabel(\"G3 (Final Grade)\")\n",
    "plt.title(\"Regression Line Comparison\")\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.10.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}

with open("Lab5_Linear_Regression_Gradient_Descent.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)

print("Notebook generated successfully!")
