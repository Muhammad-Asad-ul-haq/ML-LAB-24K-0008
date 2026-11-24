import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ast import literal_eval
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")

X, y = make_moons(n_samples=500, noise=0.22, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

architectures = [(8, 8), (2,), (16,), (16, 16)]
performance = []

for layers in architectures:
    classifier = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=layers,
            activation="relu",
            solver="adam",
            alpha=0.001,
            max_iter=2000,
            random_state=42
        )
    )

    start_time = time.perf_counter()
    classifier.fit(X_train, y_train)
    duration = time.perf_counter() - start_time

    performance.append({
        "Architecture": str(layers),
        "Training Accuracy": round(classifier.score(X_train, y_train), 3),
        "Testing Accuracy": round(classifier.score(X_test, y_test), 3),
        "Iterations": classifier[-1].n_iter_,
        "Training Time": round(duration, 3)
    })

print(pd.DataFrame(performance).to_string(index=False))

X, y = make_moons(n_samples=500, noise=0.20, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
y_train = y_train.reshape(-1, 1)


def sigmoid(value):
    return 1 / (1 + np.exp(-np.clip(value, -40, 40)))


def train_network(learning_rate, reg=0.001, epochs=4000):

    generator = np.random.default_rng(42)

    W1 = generator.normal(0, np.sqrt(2 / 2), (2, 8))
    b1 = np.zeros((1, 8))

    W2 = generator.normal(0, np.sqrt(2 / 8), (8, 1))
    b2 = np.zeros((1, 1))

    loss_history = []
    samples = len(X_train)
    epsilon = 1e-10

    for epoch in range(epochs):

        hidden_input = X_train @ W1 + b1
        hidden_output = np.maximum(hidden_input, 0)
        probabilities = sigmoid(hidden_output @ W2 + b2)

        cross_entropy = -np.mean(
            y_train * np.log(probabilities + epsilon)
            + (1 - y_train) * np.log(1 - probabilities + epsilon)
        )

        regularization = (reg / 2) * (
            np.sum(W1 ** 2) + np.sum(W2 ** 2)
        )

        loss = cross_entropy + regularization
        loss_history.append(loss)

        output_error = (probabilities - y_train) / samples

        grad_W2 = hidden_output.T @ output_error + reg * W2
        grad_b2 = output_error.sum(axis=0, keepdims=True)

        hidden_error = (output_error @ W2.T) * (hidden_input > 0)

        grad_W1 = X_train.T @ hidden_error + reg * W1
        grad_b1 = hidden_error.sum(axis=0, keepdims=True)

        W1 -= learning_rate * grad_W1
        b1 -= learning_rate * grad_b1
        W2 -= learning_rate * grad_W2
        b2 -= learning_rate * grad_b2

    test_hidden = np.maximum(X_test @ W1 + b1, 0)
    test_probabilities = sigmoid(test_hidden @ W2 + b2)

    return loss_history, test_probabilities


learning_rates = [0.01, 0.1, 1.0]
network_results = {}

for rate in learning_rates:

    history, probabilities = train_network(rate)
    network_results[rate] = (history, probabilities)

    predictions = (probabilities.flatten() >= 0.5).astype(int)

    loss_values = np.asarray(history)

    increases = int(np.sum(np.diff(loss_values) > 1e-9))

    reached_target = np.flatnonzero(loss_values < 0.3)

    target_epoch = (
        int(reached_target[0])
        if len(reached_target) > 0
        else None
    )

    print("\nLearning Rate:", rate)
    print("Final Loss:", round(history[-1], 4))
    print("Test Accuracy:", round(accuracy_score(y_test, predictions), 3))
    print("Epoch index reaching loss below 0.3:", target_epoch)
    print("Number of loss increases:", increases)

    if not np.all(np.isfinite(loss_values)):
        print("Training has diverged.")

    elif increases > 0.05 * len(loss_values):
        print("Unstable training with frequent loss oscillations.")

    elif target_epoch is None:
        print("Stable but slow learning within 4000 epochs.")

    else:
        print("Stable training with smooth convergence.")


plt.figure(figsize=(8, 4))

for rate, (history, probabilities) in network_results.items():
    plt.plot(history, label=f"Learning rate = {rate}")

plt.yscale("log")
plt.xlabel("Epochs")
plt.ylabel("Training Loss")
plt.title("Effect of Learning Rate on Network Training")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


probabilities = network_results[0.1][1].flatten()
predicted_labels = (probabilities >= 0.5).astype(int)

print("First five probabilities:", np.round(probabilities[:5], 3))
print("First five predicted classes:", predicted_labels[:5])

print(
    "Probability represents the estimated likelihood of belonging "
    "to class 1, while a predicted class is obtained by applying "
    "a classification threshold."
)

example = np.array([0.51, 0.99])

print("Example probabilities:", example)
print("Corresponding classes:", (example >= 0.5).astype(int))

print(
    "Both probabilities produce class 1 but represent "
    "different confidence levels."
)

for threshold in [0.5, 0.8]:
    positive_count = int(np.sum(probabilities >= threshold))
    print(f"Threshold {threshold}: {positive_count} positive predictions")

print(
    "Changing the threshold changes predicted classes "
    "without modifying the original probabilities."
)


dataset_url = (
    "https://raw.githubusercontent.com/rfordatascience/"
    "tidytuesday/master/data/2020/2020-02-11/hotels.csv"
)

hotel_df = pd.read_csv(dataset_url)

print("Original dataset shape:", hotel_df.shape)

selected_columns = [
    "is_canceled",
    "lead_time",
    "adr",
    "previous_cancellations",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "reservation_status"
]

print("\nMissing values:")
print(hotel_df[selected_columns].isnull().sum())

hotel_df = hotel_df.drop(
    columns=["reservation_status", "reservation_status_date"]
)

hotel_df["total_nights"] = (
    hotel_df["stays_in_weekend_nights"]
    + hotel_df["stays_in_week_nights"]
)

feature_names = [
    "lead_time",
    "adr",
    "previous_cancellations",
    "total_nights"
]

clean_data = hotel_df[
    feature_names + ["is_canceled"]
].dropna().copy()

adr_limit = clean_data["adr"].quantile(0.999)

valid_records = (
    (clean_data["adr"] >= 0)
    & (clean_data["adr"] < adr_limit)
    & (clean_data["total_nights"] > 0)
)

clean_data = clean_data.loc[valid_records].copy()

print("Cleaned dataset shape:", clean_data.shape)

print("\nClass distribution:")
print(clean_data["is_canceled"].value_counts(normalize=True).round(3))

X = clean_data[feature_names].to_numpy()
y = clean_data["is_canceled"].to_numpy()

X_remaining, X_test, y_remaining, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_remaining, y_remaining,
    test_size=0.25,
    stratify=y_remaining,
    random_state=42
)

print("Training set:", X_train.shape)
print("Validation set:", X_valid.shape)
print("Testing set:", X_test.shape)

validation_records = []

for penalty_type in ["l2", "l1"]:

    for c_value in [0.001, 0.01, 0.1, 1.0, 10.0]:

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                penalty=penalty_type,
                C=c_value,
                solver="liblinear",
                max_iter=2000
            )
        )

        model.fit(X_train, y_train)
        valid_predictions = model.predict(X_valid)

        validation_records.append({
            "Model": "LR-" + penalty_type,
            "Parameter": c_value,
            "Validation F1": f1_score(y_valid, valid_predictions),
            "Validation Accuracy": accuracy_score(y_valid, valid_predictions)
        })


for configuration in [(8,), (16,), (16, 8)]:

    for alpha_value in [0.0001, 0.001, 0.01]:

        model = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=configuration,
                alpha=alpha_value,
                learning_rate_init=0.01,
                max_iter=300,
                early_stopping=True,
                random_state=42
            )
        )

        model.fit(X_train, y_train)
        valid_predictions = model.predict(X_valid)

        validation_records.append({
            "Model": "MLP" + str(configuration),
            "Parameter": alpha_value,
            "Validation F1": f1_score(y_valid, valid_predictions),
            "Validation Accuracy": accuracy_score(y_valid, valid_predictions)
        })


validation_table = pd.DataFrame(validation_records).round(4)

print(validation_table.to_string(index=False))

lr_results = validation_table[
    validation_table["Model"].str.startswith("LR")
]

mlp_results = validation_table[
    validation_table["Model"].str.startswith("MLP")
]

best_lr = lr_results.sort_values(
    "Validation F1", ascending=False
).iloc[0]

best_mlp = mlp_results.sort_values(
    "Validation F1", ascending=False
).iloc[0]

print("\nBest Logistic Regression Configuration:")
print(best_lr.to_dict())

print("\nBest MLP Configuration:")
print(best_mlp.to_dict())


selected_penalty = best_lr["Model"].split("-")[1]
selected_layers = literal_eval(best_mlp["Model"][3:])

logistic_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        penalty=selected_penalty,
        C=best_lr["Parameter"],
        solver="liblinear",
        max_iter=2000
    )
)

neural_model = make_pipeline(
    StandardScaler(),
    MLPClassifier(
        hidden_layer_sizes=selected_layers,
        alpha=best_mlp["Parameter"],
        learning_rate_init=0.01,
        max_iter=300,
        random_state=42
    )
)

X_combined = np.concatenate((X_train, X_valid), axis=0)
y_combined = np.concatenate((y_train, y_valid))

logistic_model.fit(X_combined, y_combined)
neural_model.fit(X_combined, y_combined)

models = {
    "Regularized Logistic Regression": logistic_model,
    "MLP": neural_model
}

evaluation = []
confusion_matrices = {}

for model_name, fitted_model in models.items():

    predictions = fitted_model.predict(X_test)

    confusion_matrices[model_name] = confusion_matrix(
        y_test, predictions
    )

    evaluation.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1": f1_score(y_test, predictions)
    })

comparison_table = pd.DataFrame(evaluation).round(4)

print("\nFinal Model Comparison:")
print(comparison_table.to_string(index=False))

for model_name, matrix in confusion_matrices.items():
    print("\nConfusion Matrix:", model_name)
    print(matrix)


fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].plot(neural_model[-1].loss_curve_)
axes[0].set_xlabel("Iteration")
axes[0].set_ylabel("Training Loss")
axes[0].set_title("MLP Training Performance")
axes[0].grid(True)

line_styles = {"l2": "o-", "l1": "s--"}

for penalty_type, style in line_styles.items():

    subset = validation_table[
        validation_table["Model"] == "LR-" + penalty_type
    ]

    axes[1].semilogx(
        subset["Parameter"],
        subset["Validation F1"],
        style,
        label=penalty_type.upper()
    )

axes[1].set_xlabel("C (Inverse Regularization Strength)")
axes[1].set_ylabel("Validation F1")
axes[1].set_title("Regularization vs Validation F1")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for axis, (model_name, matrix) in zip(
    axes, confusion_matrices.items()
):

    ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Kept", "Canceled"]
    ).plot(ax=axis, colorbar=False)

    axis.set_title(model_name)

plt.tight_layout()
plt.show()


clean_data["lead_category"] = pd.cut(
    clean_data["lead_time"],
    bins=[-1, 7, 30, 90, 180, 1000],
    labels=["0-7", "8-30", "31-90", "91-180", "180+"]
)

lead_cancellation = clean_data.groupby(
    "lead_category", observed=True
)["is_canceled"].mean()

clean_data["previous_status"] = np.where(
    clean_data["previous_cancellations"] > 0,
    "Previous cancellation",
    "No previous cancellation"
)

previous_cancellation = clean_data.groupby(
    "previous_status"
)["is_canceled"].mean()

clean_data["price_category"] = pd.qcut(
    clean_data["adr"],
    q=4,
    labels=[
        "Lowest quartile",
        "Second quartile",
        "Third quartile",
        "Highest quartile"
    ]
)

price_cancellation = clean_data.groupby(
    "price_category", observed=True
)["is_canceled"].mean()

feature_coefficients = pd.Series(
    logistic_model[-1].coef_[0],
    index=feature_names
).round(3)

print("\nCancellation Rate by Lead Time:")
print(lead_cancellation.round(3).to_string())

print("\nCancellation Rate by Previous History:")
print(previous_cancellation.round(3).to_string())

print("\nCancellation Rate by Price Category:")
print(price_cancellation.round(3).to_string())

print("\nLogistic Regression Coefficients:")
print(feature_coefficients.to_string())

short_lead = lead_cancellation.iloc[0] * 100
long_lead = lead_cancellation.iloc[-1] * 100

past_cancel = previous_cancellation["Previous cancellation"] * 100
no_past_cancel = previous_cancellation["No previous cancellation"] * 100

low_price = price_cancellation.iloc[0] * 100
high_price = price_cancellation.iloc[-1] * 100

print()

print(
    f"Business Insight 1: Bookings made within seven days have "
    f"a cancellation rate of {short_lead:.1f}%, compared with "
    f"{long_lead:.1f}% for bookings made over 180 days in advance. "
    f"Hotels can consider deposits and reconfirmation for long-lead bookings."
)

print(
    f"Business Insight 2: Customers with previous cancellations "
    f"have a cancellation rate of {past_cancel:.1f}%, compared with "
    f"{no_past_cancel:.1f}% for other customers. Prepayment or "
    f"non-refundable booking options may help manage this risk."
)

print(
    f"Business Insight 3: The cancellation rate is {low_price:.1f}% "
    f"in the lowest price quartile and {high_price:.1f}% in the "
    f"highest quartile. Hotels can use these patterns to develop "
    f"appropriate cancellation policies and controlled overbooking."
)

best_model_name = comparison_table.sort_values(
    "F1", ascending=False
).iloc[0]["Model"]

print(
    f"Operational Recommendation: {best_model_name} achieved the "
    f"highest reported test F1 score. Its predicted cancellation "
    f"probabilities can identify high-risk reservations for reminders, "
    f"deposits and limited overbooking. The classification threshold "
    f"should reflect the costs of empty rooms and displaced guests."
)


xor_inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

xor_targets = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=float)

linear_neuron = LogisticRegression(
    C=1e6,
    max_iter=10000
)

linear_neuron.fit(xor_inputs, xor_targets.flatten())

linear_accuracy = linear_neuron.score(
    xor_inputs, xor_targets.flatten()
)

print("Single Neuron XOR Accuracy:", linear_accuracy)

print(
    "A single linear neuron generates a straight-line decision "
    "boundary. XOR cannot be solved using one linear boundary "
    "because its classes are not linearly separable."
)

generator = np.random.default_rng(1)

weights_hidden = generator.normal(0, 1, (2, 4))
bias_hidden = np.zeros((1, 4))

weights_output = generator.normal(0, 1, (4, 1))
bias_output = np.zeros((1, 1))

learning_rate = 0.5
xor_history = []

for epoch in range(10000):

    hidden_input = xor_inputs @ weights_hidden + bias_hidden
    hidden_activation = np.tanh(hidden_input)

    output = sigmoid(
        hidden_activation @ weights_output + bias_output
    )

    loss = -np.mean(
        xor_targets * np.log(output + 1e-10)
        + (1 - xor_targets) * np.log(1 - output + 1e-10)
    )

    xor_history.append(loss)

    output_delta = (output - xor_targets) / len(xor_inputs)

    output_gradient = hidden_activation.T @ output_delta

    output_bias_gradient = output_delta.sum(
        axis=0, keepdims=True
    )

    hidden_delta = (
        output_delta @ weights_output.T
    ) * (1 - hidden_activation ** 2)

    hidden_gradient = xor_inputs.T @ hidden_delta

    hidden_bias_gradient = hidden_delta.sum(
        axis=0, keepdims=True
    )

    weights_hidden -= learning_rate * hidden_gradient
    bias_hidden -= learning_rate * hidden_bias_gradient

    weights_output -= learning_rate * output_gradient
    bias_output -= learning_rate * output_bias_gradient


final_hidden = np.tanh(
    xor_inputs @ weights_hidden + bias_hidden
)

final_probabilities = sigmoid(
    final_hidden @ weights_output + bias_output
)

print("\nMultilayer Network XOR Predictions:")

for inputs, probability in zip(
    xor_inputs.astype(int),
    final_probabilities.flatten()
):

    predicted_class = int(probability >= 0.5)

    print(
        f"Input: {inputs} | "
        f"Probability: {probability:.4f} | "
        f"Class: {predicted_class}"
    )

plt.figure(figsize=(8, 4))
plt.plot(xor_history)
plt.xlabel("Training Epochs")
plt.ylabel("Binary Cross-Entropy Loss")
plt.title("Learning Curve for XOR Neural Network")
plt.grid(True)
plt.tight_layout()
plt.show()
