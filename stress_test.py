import time
import joblib
import numpy as np

# Load trained TF-IDF vectorizer and classification model
tfidf = joblib.load("baseline_tfidf.pkl")
model = joblib.load("baseline_model.pkl")

def run_test(text, runs):
    # List to store inference time for each request
    inference_times = []

    for i in range(runs):

        # Start high-precision timer for inference measurement
        start = time.perf_counter()

        # Transform input text into TF-IDF feature vector
        vec = tfidf.transform([text])

        # Perform model prediction (fake/real classification)
        model.predict(vec)

        # End timer after inference completes
        end = time.perf_counter()

        # Store execution time for this request
        inference_times.append(end - start)

    # Print performance analysis results
    print("\n===== LOCAL LOAD/STRESS TEST RESULTS =====")

    # Number of simulated requests
    print("Requests:", runs)

    # Average inference time across all requests
    print("Average Time:", np.mean(inference_times))

    # Maximum observed inference time
    print("Max Time:", np.max(inference_times))

    # Minimum observed inference time
    print("Min Time:", np.min(inference_times))

    # Standard deviation (measures stability of system performance)
    print("Std Dev:", np.std(inference_times))

    # Throughput: number of requests processed per second
    print("Throughput (req/sec):", 1 / np.mean(inference_times))


# Run experiments with different load scenarios:
# 10 requests → Load test (low traffic simulation)
# 50 requests → Stress test (medium load simulation)
# 100 requests → Peak load test (high load simulation)
run_test("This is a fake news example", 10)