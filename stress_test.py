import time
import joblib
import numpy as np

# تحميل الموديل
tfidf = joblib.load("baseline_tfidf.pkl")
model = joblib.load("baseline_model.pkl")

def run_test(text, runs):
    inference_times = []

    for i in range(runs):

        start = time.perf_counter()

        vec = tfidf.transform([text])
        model.predict(vec)

        end = time.perf_counter()

        inference_times.append(end - start)

    print("\n===== LOCAL LOAD/STRESS TEST RESULTS =====")
    print("Requests:", runs)
    print("Average Time:", np.mean(inference_times))
    print("Max Time:", np.max(inference_times))
    print("Min Time:", np.min(inference_times))
    print("Std Dev:", np.std(inference_times))
    print("Throughput (req/sec):", 1/np.mean(inference_times))



run_test("This is a fake news example", 10)   # Load test(10)
                                              # Stress test(50)
                                              # Peak test(100)