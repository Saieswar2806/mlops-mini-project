🌸 End-to-End MLOps Pipeline: Iris Species Predictor
This repository contains a complete, production-ready Machine Learning pipeline. It demonstrates
the full lifecycle of an ML model: training, tracking, serving, containerization, and automated CI/CD
deployment.
🏗 System Architecture
1. Model Training & Tracking (MLflow)
Trains a Random Forest Classifier on the classic Iris dataset.
Integrates MLflow to automatically log hyperparameters ( n_estimators , max_depth ),
evaluation metrics ( accuracy ), and version-control the serialized .pkl model.
2. Model Serving & Monitoring (FastAPI + Prometheus)
Wraps the trained model in a high-performance FastAPI web server.
Exposes a /predict endpoint that accepts JSON payloads with flower measurements
and returns real-time predictions.
Includes built-in system monitoring via Prometheus (tracking request volume, latency,
and error rates at the /metrics endpoint).
3. Containerization (Docker)
Packages the Python environment, dependencies, model, and API into a lightweight,
isolated Docker container using a Linux base image, ensuring consistent execution across
any environment.
4. Continuous Integration (GitHub Actions)
Automated CI/CD pipeline triggered on every push to the main branch.
Spins up an ephemeral Ubuntu runner to install dependencies, execute the training script,
and dry-run the Docker build to prevent broken code from reaching production.
5. Continuous Deployment (Render)
Connected directly to this GitHub repository for automatic deployment.
Detects successful commits, builds the Docker image in the cloud, and deploys the live
API to a public URL.
🚀 Live Demo
You can test the live, deployed model API here:
Interactive UI:
https://iris-mlops-api.onrender.com/docs
Metrics Dashboard:
https://iris-mlops-api.onrender.com/metrics
Example Prediction Payload:
{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}
💻 Local Setup
If you want to run this pipeline on your own machine:
1. Clone the repository:
git clone https://github.com/Saieswar2806/mlops-mini-project.git
cd mlops-mini-project
2. Run the API via Docker:
docker build -t iris-api .
docker run -p 8000:8000 iris-api
3. Access the local endpoints:
Interactive Docs: http://localhost:8000/docs
Prometheus Metrics: http://localhost:8000/metrics