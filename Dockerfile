FROM python:3.10-slim

#folder inside the container called /app and switch into it
WORKDIR /app

#copy API script and trained model into the container
COPY main.py .
COPY iris_model.pkl .

#install the exact libraries needed to run the application
RUN pip install --no-cache-dir fastapi uvicorn pydantic scikit-learn pandas joblib

#tell the container to open up the port 8000 to accept internet traffic
EXPOSE 8000

#exact command to run the api when the container boots up
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8000"]
