# Use the official Python 3.12 slim image as the base.
FROM python:3.12-slim

# Set the working directory inside the container.
WORKDIR /app

# Copy the requirements file and install Python dependencies.
#RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code.
COPY . .

# Optionally expose a port if your app runs a web server (e.g., 8080).
EXPOSE 8080

# Define the default command to run your app.
CMD ["python", "demo.py"]
