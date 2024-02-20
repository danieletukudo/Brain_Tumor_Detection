# Use an official Python runtime as a parent image
FROM python:3.9.11

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file into the container and install the dependencies

COPY requirements.txt /app
RUN apt-get update && apt-get install -y libgl1-mesa-glx

RUN apt-get update && apt-get install -y cmake
RUN pip install --no-cache-dir -r requirements.txt
# Copy the current directory contents into the container at /app
COPY . /app

# Expose port 5000 for the Flask application
EXPOSE 7017

# Set the environment variable for Flask to run in production mode
ENV FLASK_ENV=production

# Start the Flask application
ENV FLASK_APP=app.py
CMD ["flask", "run", "--host", "0.0.0.0","--port=7017"]

