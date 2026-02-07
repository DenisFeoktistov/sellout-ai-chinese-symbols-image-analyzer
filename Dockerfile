# Use the official Python image as the base image
FROM python:3.10-slim

# Set environment variables for FastAPI
ENV APP_HOME /app
ENV PORT 5000

# Create and set the working directory
WORKDIR $APP_HOME

COPY . $APP_HOME/
# Copy the requirements file into the container
COPY requirements.txt $APP_HOME/

RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 tesseract-ocr tesseract-ocr-chi-sim

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Expose the port on which the FastAPI application will run
EXPOSE $PORT

# Start the FastAPI application when the container is run
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
