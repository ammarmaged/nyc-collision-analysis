# Use Python 3.9 as the base image
FROM python:3.9

# Set the working directory inside the container to /code
WORKDIR /code

# Copy the requirements file first (for better caching)
COPY ./requirements.txt /code/requirements.txt

# Install the dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy all the rest of your app files into the container
COPY . .

# Expose port 7860 (Required by Hugging Face Spaces)
EXPOSE 7860

# Command to run the app using Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:server"]