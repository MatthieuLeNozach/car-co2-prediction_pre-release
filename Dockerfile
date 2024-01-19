FROM continuumio/miniconda3

# Set the working directory in the container to /app
WORKDIR /app

# Copy the environment.yml file into the Docker image
COPY environment.yml .

# Add the current directory contents into the container at /app
ADD . /app

# Create a conda environment from the environment.yml file
RUN conda env create -f environment.yml

# Install SQLite CLI
RUN apt-get update && apt-get install -y sqlite3

# Install the Python package
RUN conda run -n co2 pip install .

EXPOSE 80

# Install pandas using conda in the co2 environment
RUN conda run -n co2 conda install -y pandas

# Run the application when the Docker container starts
CMD conda run -n co2 python launcher.py