FROM continuumio/miniconda3

# Set the working directory in the container to /app
WORKDIR /app

# Copy the environment.yml file into the Docker image
COPY environment.yml .

# Add the current directory contents into the container at /app
ADD . /app

# Create a conda environment from the environment.yml file
RUN conda env create -f environment.yml

# Install SQLite CLI and wget
RUN apt-get update && apt-get install -y sqlite3 wget

# Install the Python package
RUN conda run -n co2 pip install .

EXPOSE 80

# Install pandas using conda in the co2 environment
RUN conda run -n co2 conda install -y pandas

# Start a shell in the co2 environment when the container launches
CMD ["conda", "run", "-n", "co2", "python", "launcher.py"]