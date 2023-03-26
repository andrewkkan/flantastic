FROM continuumio/miniconda3

WORKDIR /app

# Create the environment:
# COPY environment.yml .
# RUN conda env create -f environment.yml

COPY . /app
ENV CONDA_DEFAULT_ENV $(basename "$PWD")

# Note: . is similar if not exactly as the source cmd
RUN conda init bash && \
    . /root/.bashrc && \
    conda update conda && \
    conda create -n $CONDA_DEFAULT_ENV python=3.9 && \
    conda activate $CONDA_DEFAULT_ENV && \
    conda install pip git && \
    pip install -r /app/requirements.txt

# The below 2 lines came from the first snippet in https://medium.com/@chadlagore/conda-environments-with-docker-82cdc9d25754
# It resolves the problem with the container not initialized within the conda env.
# For custom env name, see the second snippet from the above link.
RUN echo "source activate $CONDA_DEFAULT_ENV" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH

