# Use an official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.8-slim

# Install manually all the missing libraries
RUN apt-get update && apt-get install -y \
build-essential \
&& rm -rf /var/lib/apt/lists/*

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Install production dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Run the web service on container startup.
CMD exec streamlit run --server.port $PORT --server.enableCORS false app.py
