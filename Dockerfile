FROM tensorflow/tensorflow:2.5.0

# Install PiNN
COPY . /opt/src/pinn
RUN pip install --upgrade pip && pip install /opt/src/pinn

# Setup
ENTRYPOINT ["pinn"]
