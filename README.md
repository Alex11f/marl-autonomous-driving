# Autonomous Driving Project - Distributed AI

## Setup Instructions

### 1. Install System Dependencies
Run the following commands in your terminal to install the required system libraries for Pygame and Highway-env:

```bash
sudo apt-get update
sudo apt-get install -y python3-dev python3-pip python3-setuptools \
    libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev \
    libfreetype6-dev libportmidi-dev libjpeg-dev build-essential
```

### 2. Set up Python Environment
Create a virtual environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Project Structure
- `src/`: Source code for agents, environments, and utilities.
  - `highway_env/`: using Highway-env for simulation and SB3 for reinforcement learning.
  - `metadrive_env/`: using MetaDrive for more complex driving scenarios and RLlib for training.
- `results/`: Output directory for logs and models.
