This repo contains implementations of reinforcement learning algorithms and tasks, primarily for my own learning. Code here is not production quality or even necessarily correct. Caveat emptor.

### Development

To set up a development environment:

```sh
# Install binary dependencies. TODO: flesh this out.
sudo apt-get install libglew2.1

git clone https://github.com/davmre/rl.git
cd rl

# Set up virtualenv, including postactivation hooks.
virtualenv .venv
echo source $PWD/postactivate.sh >> .venv/bin/activate
source .venv/bin/activate

# Install Python deps.
pip install -r requirements.txt
```