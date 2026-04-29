from dqn import DQNetwork
from env import SumoIntersection
import torch
import yaml

with open("config.yml") as f:
    config = yaml.safe_load(f)

sumoBinary = config["sumoBinary"]
sumoCmd = config["sumoCmd"]
WEIGHTS_PATH = config["weights_path"]
SIM_LEN = config["sim_len"]
N_CARS = config["n_cars"]

if __name__ == '__main__':
    q = DQNetwork()
    q.eval()
    try:
        q.load_state_dict(torch.load(WEIGHTS_PATH, map_location='cpu'))
        print(f"Loaded weights from: {WEIGHTS_PATH}")
    except FileNotFoundError:
        print(f"No model weights found at: {WEIGHTS_PATH}")

    env = SumoIntersection(sumoBinary, sumoCmd, SIM_LEN, N_CARS)

    state, _, _, _ = env.step(0)
    done = False

    while not done:
        a = q.predict(state.as_tuple, 0.00)
        s_prime, r, done, info = env.step(a)
        print(r)
        state = s_prime