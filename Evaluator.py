import gym
import cv2
import numpy as np
import pickle as pkl


cliffEnv = gym.make("CliffWalking-v0")


q_table = pkl.load(open("sarsa_q_table.pkl", "rb"))

def initialize_frame():
    width, height = 600, 200
    img = np.ones(shape=(height, width, 3)) * 255.0
    margin_horizontal = 6
    margin_vertical = 2


    for i in range(13):
        img = cv2.line(img, (49 * i + margin_horizontal, margin_vertical),
                       (49 * i + margin_horizontal, 400 - margin_vertical), color=(0, 0, 0), thickness=1)


    for i in range(5):
        img = cv2.line(img, (margin_horizontal, 49 * i + margin_vertical),
                       (600 - margin_horizontal, 49 * i + margin_vertical), color=(0, 0, 0), thickness=1)


    img = cv2.rectangle(img, (49 * 1 + margin_horizontal + 2, 49 * 3 + margin_vertical + 2),
                        (49 * 11 + margin_horizontal - 2, 49 * 4 + margin_vertical - 2), color=(255, 0, 255),
                        thickness=-1)
    img = cv2.putText(img, text="Cliff", org=(49 * 5 + margin_horizontal, 49 * 4 + margin_vertical - 10),
                      fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)


    frame = cv2.putText(img, text="G", org=(49 * 11 + margin_horizontal + 10, 49 * 4 + margin_vertical - 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)

    return frame

def put_agent(img, state):
    margin_horizontal = 6
    margin_vertical = 2
    if isinstance(state, tuple):
        state = state[0]
    row, column = np.unravel_index(indices=state, shape=(4, 12))
    cv2.putText(img, text="A", org=(49 * column + margin_horizontal + 10, 49 * (row + 1) + margin_vertical - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
    return img


def policy(state, epsilon=0.1):
    if isinstance(state, int):
        state_index = state
    elif isinstance(state, tuple):
        state_index = state[0]
    else:
        raise ValueError("Invalid state type: must be an integer or a tuple")

    action = int(np.argmax(q_table[state_index]))
    if np.random.random() <= epsilon:
        action = int(np.random.randint(low=0, high=4, size=1))

    return action


def render_frame(state):
    frame = initialize_frame()
    frame = put_agent(frame, state)
    cv2.imshow("Cliff Walking", frame)
    cv2.waitKey(100)


NUM_EPISODES = 5
for episode in range(NUM_EPISODES):
    done = False
    total_reward = 0
    episode_length = 0
    state = cliffEnv.reset()
    render_frame(state)

    while not done:
        action = policy(state, epsilon=0.1)
        next_state, reward, done = cliffEnv.step(action)[:3]
        total_reward += reward
        episode_length += 1
        render_frame(next_state)
        state = next_state

    print("Episode:", episode, "Length:", episode_length, "Reward:", total_reward)

cv2.destroyAllWindows()
cliffEnv.close()
