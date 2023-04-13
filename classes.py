from gym import Env
from gym.spaces import Box, Discrete
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common import env_checker
import time
import pydirectinput
import cv2
import pytesseract
from mss import mss
import numpy as np
import os

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# DIR
CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'
MODEL_DIR = '/models'
# paths
b1_path = os.getcwd() + '/static/images/box1.png'
d1_path = os.getcwd() + '/static/images/done1.png'
b2_path = os.getcwd() + '/static/images/box2.png'
d2_path = os.getcwd() + '/static/images/done2.png'
model_path = os.getcwd() + MODEL_DIR
folder_base_name = 'DNQ'
dir_mode = 0o666

# def get_all_window_titles():
#     return [win.title for win in pygetwindow.getAllWindows()]


def testing_model(env, model):
    for ep in range(5):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(int(action))
            time.sleep(0.01)
            total_reward += reward
        print(f'total reward for ep {ep} is {total_reward}')


def random_play(env):
    for ep in range(2):
            obs = env.reset()
            done = False
            total_reward = 0
            while not done:
                obs, reward, done, info = env.step(env.action_space.sample())
                total_reward += reward
            print(f'total reward for ep {ep} is {total_reward}')


class SaveOnBestTrainingRewardCallback(BaseCallback):

    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), "timesteps")
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose >= 1:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose >= 1:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True

# class TrainAndLoggingCallback(BaseCallback):
    
#     def __init__(self, check_freq, save_path, verbose=1):
#         super(TrainAndLoggingCallback, self).__init__(verbose)
#         self.check_freq = check_freq
#         self.save_path = save_path
        
#     def _init_callback(self):
#         if self.save_path is not None:
#             pass
#             os.makedirs(self.save_path, exist_ok=True)
            
#     def _on_step(self):
#         pass
#         if self.n_calls % self.check_freq == 0:
#             model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
#             self.model.save(model_path)


class game(Env):
    # set env action and obser shaps
    def __init__(self, gameloc = [300, 380, 750, 460], doneloc = [425, 650, 650, 70], actions = ['no_op','space', 'down'], rest_seq = [ 'click','up', 'up', 'click', 'space'], neutral_click_pos = [150, 250]):
        # use the base class
        super().__init__()
        # setup spaces
        self.observation_space = Box(low=0, high=255, shape=(1,83,100), dtype=np.uint8)
        self.action_space = Discrete(3)
        # def extraction parameters
        self.cap = mss()
        self.game_location = {'top':gameloc[0], 'left':gameloc[1], 'width':gameloc[2], 'height':gameloc[3]}
        self.done_location = {'top':doneloc[0], 'left':doneloc[1], 'width':doneloc[2], 'height':doneloc[3]}
        # create action dic
        if 'no_op' not in actions:
            actions.insert(0, 'no_op')
        else:
            actions.insert(0, actions.pop(actions.index('no_op')))
        self.action_map = {v: k for v, k in enumerate(actions)}
        # set rest_seq
        self.rest_seq = rest_seq
        self.neutral_click_pos = neutral_click_pos


    # called to do somehitng in the game     
    def step(self, action):
        
        # actions: 0 - space(jump), 1 - down(duck), 2 - no action(no op) 
        # action_map = {1: 'space', 2: 'down', 0: 'no_op'}

        # simulate press if not 
        if action != 0:
            pydirectinput.press(self.action_map[action])
    
        # check for gameover
        done, done_cap = self.get_done()
        # check next observ
        new_observ = self.get_observation()
        # reward - get point for each frame that isn't gameover
        reward = 1
        #info dic
        info = {}
        return new_observ, reward, done, info 
    
    # vusualize the game
    def render(self):
        cv2.imshow('Game', np.array(self.cap.grab(self.game_location))[:,:,:3])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.close()

    # closes the done 
    def close(self):
        cv2.destroyAllWindows()
        
    # restart the game     
    def reset(self):
        time.sleep(1)
        for action in self.rest_seq:
            if action == "click":
                pydirectinput.click(x=self.neutral_click_pos[0], y=self.neutral_click_pos[1])
            else:
                pydirectinput.press(action)
        time.sleep(2)
        return self.get_observation()

    # get a part of the observation that we want
    def get_observation(self):
        # get screen capture (only first 3 cahnnels) and extract the values from it
        raw = np.array(self.cap.grab(self.game_location))[:,:,:3]
        # grayscale
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        # resize
        resized = cv2.resize(gray, (100,83))
        # add channels first 
        channel = np.reshape(resized, (1, 83, 100))
        return channel
    
    # get done part of the observation
    def get_done_observation(self):
        raw = np.array(self.cap.grab(self.done_location))[:,:,:3]
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (100,83))
        channel = np.reshape(resized, (1, 83, 100))
        return channel 
    
    # get the done text with OCR
    def get_done(self):
        # get done screen
        done_cap = np.array(self.cap.grab(self.done_location))[:,:,:3]
        # valid done text
        done_str = ['GAME','GAHE']
        
        # Apply OCR
        # flag to recognize gameover
        done = False
        # get string
        res = pytesseract.image_to_string(done_cap)[:4]
        if res in done_str:
            # found gameover
            done = True
        # return game status
        return done, done_cap