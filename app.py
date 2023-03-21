from flask import Flask ,render_template, redirect, request, session, url_for, session
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import cv2
import classes as cl
from stable_baselines3 import DQN
from flask_executor import Executor


app = Flask(__name__)
app.secret_key = os.urandom(100)
executor = Executor(app)
app.config['EXECUTOR_TYPE'] = 'thread'
app.config['EXECUTOR_MAX_WORKERS'] = 5
env = None
model = None

callback = cl.TrainAndLoggingCallback(check_freq=500, save_path=cl.CHECKPOINT_DIR)

@app.route('/' ,methods=['GET','POST'])
def form_post():
    global env
    if request.method == 'POST':
        if request.form['submit_button'] == 'sub-bt':
        # if 'sub-bt' in request.form:
            box_top_pos_input = request.form["box-top-pos-input"]
            box_Left_input = request.form["box-left-input"]
            box_width_input = request.form["box-width-input"]
            box_hight_input = request.form["box-hight-input"]
            done_top_pos_input = request.form["done-top-pos-input"]
            done_Left_input = request.form["done-left-input"]
            done_width_input = request.form["done-width-input"]
            done_hight_input = request.form["done-hight-input"]
            session['box'] = [box_top_pos_input ,box_Left_input, box_width_input, box_hight_input]
            session['done'] = [done_top_pos_input ,done_Left_input, done_width_input, done_hight_input]
            # env = game(box, done)
            env = cl.game()
            plt.imshow(env.get_observation()[0])
            plt.savefig(cl.b1_path)
            plt.imshow(env.get_done_observation()[0])
            plt.savefig(cl.d1_path)
            plt.imshow(cv2.cvtColor(env.get_observation()[0], cv2.COLOR_BGR2RGB))
            plt.savefig(cl.b2_path)
            plt.imshow(cv2.cvtColor(env.get_done_observation()[0], cv2.COLOR_BGR2RGB))
            plt.savefig(cl.d2_path)
            return redirect(url_for('.sample')) 
            # return redirect('/sample.html', box=box, done=done) 
        else:

            all_windows = cl.get_all_window_titles()
            print(all)
            return render_template('/home.html', all_windows=all_windows)

        
    else: 
        return render_template('/home.html', all_windows=None)
    
    
@app.route('/sample' ,methods=['GET','POST'])
def sample():
    box = session.get('box')
    done = session.get('done')
    if request.method == 'POST':
        if request.form['submit_button'] == 'conti-bt':
            return redirect(url_for('.model_training'))
        else:
            return redirect(url_for('.form_post'))
    return render_template('/sample.html', box=box, done=done)


@app.route('/training' ,methods=['GET','POST'])
def model_training():
    global model
    if request.method == 'POST':
        # cl.random_play(env)
        model.learn(total_timesteps=5, callback=callback)

        return render_template('/testing.html')
    else:
        model = DQN('CnnPolicy', env, tensorboard_log=cl.LOG_DIR, verbose=1,
                    buffer_size= 150000, learning_starts= 100)
        return render_template('/training.html')
    


@app.route('/testining' ,methods=['GET','POST'])
def model_testing():
    global model
    if request.method == 'POST':
       cl.testing_model(env, model)
    return render_template('/testing.html')