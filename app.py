from flask import Flask ,render_template, redirect, request, session, url_for, session, send_file, flash, Response
import os, glob, re, cv2, matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
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
timesteps = 1

# callback = cl.TrainAndLoggingCallback(check_freq=10000, save_path=cl.CHECKPOINT_DIR)
callback = cl.SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=cl.CHECKPOINT_DIR)

@app.route('/' ,methods=['GET','POST'])
def form_post():
    global env
    global model
    if request.method == 'POST':
        env = cl.game()
        if request.form['submit_button'] == 'sub-bt':

            session['box'] = [int(request.form["box-top-pos-input"]) , int(request.form["box-left-input"]), int(request.form["box-width-input"]), int(request.form["box-hight-input"])]
            session['done'] = [int(request.form["done-top-pos-input"]) ,int(request.form["done-left-input"]), int(request.form["done-width-input"]), int(request.form["done-hight-input"])]
            keys_input = request.form["keys-input"].split(",")
            session['keys'] = [key for key in keys_input]
            session['res-seq'] = request.form["reset-input"].split(",")
            session['neut-pos'] = [int(request.form["x_coordinate"]), int(request.form["y_coordinate"])]
            plt.imshow(env.get_observation()[0])
            plt.savefig(cl.b1_path)
            plt.imshow(env.get_done_observation()[0])
            plt.savefig(cl.d1_path)
            plt.imshow(cv2.cvtColor(env.get_observation()[0], cv2.COLOR_BGR2RGB))
            plt.savefig(cl.b2_path)
            plt.imshow(cv2.cvtColor(env.get_done_observation()[0], cv2.COLOR_BGR2RGB))
            plt.savefig(cl.d2_path)
            env = cl.game(session['box'], session['done'], session['keys'], session['res-seq'], session['neut-pos']) 
            return redirect(url_for('.sample')) 
        
        elif request.form['submit_button'] == 'dino-bt':
            env = cl.game(gameloc = [300, 380, 750, 460], doneloc = [425, 650, 650, 70], actions = ['no_op','space', 'down'], rest_seq = [ 'click','up', 'up', 'click', 'space'], neutral_click_pos = [150, 250])
            return redirect(url_for('.model_training'))
        
        elif request.form['submit_button'] == 'upload-bt':
            # check if the post request has the file part
            if 'file' not in request.files:
                flash('No file part')
                return render_template('/home.html')
            file = request.files['file']
            # If the user does not select a file, the browser submits an empty file without a filename.
            if file.filename == '':
                flash('No selected file')
                return render_template('/home.html')
            print('this should save!')
            latest_model_number  = '1'
            file_name = os.path.join(cl.model_path, 'DQN1')
            if os.listdir(cl.model_path):
                latest_model_folder = max(glob.glob(cl.model_path + '/*'), key=os.path.getmtime)
                latest_model_numbers_list  = re.findall(r'\d+', latest_model_folder.split(cl.folder_base_name)[-1])
                latest_model_number = int(''.join(latest_model_numbers_list)) + 1
                file_name = os.path.join(cl.model_path, 'DQN' + str(latest_model_number))
            # file.save(os.path.join(cl.model_path, file_name))
            print(file_name + '.zip')
            file.save(file_name + '.zip')
            model = DQN.load(file_name, env=env)
            return redirect(url_for('.model_testing')) 
        
    else: 
        return render_template('/home.html')
    
    
@app.route('/sample' ,methods=['GET','POST'])
def sample():

    if request.method == 'POST':
        if request.form['submit_button'] == 'conti-bt':
            return redirect(url_for('.model_training'))
        else:
            return redirect(url_for('.form_post'))
    return render_template('/sample.html', box=session.get('box'), done=session.get('done'), keys=session.get('keys'), res_seq=session.get('res-seq'), neut_pos=session.get('neut-pos'))


@app.route('/training' ,methods=['GET','POST'])
def model_training():
    global model
    if request.method == 'POST':
        model.learn(total_timesteps=timesteps, callback=callback)

        if  not os.listdir(cl.model_path):
            model.save(os.path.join(cl.model_path, 'DQN1'))
        else:
            latest_model_folder = max(glob.glob(cl.model_path + '/*'), key=os.path.getmtime)
            latest_model_numbers_list  = re.findall(r'\d+', latest_model_folder.split(cl.folder_base_name)[-1])
            latest_model_number = int(''.join(latest_model_numbers_list)) + 1
            model.save(os.path.join(cl.model_path, 'DQN' + str(latest_model_number)))

        return redirect(url_for('model_testing'))
    else:
        model = DQN('CnnPolicy', env, tensorboard_log=cl.LOG_DIR, verbose=1, buffer_size= 150000, learning_starts= 100)
        return render_template('/training.html')
    


@app.route('/testining' ,methods=['GET','POST'])
def model_testing():
    global model
    if request.method == 'POST':
       
        if request.form['submit_button'] == 'redo-bt':
            return redirect(url_for('.form_post'))
        elif request.form['submit_button'] == 'test-bt':
            cl.testing_model(env, model)   
            return render_template('/testing.html')
        elif request.form['submit_button'] == 'downlad-bt"':
            pass
            
    return render_template('/testing.html')



@app.route('/download')
def download_model():
    file_path = max(glob.glob(cl.model_path + '/*'), key=os.path.getmtime)
    return send_file(file_path, as_attachment=True)