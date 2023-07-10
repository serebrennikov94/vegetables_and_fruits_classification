from flask import Flask, render_template, request, session, flash, redirect
import os
from utils import predict, allowed_file
import torch 
import gdown 

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # base directory

file_id = '1FFOZhmdAivlRzkYEC1XDbwpBKJgNnYPX' # pretrained model in google drive
output = BASE_DIR + '/model.pth' # path for downloading model 
if os.path.exists(output): 
    print('Model exists!') 
else:
    print("Model doesn't exist. \nDownloading the model from google drive...")
    gdown.download(id=file_id, output=output, quiet=False) # start downloading model from google drive

device = 'cuda' if torch.cuda.is_available() else 'cpu' # set device 
model = torch.load(BASE_DIR + '/model.pth', map_location=device) # set pretrained model

app = Flask(__name__) # initialize flask app
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static/uploads') # folder for saving user images 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 
app.secret_key = 'xyz' 


@app.route('/')
def main():
    return render_template('index.html')


@app.route('/', methods=["POST"])
def upload_file():
    error = ''
    target_img = os.path.join(os.getcwd(), 'static/uploads') # directory for saving user images
    if request.method == 'POST': 
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file'] # get user image
        if file.filename == '': 
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename)) # save user image
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename) # path to user image
            img_show_path = os.path.join('/static/uploads', file.filename) 
            session['uploaded_img_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

            prob_result, class_result = predict(img_path, model) # get predictions 

            predictions = {
                "class1": class_result[0],
                "class2": class_result[1],
                "class3": class_result[2],
                "class4": class_result[3],
                "class5": class_result[4],
                "prob1": prob_result[0],
                "prob2": prob_result[1],
                "prob3": prob_result[2],
                "prob4": prob_result[3],
                "prob5": prob_result[4]
            }
        else:
            error = "Please upload images of jpg , jpeg and png extension only"

        if (len(error) == 0):
            return render_template('result.html',
                                   img=img_show_path,
                                   predictions=predictions,
                                   pred_image=class_result[0])
        else:
            return render_template('index.html', error=error)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
