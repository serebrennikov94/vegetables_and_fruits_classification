from flask import Flask, render_template, request, session, flash, redirect
import os
from utils import predict, allowed_file
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = torch.load(BASE_DIR + '/model.pth', map_location='cpu')

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static/uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'xyz'


@app.route('/')
def main():
    return render_template('index.html')


@app.route('/', methods=["POST"])
def upload_file():
    error = ''
    target_img = os.path.join(os.getcwd(), 'static/uploads')
    print(target_img)
    if request.method == 'POST':
        print(1)
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        print(2)
        file = request.files['file']
        # file = request.form.get("file", False)
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            print('ok')
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            img_show_path = os.path.join('/static/uploads', file.filename)
            session['uploaded_img_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

            prob_result, class_result = predict(img_path, model)

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
    # app.run(debug=True)
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
