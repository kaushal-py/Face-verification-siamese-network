from flask import Flask, render_template, request
import os

app = Flask(__name__)

UPLOAD_FOLDER = os.path.basename('upload')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def hello():
    return render_template('/display.html')

@app.route('/upload', methods=['POST'])
def upload_pre():
    pre = request.files['pre']
    post = request.files['post']
    pref = os.path.join(app.config['UPLOAD_FOLDER'], "PRE")
    postf = os.path.join(app.config['UPLOAD_FOLDER'], "POST")
    
    # add your custom code to check that the uploaded file is a valid image and not a malicious file (out-of-scope for this post)
    pre.save(pref)
    post.save(postf)

    return render_template('/display.html')

if __name__ == "__main__":
    app.run()