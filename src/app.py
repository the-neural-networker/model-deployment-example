import os

from flask import Flask, render_template, request, redirect

from inference import get_prediction
import cv2

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return
        img_bytes = file.read()
        class_name= get_prediction(image_bytes=img_bytes)
        return render_template('result.html', class_name=class_name)
    return render_template('index.html')

# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    if 'Cache-Control' not in response.headers:
        response.headers['Cache-Control'] = 'no-store'
    return response


if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))