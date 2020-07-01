# -*- coding: utf-8 -*-
"""
Created on Jul 01, 2020
@author: yongzhengxin
"""

import os
from app import app
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from transfer import load_image, viz_color_palette, get_illuminance, FE, RD, device

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def upload_form():
    return render_template('upload.html')


@app.route('/', methods=['POST'])
def upload_image():
    print(request.form)
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed')

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # load inputs for model (FE-RD)
        img = load_image(filepath)
        plt.imsave(f'static/processed/{filename}_input.png', np.transpose(img[0].detach().cpu().numpy(), (1, 2, 0)))

        palette = viz_color_palette([request.form['fav-color-1'],
                                     request.form['fav-color-2'],
                                     request.form['fav-color-3'],
                                     request.form['fav-color-4'],
                                     request.form['fav-color-5'],
                                     request.form['fav-color-6']])
        plt.imsave(f'static/processed/{filename}_new_palette.png', palette.detach().cpu().numpy().reshape((1, 6, 3)))

        illu = get_illuminance(img[0])

        # pass through model (FE-RD)
        c1, c2, c3, c4 = FE.forward(img.float().to(device))
        out = RD.forward(c1, c2, c3, c4, palette.float().to(device), illu.float().to(device))
        out = out[0].detach().cpu().numpy()

        # save output
        plt.imsave(f'static/processed/{filename}_result.png', np.clip(np.transpose(out, (1, 2, 0)), 0, 1))
        # plt.imsave(f'static/processed/{filename}_result.png', (np.transpose(out, (1, 2, 0)) * 255).astype(np.uint8))
        return render_template('load_result.html',
                               input_filename=f"{filename}_input.png",
                               palette_filename=f"{filename}_new_palette.png",
                               result_filename=f"{filename}_result.png")

    else:
        flash('Allowed image types are -> png, jpg, jpeg')
        return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
    # print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='processed/' + filename), code=301)


if __name__ == "__main__":
    app.run()