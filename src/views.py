from flask import Flask, render_template, jsonify, request, Blueprint, make_response
import datetime
import os
from bokeh.plotting import figure, show
from bokeh.models import AjaxDataSource, CustomJS
from bokeh.embed import components
import numpy as np

from src.database import Database
from src.toolchain import GraphToolchain

opti_blueprint = Blueprint('optimization', __name__)





@opti_blueprint.route('/data', methods=['POST'])
def data():
    x.append(x[-1]+0.1)
    y.append(np.sin(x[-1])+np.random.random())
    z.append(np.cos(x[-1])+np.random.random())
    return jsonify(x=x, y=y, z=z)


@opti_blueprint.route('/server_data', methods=['POST'])
def server_load():
    t = [0] 
    x1,x2,x3 = [np.random.random()], [np.random.random()], [np.random.random()]
    return jsonify(t=t, x1=x1, x2=x2, x3=x3)

@opti_blueprint.route('/')
def show_dashboard():
    plots=[]
    plots.append(make_plot_ajax())
    plots.append(make_serverplot_ajax())

    return render_template('bokeh/plot.html', plots=plots)
