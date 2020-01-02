from bokeh.plotting import figure, show, output_file
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))

from src.database import Database
from src.optimizer import Optimizer


xlabels, ylabels, clabels = Optimizer.splitColumnNames()
iters, X, Y, C, P = Optimizer.postprocessReturnAll()
iters_distinct, data = Optimizer.postprocessReturnStatistics()

TOOLS="hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,"

p = figure(tools=TOOLS)
p.scatter(X[:,0], X[:,1], radius=0.1, fill_alpha=0.6, line_color=None) # ,fill_color=colors
output_file("color_scatter.html", title="color_scatter.py example")

show(p) 

