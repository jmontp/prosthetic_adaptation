"""
This file is meant to test loading personalized model objects in runtime
"""


from context import model_fitting
from model_fitting.load_models import load_personalized_models

#Define a list of models that you want to create
joint_models = ['jointangles_foot_x', 
                'jointangles_thigh_x', 
                'jointangles_shank_x']

subject_name = "AB01"

load_personalized_models(joint_models, subject_name)