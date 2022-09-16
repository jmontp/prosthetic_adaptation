"""
This file is meant to unit test the code in load_models.py
"""

from context import model_fitting
from model_fitting.load_models import load_simple_models

#Unit test 1 - load joint moment for AB01
test_model = load_simple_models("jointmoment_ankle_x", "AB01")