# -*- coding: utf-8 -*-
from . import helpers
import os

def get_hmm():
    """Get a thought."""
    return str(os.getcwd())


def hmm():
    """Contemplation..."""
    if helpers.get_answer():
        print(get_hmm())
