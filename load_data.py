### Script to load the data and return the initial dictioanry 

import sys

import cPickle as cp 


def load_data(path_to_data):

	sentence_dict = cp.load(open(path_to_data+"labeld_data.pkl"))

	return sentence_dict