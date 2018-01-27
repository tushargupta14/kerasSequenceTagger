### Script to load the data and return the initial dictioanry 
## computes no. of unique labels
 
import sys
import cPickle as cp 
import json 
import re 

def load_data(path_to_data = ""):

	print "Loading data"
	sentence_dict = cp.load(open(path_to_data+"labeld_data.pkl"))
	label_dict = json.load(open(path_to_data + "labels.json"))
	return sentence_dict,label_dict



def pre_process(data_dict_l2, label_dict) :

	sentence_tokens = data_dict_l2["sentence_tokens"]
	sentence_labels = data_dict_l2["sentence_labels"]
	sentence_shape = data_dict_l2["sentence_shape"]

	if len(sentence_labels) != len(sentence_tokens) :

		print "Error"

	exclude_index = [i for i,item in enumerate(sentence_shape) if re.search('[^xd]',item)]

	feed_tokens = [sentence_tokens[i] for i,item in enumerate(sentence_tokens) if i not in exclude_index ]

	feed_sentence = " ".join(feed_tokens)

	input_data = [feed_tokens,feed_sentence]




	data_labels = [sentence_labels[i] for i,item in enumerate(sentence_tokens) if i not in exclude_index]

	embed_labels = [label_dict[token_label] for token_label in data_labels]

	return input_data, embed_labels


### Lable mappping {'B-exp','B-skill', 'I-exp', 'I-skill', 'O'}

def create_model_data(sentence_dict,label_dict) :

	data_X = {}
	data_Y = {}
	uid = 0

	for k,v in sentence_dict.iteritems() :

		data_dict_l1 = v

		for k1,v1 in data_dict_l1.iteritems():

			data_dict_l2 = v1

			input_data,output_data = pre_process(data_dict_l2,label_dict)

			data_X[uid] = input_data
			data_Y[uid] = output_data
			uid+=1

		print uid

	print "dumping data to dictionaries..."


	path_to_data = "data/" 
	with open(path_to_data + "sentence_X_dict.json", "wb+") as f :

		json.dump(data_X,f)

	with open(path_to_data + "sentence_Y_dict.json", "wb+") as f :

		json.dump(data_Y,f)
	print "Done."

def main() :

	path_to_data = "data/"
	sentence_dict , label_dict= load_data(path_to_data)

	create_model_data(sentence_dict,label_dict)


if __name__ == "__main__" :

	main()