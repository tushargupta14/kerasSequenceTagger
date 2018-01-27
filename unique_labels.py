## Script computes the unique lables inthe dataset 
## Do not run this, check in the labels.json file
import cPickle as cp 
import json
def load_data(path_to_data = ""):

	print "Loading data"
	sentence_dict = cp.load(open(path_to_data+"labeld_data.pkl"))
	print "Done..."
	return sentence_dict

def get_labels(label_set,data_dict_l2): 

	sentence_labels = data_dict_l2['sentence_labels']

	for label in sentence_labels :

		label_set.add(label)



def unique_labels(sentence_dict):

	label_set = set()
	count = 0
	for k,v in sentence_dict.iteritems() :

		data_dict_l1 = v

		for k1,v1 in data_dict_l1.iteritems():

			data_dict_l2 = v1
			get_labels(label_set,data_dict_l2)

		count += 1
		print count


	## Creating a label dictioanry 

	label_dict = {}
	uid = 1
	for labels in label_set :

		label_dict[str(labels)] = uid
		uid+=1 

	return label_dict

if __name__ =="__main__" :

	sentence_dict = load_data("data/")

	label_dict = unique_labels(sentence_dict)

	print len(label_dict)
	json.dump(label_dict,open("labels.json","w+"))