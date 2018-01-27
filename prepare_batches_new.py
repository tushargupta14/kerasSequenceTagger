## Prepares batches of the same length but variable sentence length 
## finally dictioanry is created to key : batch_no value : list of sentence ids 

import json 
import random 
from collections import defaultdict
import cPickle as cp 


def load_data(path_to_data = "data/") :

	with open(path_to_data+"embedding_dict.json") as f :

		embedding_dict = json.load(f)

	return embedding_dict


def create_partitions():

	embedding_dict = load_data("data/")

	length_wise_dict = defaultdict(list)

	count = 0 
	for k,v in embedding_dict.iteritems() :

		if len(v) > 4:
			length_wise_dict[int(len(v))].append(k)
		count+=1 
		print count 

	with open("data/length_id_dict.json","wb+") as f:

		json.dump(length_wise_dict,f)

def break_data_set() :

	print "Breaking data set into train: 80,validation : 10 ; test :10 "
	length_id_dict = json.load(open("data/length_id_dict.json"))

	key_list = sorted([int(key) for key in length_id_dict.keys()])

	## Sorted list of keys

	training_ids =[]
	validation_ids = []
	testing_ids = []
	#training_dict = defaultdict(list)
	#validation_dict = defaultdict(list)
	#testing_dict = defaultdict(list)

	count = 0
	for key in key_list :

		id_list = length_id_dict[str(key)]
		len_id_list = len(id_list)

		training_size = int(0.8*len_id_list)

		validation_size = int(0.1*len_id_list)

		testing_size = len_id_list - training_size  -validation_size
		if training_size is not 0 :

			training_ids+=id_list[:training_size]
		if validation_size is not 0 :

			validation_ids+=id_list[training_size:training_size+validation_size]

		if testing_size is not 0 :

			testing_ids+=id_list[training_size+validation_size:]

		#count+=1
		#print count

	with open("data/training_ids.pkl","wb+") as f :

		cp.dump(training_ids,f)

	with open("data/testing_ids_ids.pkl","wb+") as f :

		cp.dump(testing_ids,f)


	with open("data/validation_ids.pkl","wb+") as f :

		cp.dump(validation_ids,f)

	print "At the end"

	return training_ids,testing_ids,validation_ids


def prepare_batches(training_ids,testing_ids,validation_ids):


	## creates batches of data of similar length of size batch size
	print "Here"
	batch_size = 64

	train_batch_dict_new = defaultdict(list)
	test_batch_dict_new = defaultdict(list)
	validation_batch_dict_new = defaultdict(list)

	batch_no = 1
	items_in_batch = 0
	for ids in training_ids :

		if items_in_batch < batch_size :

			train_batch_dict_new[batch_no.append(ids)
			items_in_batch+=1

		else :

			batch_no+=1
			#print batch_no
			items_in_batch = 0 

	batch_no = 1
	items_in_batch = 0
	for ids in testing_ids :

		if items_in_batch < batch_size :

			test_batch_dict_new[batch_no].append(ids)
			items_in_batch+=1

		else :

			batch_no+=1
			#print batch_no
			items_in_batch = 0 


	batch_no = 1
	items_in_batch = 0
	for ids in validation_ids :

		if items_in_batch < batch_size :

			validation_batch_dict_new[batch_no].append(ids)
			items_in_batch+=1

		else :

			batch_no+=1
			#print batch_no
			items_in_batch = 0 


	with open("data/train_batch_dict_new.json","wb+") as f :

		json.dump(train_batch_dict_new,f)

	with open("data/test_batch_dict_new.json","wb+") as f :

		json.dump(test_batch_dict_new,f)

	with open("data/validation_batch_dict_new.json","wb+") as f :

		json.dump(validation_batch_dict_new,f)







if __name__ == "__main__" :

	#create_partitions()

	## Key is the lenght of the sentence 

	training_ids,testing_ids,validation_ids = break_data_set()
	prepare_batches(training_ids,testing_ids,validation_ids)