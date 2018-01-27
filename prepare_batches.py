# Script for preparing batches of sentences and splitting into training
import json
import random 
from collections import defaultdict
def load_data(path_to_data = "data/") :

	with open(path_to_data+"embedding_dict.json") as f :

		embedding_dict = json.load(f)

	return embedding_dict


def create_batches(embedding_dict) :

	## Indexed by the length of sentences 
	training_size = int (0.8 * len(embedding_dict))

	testing_size = len(embedding_dict) - training_size

	n_sentences= len(embedding_dict)
	random.seed(0)

	embedding_batch_dict = {}

	random_ids_training = random.sample(xrange(len(embedding_dict)),len(embedding_dict))

	## training data
	train_batch_dict = defaultdict(list)
	test_batch_dict = defaultdict(list)

	for counter in xrange(training_size) :

		i = random_ids_training[counter]

		sent_embedding = embedding_dict[str(i)]

		length = len(sent_embedding)
<<<<<<< HEAD
		if length > 5 :
			train_batch_dict[length].append([str(i),sent_embedding])
=======

		train_batch_dict[length].append([str(i),sent_embedding])
>>>>>>> 4a675dd6c1a8ce59aaa6319b15308dc1bd12c717

		print counter,i,training_size
	## Testing data

	print "preparing testing data" 
	for counter in xrange(training_size,n_sentences) :

		i = random_ids_training[counter]

<<<<<<< HEAD
		if len(embedding_dict[str(i)]) > 5 :

			test_batch_dict[len(embedding_dict[str(i)])].append([str(i),embedding_dict[str(i)]])
=======
		test_batch_dict[len(embedding_dict[str(i)])].append([str(i),embedding_dict[str(i)]])
>>>>>>> 4a675dd6c1a8ce59aaa6319b15308dc1bd12c717

		print counter,i



	with open("data/train_batch_dict.json","wb+") as f :

		json.dump(train_batch_dict,f)

	with open("data/test_batch_dict.json","wb+") as f :

		json.dump(test_batch_dict,f)


if __name__ == "__main__" :

	embedding_dict = load_data()

	create_batches(embedding_dict)



	