## Script creates a vocabulary 

import json 


def load_data(path_to_data = "") :

	with open(path_to_data + "sentence_X_dict.json","rb+") as f : 
		sentence_X_dict = json.load(f)


	return sentence_X_dict


def create_vocabulary(sentence_X_dict):

	word_vocab_dict = {}
	uid = 1
	count = 0 
	for k,v in sentence_X_dict.iteritems() : 

		sentence_tokens = sentence_X_dict[k][0]
		
		for token in sentence_tokens :

			if token not in word_vocab_dict :
				word_vocab_dict[token] = uid	
				uid+=1 
		count+=1 
		print count, len(sentence_X_dict)


	with open("data/word_vocab.json","wb+") as f :

		json.dump(word_vocab_dict,f)


if __name__ == "__main__" :

	sentence_X_dict = load_data("data/")
	create_vocabulary(sentence_X_dict)

