## Script to convert the sentences into embedding vectors for input into the model 


import json 

def load_data(path_to_data = "") :

	with open(path_to_data + "sentence_X_dict.json","rb+") as f : 
		sentence_X_dict = json.load(f)

	with open(path_to_data + "word_vocab.json","rb+") as f:
		word_vocab_dict = json.load(f)

	return sentence_X_dict,word_vocab_dict


def get_embedding(tokens,word_vocab_dict): 

	return [word_vocab_dict[token] for token in tokens]		

def embed_sentences(path_to_data, sentence_X_dict,word_vocab_dict) :

	embedding_dict = {}
	embedding_dict = {k:get_embedding(v[0],word_vocab_dict) for k,v in sentence_X_dict.iteritems()}

	print "Saving the dictioanry..."
	with open(path_to_data + "embedding_dict.json","wb+") as f :

		json.dump(embedding_dict,f)

if __name__ == "__main__" :

	sentence_X_dict,word_vocab_dict = load_data("data/")

	embed_sentences("data/", sentence_X_dict,word_vocab_dict)
