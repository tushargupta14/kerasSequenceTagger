### Training model for the Sequence tagging task 
## No word embeddings have been previously trained
import keras 
import json 
from keras.models import Sequential 
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM 
from keras.layers.core import Dense,Dropout,Activation
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras import optimizers
import cPickle as cp 
import numpy as np 
import h5py


import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
set_session(tf.Session(config=config))
def load_data(path_to_data = "data/") :


	with open("data/train_batch_dict_new.json","rb+") as f :

		train_batch_dict_new = json.load(f)

	with open("data/test_batch_dict_new.json","rb+") as f :

		test_batch_dict_new = json.load(f)

	with open("data/validation_batch_dict_new.json","rb+") as f :

		validation_batch_dict_new = json.load(f)

	with open("data/embedding_dict.json","rb+") as f :

		embedding_dict = json.load(f)
	with open("data/sentence_Y_dict.json","rb+") as f :

		sentence_Y_dict = json.load(f)

	return train_batch_dict_new, test_batch_dict_new, validation_batch_dict_new, embedding_dict,sentence_Y_dict

def convert_to_onehot(batch_labels) :

	output_mat = []
	for sent_labels in batch_labels :
		temp_mat = []	
		for word_label in sent_labels :
			if word_label == -1 :

				temp = [0]*5
				temp_mat.append(temp)
				continue
			temp = [0]*5
			temp[word_label] = 1
			temp_mat.append(temp)

		output_mat.append(temp_mat)

	return np.asarray(output_mat)

def pad_zeros(ids,embedding_dict) :
	## return a numpy array of all the sentences in the batch padded with zeros for their embeddings 

	max_len = 0 
	for sent_id in ids :
		embedding = embedding_dict[sent_id]
		if len(embedding) > max_len :
			max_len = len(embedding)

	#print "max_len", max_len


	sent_mat = [embedding_dict[sent_id] for sent_id in ids]

	if len(np.array(sent_mat).shape) == 1 :
		return np.array([xi + [0]*(max_len - len(xi)) for xi in sent_mat]),max_len

	else :

		padded_mat = np.zeros((len(ids),max_len))

		sent_mat = np.array(sent_mat)
		for i in xrange(len(ids)) :
	 		padded_mat[i,:] = np.pad(sent_mat[i,:],(0,max_len - sent_mat[i,:].shape[0]),'constant',constant_values = (0))


		return padded_mat,max_len


def test_on_validation_data(model,sentence_Y_dict, validation_batch_dict_new,embedding_dict,use_whole = False) :

	batch_count = 0 
	validation_loss = 0 
	acc  = 0 
	for batch_n,sent_ids in validation_batch_dict_new.iteritems() :

		vList = []
		batch_count +=1 

		sents , max_len = pad_zeros(sent_ids,embedding_dict)

		labels = [sentence_Y_dict[item]+[-1]*(max_len - len(sentence_Y_dict[item])) for item in sent_ids] 
				
		labels = convert_to_onehot(labels)

		vList = model.test_on_batch(sents,labels)
		validation_loss+= vList[0]
		acc+=vList[1]

		if batch_count == 30 and use_whole == False :
			break 


	print "validation_loss :",validation_loss/batch_count
	acc = acc/batch_count 
	print "Accuracy :",acc
	return [validation_loss/batch_count,acc]

def train_model() :

	n_classes = 5 ;

	model = Sequential()

	n_vocab = 44203
	model.add(Embedding(n_vocab,300,mask_zero = True)) #mask_zero = False 

	# n_vocab 
	# 200 dimension embedding 
 	# increasing dropout 
	model.add(Dropout(0.30))

	model.add(Bidirectional(LSTM(200,activation = 'tanh', return_sequences = True)))

	model.add(TimeDistributed(Dense(n_classes,activation = 'softmax')))

	rmsprop = optimizers.RMSprop(lr = 0.0000001)

	model.compile(optimizer = 'rmsprop' , loss = 'categorical_crossentropy',metrics = ['binary_accuracy'])

	## Loading data 

	train_batch_dict_new, test_batch_dict_new, validation_batch_dict_new, embedding_dict ,sentence_Y_dict = load_data("data/")

	epochs = 10

	avgLoss_list = []
	vLoss_list = []
	score_list = []

	print model.metrics_names

	for i in xrange(epochs) :

		print "Training epoch: ",i

		avgLoss = 0 
		batch_count = 0 
		for batch,sent_ids in train_batch_dict_new.iteritems() :

			batch_count+= 1
			print "batch:",batch_count

			sents , max_len = pad_zeros(sent_ids,embedding_dict)

			labels = [sentence_Y_dict[item]+[-1]*(max_len - len(sentence_Y_dict[item])) for item in sent_ids] 
			
			labels = convert_to_onehot(labels)

			#print sents.shape,labels.shape

			#if sents.shape[1] > 5 :
			scores = model.train_on_batch(sents, labels)
		
			avgLoss +=scores[0]
			

			if batch_count%10 ==0:
				print "avgLoss :",avgLoss/batch_count
				vLoss = test_on_validation_data(model,sentence_Y_dict,validation_batch_dict_new,embedding_dict) 
					
		avgLoss = avgLoss / len(train_batch_dict_new)

		avgLoss_list.append(avgLoss)

		## Evaluating the loss on the whole validation data 
		vLoss = test_on_validation_data(model,sentence_Y_dict,validation_batch_dict_new,embedding_dict,use_whole = True)
		vLoss_list.append(vLoss)
	
		print "epoch : ",i," avgLoss : ",avgLoss,"validation_loss :",vLoss[0],"Accuracy :",vLoss[1]
		score_list.append(scores)
		## Saving the model weights and architecture for each epoch 
		json_object = model.to_json()
		with open("weights/model1_epoch_"+str(i)+".json","wb+") as json_file :
			json_file.write(json_object)
		model.save_weights('weights/model1_weights_epoch_'+str(i)+'.h5')


		with open("results/avgLoss_list.pkl","wb+") as f:
			cp.dump(avgLoss_list,f)

		with open("results/score_list.pkl","wb+") as f:
			cp.dump(score_list,f)
		with open("results/vLoss_list.pkl","wb+") as f:
			cp.dump(vLoss_list,f)


		## else if vLoss is less than the avgLoss keep training 



if __name__ == "__main__" :

	train_model()
