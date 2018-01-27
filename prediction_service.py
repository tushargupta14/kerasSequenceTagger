
import requests
import json 
import keras 
import h5py
import json 
import cPickle as cp 
import numpy as np 
from keras.models import model_from_json
from flask import Flask, request, abort,jsonify, request
import re 


def clean_jd(jd_text):
    
    cleaned_line = (" ").join(x.lower() for x in jd_text.split())
    cleaned_line = (" ").join(re.sub("[^a-zA-Z0-9]","",x) for x in cleaned_line.split())
    return cleaned_line.split()

def get_embedidngs(tokens) :
    
    with open("data/corpus_word_vocab.json","rb+") as f :
        corpus_word_vocab = json.load(f)
    with open("data/labels.json","rb+") as f :
        label_dict = json.load(f)
        
    #ilabel_dict = { idx:label for label,idx in label_dict.iteritems()}
    #ivocab = { idx:word for word,idx in corpus_word_vocab.iteritems()}
    
    embedding_list = [int(corpus_word_vocab[token]) for token in tokens if token in corpus_word_vocab]
    
    return np.array(embedding_list).reshape(1,-1)

def get_labels(predictions,word_indexes):
    
    with open("data/corpus_word_vocab.json","rb+") as f :
        corpus_word_vocab = json.load(f)
    with open("data/labels.json","rb+") as f :
        label_dict = json.load(f)
        
    ilabel_dict = { idx:label for label,idx in label_dict.iteritems()}
    ivocab = { idx:word for word,idx in corpus_word_vocab.iteritems()}
    
    all_pred_list = []
    for i in range(word_indexes.shape[1]) :
        
        prediction_scores = list(predictions[i,:])
        #print prediction_scores
        
        max_prob_score = max(prediction_scores)
        label_idx = prediction_scores.index(max(prediction_scores))
        
        label = ilabel_dict[label_idx]
        word = ivocab[word_indexes[0,i]]
        
        all_pred_list.append([word,label])
            
    return all_pred_list

def get_predicted_skills(model,jd_text = ""):
    
    jd_tokens = clean_jd(jd_text)
    word_indexes = get_embedidngs(jd_tokens)
    #print word_indexes
    
    predictions = model.predict(word_indexes)
    
    #print predictions.shape
        
    all_pred_list = get_labels(predictions.reshape(-1,5),word_indexes)    
   
    #print len(all_pred_list)
    skill_list  = []
    i=0
    while i < len(all_pred_list)-1:
        
        if "O" in all_pred_list[i][1] :
            i+=1
            continue
        else :
            if "I" in all_pred_list[i+1][1] :
                skill_list.append([all_pred_list[i][0]+" "+all_pred_list[i+1][0],"skill"])
                i+=2
            else :
                skill_list.append([all_pred_list[i][0],all_pred_list[i][1]])
                i+=1
    return skill_list


app = Flask(__name__)

with open("data/model2_epoch_6.json","rb+") as f:
		loaded_model_json  = f.read()

loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("data/model2_weights_epoch_6.h5")

@app.route('/dl_predict_skills', methods=['POST'])

def predict():
	print "Here"
	json_ = request.json
	#print json_['paragraph']
	prediction = get_predicted_skills(loaded_model,json_['paragraph'])
	#print prediction
	return json.dumps(prediction)






if __name__ == '__main__':

	#print get_predicted_skills(loaded_model,"recommender systems")
	app.run(port=4281,host='10.0.0.14')