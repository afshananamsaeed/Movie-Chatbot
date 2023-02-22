# import the necessary libraries
import time
import atexit
import getpass
import requests  # install the package via "pip install requests"
from collections import defaultdict
import rdflib
from rdflib.namespace import Namespace, RDF, RDFS, XSD
from rdflib.term import URIRef, Literal
import csv
import json
import networkx as nx
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
import re
from transformers import pipeline
import torch
import json
import torch
from sklearn.metrics import pairwise_distances
from PIL import Image

from collections import defaultdict, Counter
# import locale
# _ = locale.setlocale(locale.LC_ALL, '')
# from flair.data import Sentence
# from flair.models import SequenceTagger
# tagger = SequenceTagger.load('ner')
# import nltk
# from nltk.stem import PorterStemmer
# nltk.download('punkt')
# from nltk.tokenize import sent_tokenize, word_tokenize

# functions to load the models and embeddings
def load_zero_shot():
  zero_shot_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0 if(torch.cuda.is_available()) else -1)
  return zero_shot_pipeline

def load_ner_pipeline():
    ner_pipeline = pipeline('ner', model='dslim/bert-large-NER', device=0 if(torch.cuda.is_available()) else -1)
    return ner_pipeline

def load_graph_embeddings():
  # load the embeddings
  entity_emb = np.load('/Users/afshananamsaeed/Desktop/Lecture Slides/Advanced Topics in AI/Tutorials/Project/data/ddis-graph-embeddings/entity_embeds.npy')
  relation_emb = np.load('/Users/afshananamsaeed/Desktop/Lecture Slides/Advanced Topics in AI/Tutorials/Project/data/ddis-graph-embeddings/relation_embeds.npy')

  # load the dictionaries
  with open('/Users/afshananamsaeed/Desktop/Lecture Slides/Advanced Topics in AI/Tutorials/Project/data/ddis-graph-embeddings/entity_ids.del', 'r') as ifile:
    ent2id = {rdflib.term.URIRef(ent): int(idx) for idx, ent in csv.reader(ifile, delimiter='\t')}
    id2ent = {v: k for k, v in ent2id.items()}
  with open('/Users/afshananamsaeed/Desktop/Lecture Slides/Advanced Topics in AI/Tutorials/Project/data/ddis-graph-embeddings/relation_ids.del', 'r') as ifile:
    rel2id = {rdflib.term.URIRef(rel): int(idx) for idx, rel in csv.reader(ifile, delimiter='\t')}
    id2rel = {v: k for k, v in rel2id.items()}

  ent2lbl = {ent: str(lbl) for ent, lbl in graph.subject_objects(RDFS.label)}
  lbl2ent = {lbl: ent for ent, lbl in ent2lbl.items()}

  return entity_emb, relation_emb, ent2id, rel2id, id2ent, ent2lbl, lbl2ent

# function to preprocess the crowd data
def preprocess_crowd_data(crowd_data):
    # remove malicious worker
    crowd_data = crowd_data[crowd_data['FixPosition']!="yes"]
    crowd_data = crowd_data[crowd_data['FixPosition']!="I don't understand"]
    for index, str in enumerate(crowd_data["FixPosition"]):
        try:
            if str.isnumeric():
                crowd_data = crowd_data.drop([crowd_data.index[index]])
        except: continue
    crowd_data = crowd_data[crowd_data["WorkTimeInSeconds"]>10]
    crowd_data['LifetimeApprovalRate'] = crowd_data['LifetimeApprovalRate'].map(lambda x: float(x.rstrip('%')))
    crowd_data = crowd_data[crowd_data["LifetimeApprovalRate"]>40]
    crowd_data.drop(columns=["Title","Reward","AssignmentId","WorkerId","AssignmentStatus","WorkTimeInSeconds","LifetimeApprovalRate","AnswerID"],inplace=True)
    crowd_data.reset_index(drop=True)

    # aggregate the data
    crowd_data_dummied = pd.get_dummies(crowd_data,columns=["AnswerLabel"])
    crowd_data_grouped = crowd_data_dummied.groupby(["HITId","HITTypeId","Input1ID","Input2ID","Input3ID"]).sum()
    crowd_data_grouped = crowd_data_grouped.reset_index(drop=False)

    crowd_data_grouped["Input1ID"] = crowd_data_grouped["Input1ID"].replace("wd:",rdflib.term.URIRef('http://www.wikidata.org/entity/'),regex=True)
    crowd_data_grouped["Input2ID"] = crowd_data_grouped["Input2ID"].replace("wdt:",rdflib.term.URIRef('http://www.wikidata.org/prop/direct/'),regex=True)
    crowd_data_grouped["Input3ID"] = crowd_data_grouped["Input3ID"].replace("wd:",rdflib.term.URIRef('http://www.wikidata.org/entity/'),regex=True)

    # get fixed values
    crowd_data_grouped["FixPosition"] = ""
    crowd_data_grouped["FixValue"] = ""
    for index, (id, correct, incorrect) in enumerate(zip(crowd_data_grouped["HITId"],crowd_data_grouped["AnswerLabel_CORRECT"],crowd_data_grouped["AnswerLabel_INCORRECT"])):
        if incorrect>correct:
            df = crowd_data[crowd_data["HITId"]==id]
            pos_list = list(set(df["FixPosition"]))
            val_list = list(set(df["FixValue"]))
            if np.nan in pos_list:
                pos_list.remove(np.nan)
            if np.nan in val_list:
                val_list.remove(np.nan)
            try:
                crowd_data_grouped.loc[index,"FixPosition"] = pos_list[0]
            except: continue
            try:
                crowd_data_grouped.loc[index,"FixValue"] = val_list[0]
            except: continue

    return crowd_data_grouped

# load pipelines
ner = load_ner_pipeline()
zero_shot = load_zero_shot()
# load graph and embeddings
graph = rdflib.Graph()
graph.parse('/Users/afshananamsaeed/Desktop/Lecture Slides/Advanced Topics in AI/Tutorials/Project/data/14_graph.nt', format='turtle')
entity_emb, relation_emb, ent2id, rel2id, id2ent, ent2lbl, lbl2ent = load_graph_embeddings()
wd_dict = np.load('/Users/afshananamsaeed/Desktop/Lecture Slides/Advanced Topics in AI/Tutorials/Project/Saved_files/wd_dict.npy',allow_pickle='TRUE').item()
wdt_dict = np.load('/Users/afshananamsaeed/Desktop/Lecture Slides/Advanced Topics in AI/Tutorials/Project/Saved_files/wdt_dict.npy',allow_pickle='TRUE').item()
schema_dict = np.load('/Users/afshananamsaeed/Desktop/Lecture Slides/Advanced Topics in AI/Tutorials/Project/Saved_files/schema_dict.npy',allow_pickle='TRUE').item()
# load image data
images = json.load(open('/Users/afshananamsaeed/Desktop/Lecture Slides/Advanced Topics in AI/Tutorials/Project/data/images.json'))
# load and preprocess crowd data
crowd_data = pd.read_table("/Users/afshananamsaeed/Desktop/Lecture Slides/Advanced Topics in AI/Tutorials/Project/data/crowd_data.tsv")
crowd_data_processed = preprocess_crowd_data(crowd_data)

RDFS = rdflib.namespace.RDFS
WD = Namespace('http://www.wikidata.org/entity/')
WDT = Namespace('http://www.wikidata.org/prop/direct/')
SCHEMA = Namespace('http://schema.org/')
DDIS = Namespace('http://ddis.ch/atai/')

# url of the speakeasy server
url = 'https://server5.speakeasy-ai.org'
listen_freq = 3

# The ChatBot class
class ChatBot:
    def __init__(self, question):
        self.question = question
        self.image_question_keywords = ["image","picture","poster","Show","look","portrait","photo","photograph","scene","scenes"]
        self.recommendation_word_list = ["recommend","recommendation","suggest","suggestion"]

    def remove_question_marks(self):
        self.question = "".join(x for x in self.question if x != "?")

    # function to distribute the asked question into recommendation and question type
    def question_type(self):
        self.get_subjects_predicates()
        intents = ["Question","Recommendation"]
        intents = zero_shot(self.question, intents)
        top_intent = intents['labels'][0]
        if top_intent=="Question":
            answer = self.answer_questions()
        elif top_intent=="Recommendation":
            answer = self.recommendation_questions()
        return answer

    # function to extract information out of the question
    def get_subjects_predicates(self):

        # obtaining the genre keywords
        self.genre_keywords = []
        for _,_,o in graph.triples((None,rdflib.term.URIRef('http://www.wikidata.org/prop/direct/P136'),None)):
            try:
                self.genre_keywords.append(ent2lbl[o])
            except: continue
        self.genre_keywords = list(set(self.genre_keywords))

        # getting the entities
        self.remove_question_marks()
        self.entity_list = []
        self.entity_cls = []
        self.pred_list = []
        self.keyword_list = []

        # check for other forms of words in the sentence
        word_dict = {"director":["direct","directed","directing"],
                    "executive producer":["produce","produced","producing","producer"],
                    "film editor":["edit","edited","editing"],
                    "cast member":["acted","act","acting","actor"],
                    "place of birth":["birthplace","born"],
                    "publication date":["date of publication"],
                    "place of publication":["publication place"],
                    "nominated for":["nominated"],
                    "location":["located","locating"],
                    "country of origin":["origin country"],
                    "cast member":["actors", "cast member","lead actor","actor","actress"]}
        
        for key,value in word_dict.items():
            for subvalue in value:
                if subvalue in self.question:
                    self.question = self.question.replace(subvalue,key)

        # getting the named entities using bert ner
        entities = ner(self.question, aggregation_strategy="simple")           
        for entity in entities:
            try:
                subject = lbl2ent[entity['word']]
                self.entity_list.append(subject)
                self.entity_cls.append(entity['entity_group'])
            except: continue
        
        # getting the named entities using flair ner
        # sentence = Sentence(self.question)         
        # tagger.predict(sentence)
        # for entity in sentence.get_spans('ner'):
        #     try:
        #         subject = lbl2ent[entity.text]
        #         if subject not in self.entity_list:
        #             self.entity_list.append(subject)
        #             self.entity_cls.append(entity.get_label('ner').value)
        #     except: continue

        # getting the predicate values if any
        for key,value in wdt_dict.items():
            if str(value) in self.question:
                self.pred_list.append(key)

        # obtaining other keywords if any
            for word in self.genre_keywords:
                if word in self.question:
                    self.keyword_list.append(lbl2ent[word])

    # function for passing the question into factual, embedding and multimedia based functions
    def answer_questions(self):
        try:
            # passing the question to the recommendation function in case it is misdetected
            for keyword in self.recommendation_word_list:
                if keyword in self.question.lower():
                    answer = self.recommendation_questions()
                    return answer
            
            multimedia_question = False
            for keyword in self.image_question_keywords:
                if keyword in self.question:
                    multimedia_question = True
            
            if len(self.entity_list)==0 or len(self.pred_list)==0:
                answer = "Sorry I could not find what you are looking for. Please check your question."
            if multimedia_question:
                answer = self.get_images()
            elif len(self.pred_list)!=0 and len(self.entity_list)!=0:
                answer = self.get_factual_answer()
            return answer
        except:
            answer = "Looks like you may have asked me a faulty question"
            return answer
        
########################## factual/embedding question functions ###########################
    def get_factual_answer(self):
        subject = None
        predicate = None
        try:
            subject = self.entity_list[0]
        except: pass
        try:
            predicate = self.pred_list[0]
        except: pass

        if subject == None or predicate == None:
            answer = "Looks like you may have asked me a faulty question or I may not have data on what you asked!"
            return answer

        crowdsource = False
        for (s,p,o) in zip(crowd_data_processed["Input1ID"],crowd_data_processed["Input2ID"],crowd_data_processed["Input3ID"]):
            if all(str(x) in [s,p,o] for x in [subject,predicate]):
                crowdsource = True
        
        output = [o for s,p,o in graph.triples((subject,predicate,None))]
        if len(output)==0:
            answer = self.get_embedding_answer(subject, predicate)
        else:
            ans = []
            for out in output:
                try: ans.append(ent2lbl[out])
                except: ans.append(out)
            answer = "The answer to your question is "+",".join(ans)

        # function to pass question into crowdsourcing functions
        if crowdsource:
            answer_crowd = self.get_crowdsource_results(subject, predicate)
            if answer_crowd == answer:
                return answer
            else: return answer_crowd
        
        return answer

    def get_embedding_answer(self, subject, predicate):
        try:
            head = entity_emb[ent2id[rdflib.term.URIRef(subject)]]
            pred = relation_emb[rel2id[rdflib.term.URIRef(predicate)]]
            lhs = head + pred
            dist = pairwise_distances(lhs.reshape(1, -1), entity_emb).reshape(-1)
            most_likely = dist.argsort()

            ans = pd.DataFrame([
                    (id2ent[idx][len(WD):], wd_dict[id2ent[idx]], dist[idx], rank+1)
                    for rank, idx in enumerate(most_likely[:10])],
                    columns=('Entity', 'Label', 'Score', 'Rank'))
            answer = "The answer to your question is "+",".join(list(ans['Label']))
        except: answer = "Sorry the answer is unavailable!"
        return answer

######################### crowd-sourcing questions ###########################
    def get_crowdsource_results(self, subject, predicate):
        # get the required datapoint
        data = crowd_data_processed[(crowd_data_processed["Input1ID"]==str(subject)) & (crowd_data_processed["Input2ID"]==str(predicate))]
        correct = int(data['AnswerLabel_CORRECT'])
        incorrect = int(data['AnswerLabel_INCORRECT'])
        part = correct+incorrect
        pi_score = ((correct*(correct-1))+(incorrect*(incorrect-1)))/(part*(part-1))
        
        if correct>=incorrect:
            try:
                answer = "The answer to your question is "+",".join(ent2lbl[data["Input3ID"].iat[0]])
            except: 
                answer = "The answer to your question is "+",".join(data["Input3ID"].iat[0])
            return answer
        elif incorrect>correct:
            # correct the output and print
            try:
                if data["FixValue"].iat[0]!=np.nan:
                    answer = "{} - according to the crowd, who had an inter-rater agreement of {} in this batch. The answer distribution for this specific task was {} support votes and {} reject vote.".format(ent2lbl[data["FixValue"].iat[0]],round(pi_score,2),correct,incorrect)
                else:
                    answer = "{} - according to the crowd, who had an inter-rater agreement of {} in this batch. The answer distribution for this specific task was {} support votes and {} reject vote.".format(ent2lbl[data["Input3ID"].iat[0]],round(pi_score,2),correct,incorrect)
            except:
                if data["FixValue"].iat[0]!=np.nan:
                    answer = "{} - according to the crowd, who had an inter-rater agreement of {} in this batch. The answer distribution for this specific task was {} support votes and {} reject vote.".format(data["FixValue"].iat[0],round(pi_score,2),correct,incorrect)
                else:
                    answer = "{} - according to the crowd, who had an inter-rater agreement of {} in this batch. The answer distribution for this specific task was {} support votes and {} reject vote.".format(data["Input3ID"].iat[0],round(pi_score,2),correct,incorrect)
        return answer

########################## recommendation functions ###########################
    def recommendation_questions(self): 
        try:
            # check for misdetected questions
            for keyword in self.image_question_keywords:
                if keyword in self.question:
                    answer = self.get_images()
                    return answer

            if len(self.keyword_list)!=0:
                if len(self.entity_list)==0:
                    answer = self.get_keyword_recommendations()
                elif len(self.entity_list)!=0 and (len(self.entity_cls)!=0 and self.entity_cls[0]=="PER"):
                    answer = self.give_person_keyword_recommendations()
                elif len(self.entity_list)!=0 and (len(self.entity_cls)!=0 and self.entity_cls[0]!="PER"):
                    answer = self.give_simple_recommendations()
            elif len(self.pred_list)!=0 and len(self.entity_list)!=0:
                answer = self.recommend_predicate_based_questions()
            else:
                answer = self.give_simple_recommendations()
            return answer
        except: 
            answer = "Sorry I cannot come up with a suitable recommendation"
            return answer

    def get_keyword_recommendations(self):
        try:
            object = self.keyword_list[0]
            if len(self.pred_list)==0:
                predicate = Counter([p for s,p,o in graph.triples((None,None,object))]).most_common(1)[0][0]
            else:
                predicate = self.pred_list[0]

            # obtain the triple
            output = [s for s,p,o in graph.triples((None,predicate,object))]
            if len(output)==0:
                try:
                    tail = entity_emb[ent2id[rdflib.term.URIRef(object)]]
                    pred = relation_emb[rel2id[rdflib.term.URIRef(predicate)]]
                    lhs = tail - pred
                    dist = pairwise_distances(lhs.reshape(1, -1), entity_emb).reshape(-1)
                    most_likely = dist.argsort()

                    ans = pd.DataFrame([
                            (id2ent[idx][len(WD):], wd_dict[id2ent[idx]], dist[idx], rank+1)
                            for rank, idx in enumerate(most_likely[:10])],
                            columns=('Entity', 'Label', 'Score', 'Rank'))
                    ans = ans['Label']
                    answer = "The answer to your question is "+",".join(list(ans[:5]))
                    return answer
                except: 
                    answer = print("Sorry the answer is unavailable!")
                    return answer
            else:
                try:
                    ans = []
                    for out in output:
                        try: ans.append(ent2lbl[out])
                        except: continue
                    answer = "The answer to your question is "+",".join(list(ans[:5]))
                    return answer
                except:
                    answer = "Sorry I do not have a suitable recommendation"
                    return answer
        except:
            answer = "Sorry I could not find a suitable recommendation for you"
            return answer

    def give_person_keyword_recommendations(self):
        try:
            for entity in self.entity_list:
                predicate = Counter([p for s,p,o in graph.triples((None,None,entity))]).most_common(1)[0][0]                
                object = entity
                obtained_subjects = [s for s,p,o in graph.triples((None,predicate,object))]

                if len(self.keyword_list)==0:
                    ans = [ent2lbl[s] for s in obtained_subjects]
                else:
                    ans = []
                    for subject in obtained_subjects:
                        if graph.triples((subject,lbl2ent["genre"],self.keyword_list[0])):
                            ans.append(ent2lbl[subject])
                answer = "My recommendations are "+",".join(ans[:5])
                return answer
        except:
            answer = "Sorry I could not obtain a suitable recommendation"
            return answer    

    def give_simple_recommendations(self):
        try:
            movie_keywords = ["movie","movies","film","films"]
            if self.entity_cls[0]=="PER":
                for keyword in movie_keywords:
                    if keyword in self.question:
                        # find how the person is related to the movie
                        predicate = Counter([p for s,p,o in graph.triples((None,None,self.entity_list[0]))]).most_common(1)[0][0]
                        tail = self.entity_list[0]
                        list_s = [s for s,p,o in graph.triples((None,predicate,tail))]
                        ans_list = []
                        for out in list_s:
                            ans_list.append(ent2lbl[out])
                        answer = "My recommendations are "+",".join(ans_list[:5])
                        return answer

            else:
                head = entity_emb[ent2id[self.entity_list[0]]]
                dist = pairwise_distances(head.reshape(1, -1), entity_emb).reshape(-1)
                most_likely = dist.argsort()
                ranks = dist.argsort().argsort()

                ans = pd.DataFrame([
                        (id2ent[idx][len(WD):], wd_dict[id2ent[idx]], dist[idx], rank+1)
                        for rank, idx in enumerate(most_likely[:10])],
                        columns=('Entity', 'Label', 'Score', 'Rank'))
                ans_list = [value for value in list(ans['Label']) if value not in [ent2lbl[entity] for entity in self.entity_list]]
                answer = "My recommendations are "+",".join(ans_list[:5])
                return answer
        except: 
            answer = "Sorry I cannot come up with a suitable recommendation"
            return answer

    def recommend_predicate_based_questions(self):
        try:
            head = entity_emb[ent2id[self.entity_list[0]]]
            pred = relation_emb[rel2id[self.pred_list[0]]]
            lhs = head - pred
            dist = pairwise_distances(lhs.reshape(1, -1), entity_emb).reshape(-1)
            most_likely = dist.argsort()
            ranks = dist.argsort().argsort()

            ans = pd.DataFrame([
                    (id2ent[idx][len(WD):], wd_dict[id2ent[idx]], dist[idx], rank+1)
                    for rank, idx in enumerate(most_likely[:10])],
                    columns=('Entity', 'Label', 'Score', 'Rank'))
            answer = "My recommendations are "+",".join(list(ans['Label'])[1:])
            return answer
        except:
            answer = "Sorry I cannot come up with a suitable recommendation"
            return answer

######################### image functions ###########################
    def get_images(self):
        try:
            # sets the type of image
            image_types = ['behind_the_scenes','event','poster','product','production_art','publicity','still_frame','user_avatar']
            self.image_type = "still_frame"
            for type in image_types:
                if type in self.question:
                    self.image_type = type
            
            if len(set(self.entity_cls))==2 and set(self.entity_cls)=={"PER","MISC"}:
                answer = self.get_double_relation_image()
            elif ("PER" in set(self.entity_cls)) or ("MISC" in set(self.entity_cls)):
                answer = self.get_single_relation_image()
            return answer
        except:
            answer = "Sorry I cannot come up with a suitable image"
            return answer

    # return image for unrelated entities
    def get_single_relation_image(self):
        try:
            if len(self.entity_list)==0:
                answer = "Sorry an image for the entity was not found!"
                return answer
            
            else:
                for index,entity in enumerate(self.entity_list):
                    # first extract the imdb id
                    Imdb_pred_lbl = [key for key,value in wdt_dict.items() if value=="IMDb ID"][0]
                    subject = entity
                    predicate = Imdb_pred_lbl
                    imdb_id = None
                    for _,_,o in graph.triples((subject,predicate,None)):
                        imdb_id = o.toPython()
                    if imdb_id==None:
                        print("Sorry the image for the entity is not available!")
                    
                    # return the image from the image dataset
                    for image_details in images:
                        if self.entity_cls[index]=="PER":
                            if imdb_id in image_details['cast'] and image_details['type']==self.image_type:
                                answer = image_details['img'][: image_details['img'].find('.')]
                                # url = "https://files.ifi.uzh.ch/ddis/teaching/2021/ATAI/dataset/movienet/images/"+str(image_details['img'][: image_details['img'].find('.')])+".jpg"
                                break
                            elif imdb_id in image_details['cast']:
                                answer = image_details['img'][: image_details['img'].find('.')]
                                # url = "https://files.ifi.uzh.ch/ddis/teaching/2021/ATAI/dataset/movienet/images/"+str(image_details['img'][: image_details['img'].find('.')])+".jpg"
                                break                            
                        elif self.entity_cls[index]=="MISC":
                            if imdb_id in image_details['movie'] and image_details['type']==self.image_type:
                                answer = image_details['img'][: image_details['img'].find('.')]
                                # url = "https://files.ifi.uzh.ch/ddis/teaching/2021/ATAI/dataset/movienet/images/"+str(image_details['img'][: image_details['img'].find('.')])+".jpg"
                                break
                            elif imdb_id in image_details['movie']:
                                answer = image_details['img'][: image_details['img'].find('.')]
                                # url = "https://files.ifi.uzh.ch/ddis/teaching/2021/ATAI/dataset/movienet/images/"+str(image_details['img'][: image_details['img'].find('.')])+".jpg"
                                break
                return "Here is your image image:"+answer
        except:
            answer = "Sorry I cannot come up with a suitable image"
            return answer

    # returns image for related entities
    def get_double_relation_image(self):
        try:
            if len(self.entity_list) == 0:
                answer = "Sorry an image for the entity was not found!"
                return answer
            else:
                per_imdb_id = None
                misc_imdb_id = None
                for index,entity in enumerate(self.entity_list):
                    # first extract the imdb id
                    Imdb_pred_lbl = [key for key,value in wdt_dict.items() if value=="IMDb ID"][0]
                    subject = entity
                    predicate = Imdb_pred_lbl
                    for _,_,o in graph.triples((subject,predicate,None)):
                        if self.entity_cls[index] == "PER":
                            per_imdb_id = o.toPython()
                        elif self.entity_cls[index] == "MISC":
                            misc_imdb_id = o.toPython()

                if per_imdb_id==None or misc_imdb_id==None:
                    answer = "Sorry the image is not available"
                    return answer
                # return the image from the image dataset
                for image_details in images:
                    if per_imdb_id in image_details['cast'] and misc_imdb_id in image_details['movie'] and image_details['type']==self.image_type:
                        answer = image_details['img'][: image_details['img'].find('.')]
                        # url = "https://files.ifi.uzh.ch/ddis/teaching/2021/ATAI/dataset/movienet/images/"+str(image_details['img'][: image_details['img'].find('.')])+".jpg"
                        return "Here is your image image:"+answer
                answer = "Sorry I could not find an image"
                return answer
        except:
            answer = "Sorry I cannot come up with a suitable image"
            return answer

    #####################################################

class DemoBot:
    def __init__(self, username, password):
        self.agent_details = self.login(username, password)
        self.session_token = self.agent_details['sessionToken']
        self.chat_state = defaultdict(lambda: {'messages': defaultdict(dict), 'initiated': False, 'my_alias': None})

        atexit.register(self.logout)

    def listen(self):
        while True:
            # check for all chatrooms
            current_rooms = self.check_rooms(session_token=self.session_token)['rooms']
            for room in current_rooms:
                # ignore finished conversations
                if room['remainingTime'] > 0:
                    room_id = room['uid']
                    if not self.chat_state[room_id]['initiated']:
                        # send a welcome message and get the alias of the agent in the chatroom
                        self.post_message(room_id=room_id, session_token=self.session_token, message='Hi there, Good morning. I am here to answer your questions on the movie industry. You can ask me a question and check if it is echoed in {} seconds.'.format(listen_freq))
                        self.chat_state[room_id]['initiated'] = True
                        self.chat_state[room_id]['my_alias'] = room['alias']

                    # check for all messages
                    all_messages = self.check_room_state(room_id=room_id, since=0, session_token=self.session_token)['messages']

                    # you can also use ["reactions"] to get the reactions of the messages: STAR, THUMBS_UP, THUMBS_DOWN

                    for message in all_messages:
                        if message['authorAlias'] != self.chat_state[room_id]['my_alias']:

                            # check if the message is new
                            if message['ordinal'] not in self.chat_state[room_id]['messages']:
                                self.chat_state[room_id]['messages'][message['ordinal']] = message
                                print('\t- Chatroom {} - new message #{}: \'{}\' - {}'.format(room_id, message['ordinal'], message['message'], self.get_time()))

                                ##### You should call your agent here and get the response message #####
                                bot = ChatBot(message['message'])
                                answer = bot.question_type()

                                # self.post_message(room_id=room_id, session_token=self.session_token, message='Got your message: \'{}\' at {}.'.format(message['message'], self.get_time()))
                                self.post_message(room_id=room_id, session_token=self.session_token, message='{}'.format(answer))
            time.sleep(listen_freq)

    def login(self, username: str, password: str):
        agent_details = requests.post(url=url + "/api/login", json={"username": username, "password": password}).json()
        print('- User {} successfully logged in with session \'{}\'!'.format(agent_details['userDetails']['username'], agent_details['sessionToken']))
        return agent_details

    def check_rooms(self, session_token: str):
        return requests.get(url=url + "/api/rooms", params={"session": session_token}).json()

    def check_room_state(self, room_id: str, since: int, session_token: str):
        return requests.get(url=url + "/api/room/{}/{}".format(room_id, since), params={"roomId": room_id, "since": since, "session": session_token}).json()

    def post_message(self, room_id: str, session_token: str, message: str):
        tmp_des = requests.post(url=url + "/api/room/{}".format(room_id),
                                params={"roomId": room_id, "session": session_token}, data=message).json()
        if tmp_des['description'] != 'Message received':
            print('\t\t Error: failed to post message: {}'.format(message))

    def get_time(self):
        return time.strftime("%H:%M:%S, %d-%m-%Y", time.localtime())

    def logout(self):
        if requests.get(url=url + "/api/logout", params={"session": self.session_token}).json()['description'] == 'Logged out':
            print('- Session \'{}\' successfully logged out!'.format(self.session_token))


if __name__ == '__main__':
    username = 'needyHyena6_bot'
    password = getpass.getpass()
    demobot = DemoBot(username, password)
    demobot.listen()
