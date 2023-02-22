# Movie-Chatbot
I created a Chatbot that reads WikiData of the movies and answers questions related to it.

Four types of Data Sets have been used in this work:
- Movie Knowledge Graph from the Wiki Data 
- MovieNet Data set describing the plot and images of the movies
- TransE and RESCAL Embeddings generated from LibKGE
- Crowd Source Data from MTurk

The Chatbot can perform five different types of tasks
- Give answers to factual questions
- Give embedding based answers
- Give you multimedia outputs
- Give recommendations
- Give crowdsource based answers whilst correcting the knowledge graph

The code uses the following concepts:
- Natural Language Processing
- Embeddings
- Zero Shot Classification
- NER (BERT)/POS Tags
- Cosine Similarity
