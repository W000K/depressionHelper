import numpy as np
import math
import sys,os
from collections import Counter 
import nltk
from nltk.corpus import stopwords

def normalize(input_matrix):
    """
    Normalizes the rows of a 2d input_matrix so they sum to 1
    """

    row_sums = input_matrix.sum(axis=1)
    assert (np.count_nonzero(row_sums)==np.shape(row_sums)[0]) # no row should sum to zero
    new_matrix = input_matrix / row_sums[:, np.newaxis]
    return new_matrix

       
class Corpus(object):

    """
    A collection of documents.
    """

    def __init__(self, document):
        """
        Initialize empty document list.
        """
        self.documents = []
        self.vocabulary = []
        self.likelihoods = []
        self.document = document
        #self.documents_path = documents_path
        self.term_doc_matrix = None 
        self.document_topic_prob = None  # P(z | d)
        self.topic_word_prob = None  # P(w | z)
        self.topic_prob = None  # P(z | d, w)
        self.k = 0
        self.number_of_documents = 0
        self.vocabulary_size = 0

    def build_corpus(self,line):
        """
        Read document, fill in self.documents, a list of list of word
        self.documents = [["the", "day", "is", "nice", "the", ...], [], []...]
        
        Update self.number_of_documents
        """
        # #############################
        # your code here
        # #############################
        stop_words = set(stopwords.words('english')) 

        # with open(self.documents_path, encoding="utf-8") as fp:
        #     line = fp.readline()
        #     cnt = 0
        #    while line:
        filtered = [word for word in line[1:].split() if not word.lower() in stop_words]
        #print('******* filtered ********')

        #print(filtered)
        self.documents.append(filtered)
        #        line = fp.readline()
        #cnt += 1
        self.number_of_documents = 1
        #print(self.number_of_documents)
            #print(line.split())
        #print(self.documents)
        #pass    # REMOVE THIS

    def build_vocabulary(self):
        """
        Construct a list of unique words in the whole corpus. Put it in self.vocabulary
        for example: ["rain", "the", ...]

        Update self.vocabulary_size
        """
        # #############################
        # your code here
        # #############################
        for i in range(0,self.number_of_documents):
            for j in range(0,len(self.documents[i])):
                if self.documents[i][j] not in self.vocabulary:
                    self.vocabulary.append(self.documents[i][j])
                    self.vocabulary_size+=1
        #print(self.vocabulary_size)    # REMOVE THIS

    def build_term_doc_matrix(self):
        """
        Construct the term-document matrix where each row represents a document, 
        and each column represents a vocabulary term.

        self.term_doc_matrix[i][j] is the count of term j in document i
        """
        # ############################
        # your code here
        # ############################
        self.term_doc_matrix = np.zeros((self.number_of_documents,self.vocabulary_size))
        for i in range(0,self.number_of_documents):
            for j in range(0,self.vocabulary_size):
                #print(i,j)
                # print(self.vocabulary[j])
                # print(self.documents[i])
                docCount = Counter(self.documents[i])
                #print(docCount)
                docDict = dict(docCount)
                #print(docDict)
                if self.vocabulary[j] in docDict:
                    self.term_doc_matrix[i][j] = docDict[self.vocabulary[j]]
                else:
                    self.term_doc_matrix[i][j] = 0    
        #pass    # REMOVE THIS


    def initialize_randomly(self, number_of_topics):
        """
        Randomly initialize the matrices: document_topic_prob and topic_word_prob
        which hold the probability distributions for P(z | d) and P(w | z): self.document_topic_prob, and self.topic_word_prob

        Don't forget to normalize!
        """
        # ############################
        # your code here
        # ############################
        self.k = number_of_topics
        self.document_topic_prob = np.random.rand(self.number_of_documents,number_of_topics)
        self.document_topic_prob = normalize(self.document_topic_prob)

        self.topic_word_prob = np.random.rand(number_of_topics,len(self.vocabulary))
        self.topic_word_prob = normalize(self.topic_word_prob)
        #pass    # REMOVE THIS
        

    def initialize_uniformly(self, number_of_topics):
        """
        Initializes the matrices: self.document_topic_prob and self.topic_word_prob with a uniform 
        probability distribution. This is used for testing purposes.

        DO NOT CHANGE THIS FUNCTION
        """
        self.document_topic_prob = np.ones((self.number_of_documents, number_of_topics))
        self.document_topic_prob = normalize(self.document_topic_prob)

        self.topic_word_prob = np.ones((number_of_topics, len(self.vocabulary)))
        self.topic_word_prob = normalize(self.topic_word_prob)

    def initialize(self, number_of_topics, random=False):
        """ Call the functions to initialize the matrices document_topic_prob and topic_word_prob
        """
        #print("Initializing...")

        if random:
            self.initialize_randomly(number_of_topics)
        else:
            self.initialize_uniformly(number_of_topics)

    def expectation_step(self):
        """ The E-step updates P(z | w, d)
        """
        #print("E step:")
        
        # ############################
        # your code here
        # ############################
        for i in range(0,self.number_of_documents):
            for j in range(0,self.vocabulary_size):
                denom = 0
                for k in range(0,self.k):
                    # print(self.document_topic_prob[i,k])
                    # print("---------------")
                    # print(self.topic_word_prob[k,j])
                    # print("---------------")
                    # print(self.topic_prob.shape)
                    # print(".................")
                    self.topic_prob[i,k,j] = self.document_topic_prob[i,k]*self.topic_word_prob[k,j]
                    denom+=self.topic_prob[i,k,j]
                if denom==0:
                    for k in range(0,self.k):
                        self.topic_prob[i,k,j]=0
                else:
                    for k in range(0,self.k):
                        self.topic_prob[i,k,j]/=denom
        
        #pass    # REMOVE THIS
            

    def maximization_step(self, number_of_topics):
        """ The M-step updates P(w | z)
        """
        #print("M step:")
        
        # update P(w | z)
        
        # ############################
        # your code here
        # ############################
        for k in range(0,number_of_topics):
            denom = 0
            for j in range(0,self.vocabulary_size):
                self.topic_word_prob[k][j] = 0
                for i in range(0,self.number_of_documents):
                    self.topic_word_prob[k,j]+=self.term_doc_matrix[i,j]*self.topic_prob[i,k,j]
                denom+=self.topic_word_prob[k,j]
            if denom ==0:
                for j in range(0,self.vocabulary_size):
                    self.topic_word_prob[k,j]=1/self.vocabulary_size
            else:
                for j in range(0,self.vocabulary_size):
                    self.topic_word_prob[k,j]/=denom
        self.topic_word_prob = normalize(self.topic_word_prob)
        # update P(z | d)

        # ############################
        # your code here
        # ############################
        for i in range(0,self.number_of_documents):
            for k in range(0,number_of_topics):
                self.document_topic_prob[i][k]=0
                denom = 0
                for j in range(0,self.vocabulary_size):
                    self.document_topic_prob[i][k]+=self.term_doc_matrix[i][j]*self.topic_prob[i,k,j]
                    denom +=self.term_doc_matrix[i][j]
                if denom==0:
                    self.document_topic_prob[i][k]=1/number_of_topics
                else:
                    self.document_topic_prob[i][k]/=denom
        self.document_topic_prob = normalize(self.document_topic_prob)
        #pass    # REMOVE THIS


    def calculate_likelihood(self, number_of_topics):
        """ Calculate the current log-likelihood of the model using
        the model's updated probability matrices
        
        Append the calculated log-likelihood to self.likelihoods

        """
        # ############################
        # your code here
        # ############################
        log_likelihood = 0
        for i in range(0,self.number_of_documents):
            for j in range(0,self.vocabulary_size):
                tmp=0
                for k in range(0,number_of_topics):
                    tmp+=self.topic_word_prob[k][j]*self.document_topic_prob[i][k]
                if tmp>0:
                    log_likelihood+=self.term_doc_matrix[i][j]*math.log(tmp)
        return log_likelihood

    def plsa(self, number_of_topics, max_iter, epsilon):

        """
        Model topics.
        """
        #print ("EM iteration begins...")
        
        # build term-doc matrix
        self.build_term_doc_matrix()
        
        #print("term_doc_matrix built")
        # Create the counter arrays.
        
        # P(z | d, w)
        self.topic_prob = np.zeros([self.number_of_documents, number_of_topics, self.vocabulary_size], dtype=np.float)

        #print("topic_prob initialized")
        # P(z | d) P(w | z)
        self.initialize(number_of_topics, random=True)

        #print("Starting iterations")
        # Run the EM algorithm
        current_likelihood = 0.0

        for iteration in range(max_iter):
            #print("Iteration #" + str(iteration + 1) + "...")

            # ############################
            # your code here
            # ############################
            self.expectation_step()
            self.maximization_step(number_of_topics)
            current_likelihood = self.calculate_likelihood(number_of_topics)
            #print(current_likelihood)
            #pass    # REMOVE THIS
        #print(current_likelihood)


def main():
    documents_path = 'data/userTranscript.txt'
    helpdocs_path = 'data/helpDocs.txt'

    # build help doc database, takes a little while to do... ~3min
    helpDocs = []
    with open(helpdocs_path,encoding="utf-8") as fp2:
        for line in fp2:
            corpus = Corpus(line)  # instantiate corpus
            corpus.build_corpus(line)
            corpus.build_vocabulary()
            number_of_topics = 20
            max_iterations = 50
            epsilon = 0.001
            corpus.plsa(number_of_topics, max_iterations, epsilon)

            # print('     ***** HELP DOC TOPIC WORD DISTR *****')
            idx = np.argmax(corpus.document_topic_prob)
            chosen_topic = corpus.topic_word_prob[idx]
            chosen_word_idx = chosen_topic.argsort()[-20:][::-1]
            help_topics = []
            for idx in chosen_word_idx:
                chosen_word = corpus.vocabulary[idx]
                help_topics.append(chosen_word)
            helpDocs.append(help_topics)
    print('Help doc db built')

    # save a local data structure for all help docs for future usage
    allHelpDocs = open('data/helpDocs.txt', "r")
    docs = allHelpDocs.read().splitlines()

    # now for each user transcript...
    with open(documents_path,encoding="utf-8") as fp:

        # for each user transcript, recommend documents!
        for line in fp:
            corpus = Corpus(line)  # instantiate corpus
            corpus.build_corpus(line)
            corpus.build_vocabulary()
            number_of_topics = 20
            max_iterations = 50
            epsilon = 0.001
            corpus.plsa(number_of_topics, max_iterations, epsilon)

            print('***** TOPIC WORD DISTR *****')
            idx = np.argmax(corpus.document_topic_prob)
            chosen_topic = corpus.topic_word_prob[idx]
            chosen_word_idx = chosen_topic.argsort()[-3:][::-1]
            topics = []
            for idx in chosen_word_idx:
                chosen_word = corpus.vocabulary[idx]
                topics.append(chosen_word)
            print(topics)
            
            # compare the user transcript vector with every help doc vector, get top 10 documents!
            sims = []
            for helpDoc in helpDocs:
                a = set(helpDoc) 
                b = set(topics)
                c = a.intersection(b)

                #jacquard similarity: https://towardsdatascience.com/overview-of-text-similarity-metrics-3397c4601f50
                sim = float(len(c)) / (len(a) + len(b) - len(c))
                sims.append(sim)
            sims = np.array(sims)
            # get the 5 best doc

            fiveBestDocs = sims.argsort()[-5:][::-1]
            print('***** HERES WHAT I RECOMMEND... *****')
            for idx in fiveBestDocs:
                # print(idx)
                print(docs[idx])



if __name__ == '__main__':
    main()
