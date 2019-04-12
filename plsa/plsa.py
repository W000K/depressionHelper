import numpy as np
import math
from pandas import *
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

	def __init__(self, documents_path):
		"""
		Initialize empty document list.
		"""
		self.documents = []
		self.vocabulary = []
		self.likelihoods = []
		self.documents_path = documents_path
		self.term_doc_matrix = None 
		self.document_topic_prob = None  # P(z | d), D
		self.topic_word_prob = None  # P(w | z), T
		self.topic_prob = None  # P(z | d, w), Z, hidden variables

		self.number_of_documents = 0
		self.vocabulary_size = 0

	def build_corpus(self):
		"""
		Read document, fill in self.documents, a list of list of word
		self.documents = [["the", "day", "is", "nice", "the", ...], [], []...]
		
		Update self.number_of_documents
		"""
		
		#added stop words removal
		stop_words = set(stopwords.words('english')) 

		with open(self.documents_path) as fp:
			line = fp.readline()
			cnt = 0
			while line:
				filtered = [word for word in line[1:].split() if not word.lower() in stop_words]
				print('******* filtered ********')

				print(filtered)
				self.documents.append(filtered)
				line = fp.readline()
				cnt += 1
		self.number_of_documents = cnt
		print(self.number_of_documents)

	def build_vocabulary(self):
		"""
		Construct a list of unique words in the whole corpus. Put it in self.vocabulary
		for example: ["rain", "the", ...]

		Update self.vocabulary_size
		"""
		
		self.vocabulary = np.unique(np.array(self.documents))
		self.vocabulary_size = len(self.vocabulary)

	def build_term_doc_matrix(self):
		"""
		Construct the term-document matrix where each row represents a document, 
		and each column represents a vocabulary term.

		self.term_doc_matrix[i][j] is the count of term j in document i
		"""
		mat = []
		for document in self.documents:
			cnts = []
			for vocab in self.vocabulary:
				cnt = 0
				cnts.append(document.count(vocab))
			mat.append(cnts)
		self.term_doc_matrix = np.array(mat)


	def initialize_randomly(self, number_of_topics):
		"""
		Randomly initialize the matrices: document_topic_prob and topic_word_prob
		which hold the probability distributions for P(z | d) and P(w | z): self.document_topic_prob, and self.topic_word_prob

		Don't forget to normalize!
		"""
		self.document_topic_prob = np.random.rand(self.number_of_documents, number_of_topics)
		self.document_topic_prob = normalize(self.document_topic_prob)

		self.topic_word_prob = np.random.rand(number_of_topics, len(self.vocabulary))
		self.topic_word_prob = normalize(self.topic_word_prob)
		

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
		print("Initializing...")

		if random:
			self.initialize_randomly(number_of_topics)
		else:
			self.initialize_uniformly(number_of_topics)

	def expectation_step(self):
		""" The E-step updates P(z | w, d)
		"""
		# print("E step:")
		
		for d_idx in range(self.number_of_documents):
			summ = []
			for w_idx in range(self.vocabulary_size):
				summ.append(self.document_topic_prob[d_idx, :] * self.topic_word_prob[:, w_idx])

			summ_n = normalize(np.array(summ))
			self.topic_prob[d_idx] = summ_n
				

	def maximization_step(self, number_of_topics):
		""" The M-step updates P(w | z)
		"""
		# print("M step:")
		
		# update P(w | z)
		for t_idx in range(number_of_topics):
			acc = []
			for w_idx in range(self.vocabulary_size):
				summ = 0
				for d_idx in range(self.number_of_documents):
					summ += self.term_doc_matrix[d_idx][w_idx] * self.topic_prob[d_idx, w_idx, t_idx]
				acc.append(summ)
			self.topic_word_prob[t_idx] = acc 
		self.topic_word_prob = normalize(self.topic_word_prob)
		
		# update P(z | d)
		for d_idx in range(self.number_of_documents):
			acc = []
			for t_idx in range(number_of_topics):
				summ = 0
				for w_idx in range(self.vocabulary_size):
					summ += self.term_doc_matrix[d_idx][w_idx] * self.topic_prob[d_idx][w_idx][t_idx]
				acc.append(summ)
			self.document_topic_prob[d_idx] = acc
		self.document_topic_prob = normalize(self.document_topic_prob)

	def calculate_likelihood(self, number_of_topics):
		""" Calculate the current log-likelihood of the model using
		the model's updated probability matrices
		
		Append the calculated log-likelihood to self.likelihoods

		"""
		summ = 0
		for d_idx in range(self.number_of_documents):
			summm = 0
			for w_idx in range(self.vocabulary_size):
				summm += self.term_doc_matrix[d_idx][w_idx] * math.log(np.dot(self.document_topic_prob[d_idx], self.topic_word_prob[:, w_idx]))
			summ += summm
		
		return summ

	def plsa(self, number_of_topics, max_iter, epsilon):

		"""
		Model topics.
		"""
		print ("EM iteration begins...")
		
		# build term-doc matrix
		self.build_term_doc_matrix()
		
		# Create the counter arrays.
		
		# P(z | d, w)
		self.topic_prob = np.zeros([self.number_of_documents, self.vocabulary_size, number_of_topics], dtype=np.float)
		# P(z | d) P(w | z)
		self.initialize(number_of_topics, random=True)

		# Run the EM algorithm
		current_likelihood = 0.0

		for iteration in range(max_iter):
			print("Iteration #" + str(iteration + 1) + "...")

			self.expectation_step()

			self.maximization_step(number_of_topics)

			likelihood = self.calculate_likelihood(number_of_topics)
			print(likelihood)

			# if (abs(likelihood - current_likelihood) <= 0.001):
				# return

			current_likelihood = likelihood


def main():
	documents_path = 'data/testinput.txt'
	corpus = Corpus(documents_path)  # instantiate corpus
	corpus.build_corpus()
	corpus.build_vocabulary()
	print('******* VOCABULARY ********')
	print(corpus.vocabulary)
	print("Vocabulary size:" + str(len(corpus.vocabulary)))
	print("Number of documents:" + str(len(corpus.documents)))
	number_of_topics = 5
	max_iterations = 50
	epsilon = 0.001
	corpus.plsa(number_of_topics, max_iterations, epsilon)
	print('***** TOPICS PROB FOR THIS DOC*****')
	print(DataFrame(corpus.document_topic_prob))

	print('***** TOPIC WORD DISTR *****')
	idx = np.argmax(corpus.document_topic_prob)
	chosen_topic = corpus.topic_word_prob[idx]
	chosen_word_idx = chosen_topic.argsort()[-3:][::-1]
	for idx in chosen_word_idx:
		chosen_word = corpus.vocabulary[idx]
		print(chosen_word)

if __name__ == '__main__':
	main()
