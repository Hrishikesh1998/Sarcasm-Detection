The aim of this project is to detect sarcasm in tweets
The data files used are Mature.train and Joy.train

The project consist of 4 files:
Preprocess.py-used to preprocess the obtained tweets.
Word2vecTest.py-used to train and save word2vec vectors for the data file
Project.py-used to implement the MVME algorithm and hence helps in designing the kernel matrix
Main.py - the main file, used to execute the project.

To the run the project following steps should be followed.
i) We need to train the word2vector vectors for the data. For that execute word2vecTest.py
ii) Specify the data file to be used in the driver method at the end of the word2vec.py file.
iii) This would train the word2vec vectors and save them to a file with the target word
iv) After the vectors are trained we can start the classification process.
v) For classification execute main.py
viii) The main.py file will show the classification accuracy before stopping.
