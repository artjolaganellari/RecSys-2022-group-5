# RecSys Challenge 2022 - Group 5

For every implemented approach, there are 5 files:

* `<approach>.ipynb`: a Jupyter Notebook containing a readable form of the source code.
* `<approach>.py`: a Python script containing the very same code as the Jupyter notebook.
* `run_<approach>.sh`: a shell script which runs the above Python script.
* `results/results_<approach>_x.csv`: 2 csv files containing the results.

We have implemented the following different recommender systems.

(1) `iicf`: Item-item collaborative filtering. To run this script, just run the shell script `run_iicf.sh` from the terminal.

(2) `content_based`: A content-based model. To run this script, just run the shell script `run_content_based.sh` from the terminal.

(3) `lstm_word2vec`: An LSTM-based model using Word2Vec for Embedding. To run this script, just run the shell script `run_lstm_word2vec.sh` from the terminal.

(4) `lstm_with_contents`: An LSTM-based model using item contents for Embedding. To run this script, just run the shell script `run_lstm_with_contents.sh` from the terminal.

(5) `uucf`: User-user collaborative filtering (or more precisely, session-session collaborative filtering). To run this script, just run the shell script `run_uucf.sh` from the terminal.