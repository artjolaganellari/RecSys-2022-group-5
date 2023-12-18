pip install gensim
python lstm_word2vec.py

split -l 2500001 -d results_lstm_word2vec.csv results_lstm_word2vec_
mv results_lstm_word2vec_00 results/results_lstm_word2vec_1.csv
mv results_lstm_word2vec_01 results/results_lstm_word2vec_2.csv
rm results_lstm_word2vec.csv