python content_based.py

split -l 2500001 -d results_content_based.csv results_content_based_
mv results_content_based_00 results/results_content_based_1.csv
mv results_content_based_01 results/results_content_based_2.csv
rm results_content_based.csv
