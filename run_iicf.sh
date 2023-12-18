python iicf.py

split -l 2500001 -d results_iicf.csv results_iicf_
mv results_iicf_00 results/results_iicf_1.csv
mv results_iicf_01 results/results_iicf_2.csv
rm results_iicf.csv
