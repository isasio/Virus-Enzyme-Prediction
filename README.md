# Virus-Enzyme-Prediction
Repository for the code and data used in the article Efficient searches in protein sequence space through AI-driven iterative learning


List of files:

Data Directory

  -lasso_prediction_CB6_aic_opt_order_biochem.txt: Experimental data for the binding study on CB6 with the virus sequences.

  -lasso_prediction_CoV555_aic_opt_order_biochem.txt: Experimental data for the binding study on CoV555 with the virus sequences.

  -lasso_prediction_REGN10987_aic_opt_order_biochem.txt: Experimental data for the binding study on REGN10987 with the virus sequences.

  -fitness_data_wt.csv: Dihydrofolate reductase fitness experimental data. 

Code Directory:

  -xg_iter_random_cb6.py: Python code for the XGBoost classification of the binding study on CB6 with the virus sequences.

  -xg_iter_random_cov555.py: Python code for the XGBoost classification of the binding study on CoV555 with the virus sequences.

  -xg_iter_random_regn10987.py: Python code for the XGBoost classification of the binding study on REGN10987 with the virus sequences.

  -rf_iter_random_cb6.py: Python code for the Random Forest classification of the binding study on CB6 with the virus sequences.

  -rf_iter_random_cov555.py: Python code for the Random Forest classification of the binding study on CoV555 with the virus sequences.

  -rf_iter_random_regn10987.py: Python code for the Random Forest classification of the binding study on REGN10987 with the virus sequences.

  -mlp_iter_random_cb6.py: Python code for the MLP classification of the binding study on CB6 with the virus sequences.

  -mlp_iter_random_cov555.py: Python code for the MLP classification of the binding study on CoV555 with the virus sequences.

  -mlp_iter_random_regn10987.py: Python code for the MLP classification of the binding study on REGN10987 with the virus sequences.

  -codon_multi_xg_high.py: Python code for the XGBoost regression search on the fitness combinatorial space of the Dihydrofolate reductase experimental data.

  -codon_multi_rf_high.py: Python code for the Random Forest regression search on the fitness combinatorial space of the Dihydrofolate reductase experimental data.

  -codon_multi_mlp_high.py: Python code for the MLP regression search on the fitness combinatorial space of the Dihydrofolate reductase experimental data.

