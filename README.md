# Master thesis project of Tobias Steinbach

## **Machine Learning Approach to Detect Adverse Outcomes After CABG Surgery Derived From Cardiovascular Time Series Extraction**

### TU Darmstadt
### Electrical Engineering and Information Technology Department
### KIS*MED - Künstlich Intelligente Systeme der Medizin

In this thesis machine learning algorithms are used to predict adverse outcomes of coronary bypass
graft (CABG) surgery. This is motivated by a correlation of morbidity and the postoperative complications
considered in this work, specifically acute kidney dysfunction (AKD) stage three, acute kidney injury (AKI1)
stage one, low cardiac output syndrome (LCOS) and atrial fibrillation (AF). By being able to predict these
complications life-saving measures could be taken earlier. Usually these outcomes are predicted based
on demographical and clinical data of the patients. In recent research it was found that markers of the
cardiovascular control system do also have predictive power regarding those complications. This includes
especially the baroreflex sensitivity (BRS), a measure about the current degree of the cardiovascular control
functionality. Therefore these markers, which can be monitored during surgery, could be an alternative and
possibly more reliable way of predicting these outcomes. The focus of the present work is to leverage the
methods of machine learning algorithms to get another perspective on the influence of those markers. This
should lead to results that make a deeper estimation of the potential of these markers possible.

In a clinical study of the IRCCS Policlinico San Donato (GR-2013-02356272) data from about 290 patients
undergoing CAGB was collected. This includes beat-to-beat time series derived from an electrocardiogram
(ECG) signal as well as the invasive arterial pressure (AP) before (PRE) and after (POST) propofol general
anesthesia induction. Additionally, there are features regarding the background of the individual patients as
well as information about the concrete surgical process and its outcome. This combination of cardiovascular
markers as well as demographical and clinical features should be used as the basis for the predictions.

The influence of the features on a possible prediction and their correlation
with the adverse outcomes is analyzed. Afterwards machine learning algorithms are implemented
with the goal of predicting the outcome of a surgery based on the given data set. 
Several studies covering different aspects can be performed to gain insight on the different
features and data subsets on the prediction performance. For once the prediction based on only PRE
and POST are compared, to see if the markers collected from either part of the surgery are better as
predictors. To analyze their overall importance the cardiovascular markers are compared to the clinical
data regarding their prediction performance. As different BRS estimates are collected in the set, their
difference in correlation and prediction is also a valuable insight to investigate.

The thesis project was developed in collaboration with Laboratory of Complex Systems
Modeling, University of Milan, located at IRCCS Policlinico San Donato, San Donato Milanese. Part of the project was developed at DEIB –
Politecnico di Milano.

### Respository

This is the code base that supports the thesis to conduct the desired studies. There are three
subdirectories: *code_base* contains the python code needed to run the program, *results* the output files 
produced by the program and *data* the data set files that are the basis of this thesis. In *code_base* the main file (*main.py*)
from where all remaining functions get called and a file to set parameters of the program (*global_variables.py*) can be found. 
The rest of the folder is split into *exploration*, *classification*, *evaluation* and *util*. In *exploration* the files
to analyze the data set are contained. This involves exploration plots (*explorational_data_analysis.py*) and 
correlation calculations (*correlation_analysis.py*). Those discover characteristics of the data set before 
starting the prediction analysis. This then is contained in the *prediction* directory that is filled with 
*preproces_data.py* to implement preprocessing steps necessary to prepare the data set and *classification.py* 
that implements the actual prediction. To perform the prediction from outside the functions *classify()* or 
*classify_k_fold()* can be called either with a split train and test set or with a complete set using cross validation. The prediction
implementation is then used to perform several studies implemented in the *evaluation* folder. In *data_set_evaluation.py* 
the different subsets of the complete set are compared regarding their prediction potential, while *feature_avaluation*
focuses on the predictive performance of single features or small combinations. Lastly *parameter_evaluation* can be used to perform 
a Bayesian optimization on the preprocessing and prediction model parameters. To support the program *util* contains either
plotting needed for the report (*additional_functions.py*) or the parsing of the Excel files that contain the data set
(*read_excel.py*). 

It is notable that to run this project Python 3.8.5 should be used. It is necessary to install mulitple packaged that are used in the code.
If a problem with the SMOTE oversampling occurs try to reinstall the package threadpoolctl. The program can be modified using the 
parameters in *global_variables.py*. Here the used classifiers, the outcomes to consider and the data subsets that should be compared
are specified. Also this is the way to alter the used preprocessing steps. It is possible to enable standardization, enable cross validation
and choose the number of folds, control the amount of outliers filtered by z-score filtering and enable the SMOTE oversampling. For the 
models SVM and NaiveBayes the parameters for preprocessing and prediction are stored in this file as well. They were optimized using 
the Bayesian optimization. To enable the SHAP analysis in a prediction you can set *explain_predictions* to True. If you want to penalize
a non-learning of a model more in an evaluation study set *scale_bad_performance_results* to True. For detailed information on how to set all of 
these parameters refer to the comments in this file. To only perform single parts of the whole program comment out the undesired functionalities
in the main function in *main.py*. The subdirectory containing the results is not added to git and is created on demand.


