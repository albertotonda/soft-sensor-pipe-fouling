# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 08:51:09 2023

Analyze the new dataset, that now includes adimensional features created by
Guillaume Delaplace.

@author: Alberto Tonda
"""
import datetime
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import ShuffleSplit

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.utils import all_estimators

from gplearn.genetic import SymbolicRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from inspect import signature, Parameter # this is to set the regressors' parameters

# this is to remove annoying warnings from PySR
# these two lines here are just to get rid of annoying warnings from PySR
import warnings
warnings.filterwarnings('ignore')

sns.set_theme() # set the default theme of seaborn

def sanitize_feature_names(feature_names) :
    """
    This is just to avoid weird characters in the feature names, that creates
    errors with PySRRegressor.
    """
    new_feature_names = []
    for fn in feature_names :
        new_fn = re.sub(r'[\W_]', '', fn)
        new_feature_names.append(new_fn)
    return new_feature_names

def create_predicted_vs_measured_plot(y_true, y_pred, figure_title, file_name) :
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # we add a y = x dotted line to be used as a reference
    max_value = max(max(y_true, y_pred))
    x = np.linspace(0, max_value * 1.1, num=100)
    ax.plot(x, x, alpha=0.3, color='red', linestyle='dashed', label="Theoretical best" )
    
    ax.scatter(y_true, y_pred, alpha=0.3, label="Actual points")
    
    ax.set_title(figure_title)
    ax.set_xlabel("Measured")
    ax.set_ylabel("Predicted")
    ax.legend(loc='best')
    
    plt.savefig(file_name, dpi=300)
    plt.close(fig)
    
    return

def create_feature_importance_barplots(regressors, folder_name) :
    
    # step 1: find all .csv with a specific name
    feature_importance_files = [f for f in os.listdir(folder_name) if f.find("fold") != -1]
    
    # step 2: iterate over regressors, find all .csv files that refer to that regressor
    for regressor_name in regressors :
        print("Analyzing feature importance files for regressor \"%s\"..." % regressor_name)
        regressor_feature_importance_files = [f for f in feature_importance_files if f.startswith(regressor_name)]
        
        feature_importance_values = {}
        for csv_file in regressor_feature_importance_files :
            df = pd.read_csv(os.path.join(folder_name, csv_file))
            feature_names = df["feature_name"].values
            feature_importance = df["importance"].values
            
            for i in range(0, len(feature_names)) :
                if feature_names[i] not in feature_importance_values :
                    feature_importance_values[feature_names[i]] = []
                feature_importance_values[feature_names[i]].append(feature_importance[i])
        
        # prepare everything for a barplot
        feature_names = [f for f in feature_importance_values]
        feature_means = [np.mean(v) for k, v in feature_importance_values.items()]
        feature_stdevs = [np.std(v) for k, v in feature_importance_values.items()]
        feature_xpos = np.arange(len(feature_names))        
        
        # and now create a barplot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        ax.bar(feature_xpos, feature_means, yerr=feature_stdevs, align='center', alpha=0.5, ecolor='black', capsize=10)
        ax.set_ylabel('Feature importance')
        ax.set_xticks(feature_xpos)
        ax.set_xticklabels(feature_names, rotation=45, rotation_mode='default', 
                           horizontalalignment='right', verticalalignment='top', fontsize=8)
        ax.set_title('Feature importance for regressor \"%s\"' % regressor_name)
        
        fig.tight_layout()
        ax.tick_params(axis='x', pad=0)
        
        plt.savefig(os.path.join(folder_name, regressor_name + "_feature_importance.png"), dpi=300)
        plt.close(fig)
    
    return

def initialize_regressor(regressor_class, random_seed=42, population_size=10000, generations=100, feature_names=None) :
    """
    This function initializes a classifier, setting some parameters to default
    values, if the classifier accepts them.
    """
    
    print("Now initializing regressor \"%s\"" % regressor_class.__name__)
    
    sig = signature(regressor_class.__init__)
    params = sig.parameters # these are not regular parameters, yet
    # we need to convert them to a dictionary
    params_dict = {}
    for p_name, param in params.items() :
        if params[p_name].default != Parameter.empty :
            params_dict[p_name] = params[p_name].default
    
    # a few parameters that apply to most regressors 
    if 'random_seed' in params :
        params_dict['random_seed'] = random_seed
        
    if 'random_state' in params :
        params_dict['random_state'] = random_seed
        
    if 'n_jobs' in params :
        params_dict['n_jobs'] = max(multiprocessing.cpu_count() - 1, 1) # use maximum available minus one; if it is zero, just use one

    # specific parameters for gplearn
    if 'population_size' in params :
        params_dict['population_size'] = population_size
    
    if 'generations' in params :
        params_dict['generations'] = generations
    
    if 'verbose' in params and regressor_class.__name__.startswith("SymbolicRegressor") :
        params_dict['verbose'] = 1
        
    if 'feature_names' in params :
        params_dict['feature_names'] = feature_names
        
    if 'niterations' in params and regressor_class.__name__.startswith("PySR") :
        params_dict['niterations'] = 100
        
    if 'population_size' in params :
        params_dict['population_size'] = 500
    
    if 'verbosity' in params and regressor_class.__name__.startswith("PySR") :
        params_dict['verbosity'] = 2
    
    if 'progress' in params and regressor_class.__name__.startswith("PySR") :
        params_dict['progress'] = False
        
    if 'variable_names' in params and feature_names is not None and regressor_class.__name__.startswith("PySR") :
        params_dict['variable_names'] = sanitize_feature_names(feature_names)
        
    # instantiate the regressor with the flattened dictionary of parameters
    regressor = regressor_class(**params_dict)
    
    return regressor


def create_experiments(df, value_bend=90) :
    """
    This function creates a list of "experiments", where each experiment is a
    dictionary with: (i) name of the experiment, (ii) X, (iii) y, (iv) feature
    names

    """
    experiments = []
    
    # these are the columns to be considered for dimensional experiments
    features_dimensional = ["Fitting_type", "D_in [mm]", "A_in [m^2]", "LD_Ratio [-]", 
                            "Temperature [°C]", "PMF [g_Part/g_total]", "Time [min]",
                            "Volume_flow [L/min]", "Appr. Re number [-]", "Fluid_Density [kg/m^3]",
                            "Dyn Viscosity [kg/(m*s)]",	"kin. Viscosity [m^2/s]"]
    target_dimensional = "Soil_mass [g]"
    
    # these are the columns to be considered for adimensional experiments
    features_adimensional = ["Fitting_type", "Re", "PMF", "time*", "Ar", "dens*"]
    target_adimensional = "w*"
    
    # experiment 1: all samples, dimensional
    X = df[features_dimensional].values
    y = df[target_dimensional].values
    
    experiment = {"name" : "all_samples_dimensional_features", "X" : X, "y" : y, 
                  "feature_names" : features_dimensional, "target_name" : target_dimensional}
    #experiments.append(experiment)
    
    # experiment 2 : Pipe_Socket, dimensional
    X = df[ df["Fitting_type"] != value_bend ][features_dimensional].values
    y = df[ df["Fitting_type"] != value_bend ][target_dimensional].values

    experiment = {"name" : "pipe_socket_dimensional_features", "X" : X, "y" : y, 
                  "feature_names" : features_dimensional, "target_name" : target_dimensional}
    #experiments.append(experiment)
    
    # experiment 3: Bend, dimensional
    X = df[ df["Fitting_type"] == value_bend ][features_dimensional].values
    y = df[ df["Fitting_type"] == value_bend ][target_dimensional].values

    experiment = {"name" : "bend_dimensional_features", "X" : X, "y" : y, 
                  "feature_names" : features_dimensional, "target_name" : target_dimensional}
    #experiments.append(experiment)

    # experiment 4: all samples, adimensional
    X = df[features_adimensional].values
    y = df[target_adimensional].values
    
    experiment = {"name" : "all_samples_adimensional_features", "X" : X, "y" : y, 
                  "feature_names" : features_adimensional, "target_name" : target_adimensional}
    experiments.append(experiment)
    
    # experiment 5 : Pipe_Socket, adimensional
    X = df[ df["Fitting_type"] != value_bend ][features_adimensional].values
    y = df[ df["Fitting_type"] != value_bend ][target_adimensional].values

    experiment = {"name" : "pipe_socket_adimensional_features", "X" : X, "y" : y, 
                  "feature_names" : features_adimensional, "target_name" : target_adimensional}
    #experiments.append(experiment)
    
    # experiment 6: Bend, adimensional
    X = df[ df["Fitting_type"] == value_bend ][features_adimensional].values
    y = df[ df["Fitting_type"] == value_bend ][target_adimensional].values

    experiment = {"name" : "bend_adimensional_features", "X" : X, "y" : y, 
                  "feature_names" : features_adimensional, "target_name" : target_adimensional}
    #experiments.append(experiment)
    
    return experiments

def write_experiment_information(experiment, file_name) :
    
    with open(file_name, "w") as fp :
        fp.write("# Experiment %s\n" % experiment["name"])
        fp.write("Number of samples: %d\n" % experiment["X"].shape[0])
        fp.write("Features: %s\n" % str(experiment["feature_names"]))
        fp.write("Target: %s\n" % experiment["target_name"])
        
    return

def save_regressor_feature_importance(regressor, feature_names, file_name) :
    
    regressor_name = regressor.__class__.__name__
    
    feature_importance = []
    if hasattr(regressor, "feature_importances_") :
        feature_importance = regressor.feature_importances_
    elif hasattr(regressor, "coef_") :
        feature_importance = regressor.coef_
    elif regressor_name == "SymbolicRegressor" :
        # TODO actually, save the whole population, and/or count frequency of variables inside population
        # the current population should be accessible under regressor._programs
        with open(file_name[:-4] + ".txt", "w") as fp :
            programs = regressor._programs[-1]
            for program in programs :
                if program is not None :
                    fp.write(str(program) + "\n")
        return
    elif regressor_name == "PySRRegressor" :
        regressor.equations_.to_csv(file_name)
        return
    else :
        print("No method for determining feature importance detected, skipping...")
        return
    
    # let's get the relative feature importance
    df_dictionary = {"feature_name" : feature_names, "importance" : feature_importance}
    
    # create dataframe and save to disk
    df = pd.DataFrame.from_dict(df_dictionary)
    df.to_csv(file_name, index=False)        
    
    return

if __name__ == "__main__" :
    
    # load and preprocess dataset
    xlsx_file_name = "../data/230824_Processed_Data_Set_Pipe_Socket_Bend_Set1_CorrMax_spec_Fouling_Mass_DAsymbolic.xlsx"
    print("Reading file \"%s\"..." % xlsx_file_name)
    df = pd.read_excel(xlsx_file_name, sheet_name="Data and dimensionless numbers", header=1)
    print(df)
    
    # prepare data for a first regression: take care of the categorical columns
    # and of potentially missing values
    c_missing_values = ["LD_Ratio [-]"]
    for c in c_missing_values :
        df[c].replace({'none': 0}, inplace=True)
        
    # also, BEFORE going over the other categorical columns, we replace the "Fitting_type"
    # categorical variables as such: {"Bend" : 90, "Pipe_socket" : 180}
    df["Fitting_type"].replace({'Bend' : 90, 'Pipe_Socket' : 180}, inplace=True)
    df["Fitting_type.1"].replace({'Bend' : 90, 'Pipe_Socket' : 180}, inplace=True)
    
    categorical_columns = [ c for c in df.columns if c not in df._get_numeric_data().columns ]
    print("Categorical columns identified:", categorical_columns)

    # for each categorical columns, replace values with integers
    for categorical_column in categorical_columns :

        # get unique values
        unique_values = df[categorical_column].unique()
        print("Unique values in column \"%s\": %s" % (categorical_column, str(unique_values)))

        # create dictionary unique_value -> index
        replacement_dict = { unique_values[index] : index for index in range(0, len(unique_values)) }

        # replace values
        df[categorical_column].replace(replacement_dict, inplace=True)
        
    # end of preprocessing; now we create the experiments from the dataset
    experiments = create_experiments(df)
    
    # set name of the folder that will contain the results of the experiments
    folder_name = "results_230824_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    
    # iterate over the experiments
    for experiment in experiments :
        
        experiment_name = experiment["name"]
        feature_names = experiment['feature_names']
        print("Starting experiment \"%s\"..." % experiment_name)
        
        X = experiment["X"]
        y = experiment["y"].ravel()
        
        # ten-fold cross-validation
        # start a classical cross-validation
        ss = ShuffleSplit(n_splits=10, random_state=42)
        
        # classifiers
        #classifiers = [RandomForestRegressor, ExtraTreesRegressor, ExtraTreeRegressor, XGBRegressor, LinearRegression]#, SymbolicRegressor]
        
        # this is a more complex experiment, where we are trying to get a lot of
        # regressors, but remove the ones that require special stuff (e.g. quantile regressors,
        # or multitask regressors)
        candidate_regressors = all_estimators(type_filter="regressor")
        classifiers = [r_class for r_name, r_class in candidate_regressors 
                       if not r_name.startswith("Multi") and not r_name.startswith("Voting")
                       and not r_name.startswith("CCA") and not r_name.startswith("Gamma")
                       and not r_name.startswith("Huber") and not r_name.startswith("Isotonic")
                       and not r_name.startswith("PLSCanonical") and not r_name.startswith("PLSRegression")
                       and not r_name.startswith("Poisson") and not r_name.startswith("RegressorChain")
                       and not r_name.startswith("StackingRegressor") and not r_name.startswith("Tweedie")]
        
        # add CatBoost, LightGBM, XGBoost and Symbolic Regressor (gplearn)
        classifiers.extend([SymbolicRegressor, XGBRegressor, LGBMRegressor, CatBoostRegressor])
        
        # and now we IGNORE everything and just put the regressors we are interested in
        #classifiers = [LinearRegression, SymbolicRegressor, RandomForestRegressor]
        #classifiers = [LinearRegression, RandomForestRegressor]
        
        # this is special stuff to try the PySR regressor
        from pysr import PySRRegressor
        classifiers = [LinearRegression, PySRRegressor, RandomForestRegressor]
        #classifier = [RandomForestRegressor]
        
        print("Regressors picked for the experiments:", classifiers)
        
        # metrics to be used
        metrics = {"R2" : r2_score, "Q2" : r2_score, "MSE_train" : mean_squared_error,
                   "MSE_test" : mean_squared_error, "MAE_test" : mean_absolute_error, 
                   "EV_test" : explained_variance_score, "MAPE_test" : mean_absolute_percentage_error }
        
        # dictionary that will store the results
        dict_results = dict()
        
        # just before starting the cross-validation, create a folder to store the results
        if not os.path.exists(folder_name) : os.makedirs(folder_name)
        
        # create a sub-folder for the experiment
        experiment_folder_name = os.path.join(folder_name, experiment_name)
        if not os.path.exists(experiment_folder_name) : os.makedirs(experiment_folder_name)
        
        # write some information about the experiment in a .txt file
        write_experiment_information(experiment, os.path.join(experiment_folder_name, "information.txt"))
        
        print("Starting cross-validation...")
        for index_fold, [train_index, test_index] in enumerate(ss.split(X)) :
            
            print("Now analyzing fold #%d..." % index_fold)
        
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
        
            # normalization
            scaler_X = StandardScaler()
            X_train = scaler_X.fit_transform(X_train)
            X_test = scaler_X.transform(X_test)
        
            scaler_y = StandardScaler()
            y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))
            y_test = scaler_y.fit_transform(y_test.reshape(-1, 1))
        
            # TODO change random seed for each fold? but we will have to store them
            fold_random_seed = 42 * (index_fold + 1)
        
            # test all classifiers
            for classifier_class in classifiers :
        
                #classifier = classifier_class()
                classifier = initialize_regressor(classifier_class, random_seed=fold_random_seed, feature_names=feature_names)
                classifier_name = classifier.__class__.__name__
        
                print("Now testing classifier \"%s\"..." % classifier_name)
                classifier.fit(X_train, y_train.ravel())
                y_test_pred = classifier.predict(X_test)
                y_train_pred = classifier.predict(X_train)
        
                # store results
                if classifier_name not in dict_results :
                    dict_results[classifier_name] = { k : [] for k in metrics }
                    dict_results[classifier_name]["y_test_pred"] = []
                    dict_results[classifier_name]["y_test_true"] = []
                    dict_results[classifier_name]["y_train_pred"] = []
                    dict_results[classifier_name]["y_train_true"] = []
        
                for metric_name, metric_function in metrics.items() :
                    metric_score = -100.0
                    if metric_name == "R2" or metric_name == "MSE_train" :
                        metric_score = metric_function(y_train, y_train_pred)
                    else :
                        metric_score = metric_function(y_test, y_test_pred)
                    dict_results[classifier_name][metric_name].append(metric_score)
        
                    if metric_name == "Q2" :
                        print("Classifier \"%s\": R2 test (Q2) = %.4f" % (classifier_name, metric_score))
                        
                # add complete information on points predicted and true in the test set
                # (used later for predicted vs true plots)
                dict_results[classifier_name]["y_test_pred"].extend(scaler_y.inverse_transform(y_test_pred.reshape(-1,1)))
                dict_results[classifier_name]["y_test_true"].extend(scaler_y.inverse_transform(y_test))
                dict_results[classifier_name]["y_train_pred"].extend(scaler_y.inverse_transform(y_train_pred.reshape(-1,1)))
                dict_results[classifier_name]["y_train_true"].extend(scaler_y.inverse_transform(y_train))
                
                # save feature importance information
                save_regressor_feature_importance(classifier, experiment["feature_names"], 
                                                  os.path.join(experiment_folder_name, classifier_name + "_fold_%d.csv" % index_fold))
                
                # for PySRRegressor, save the best equation as a Latex file
                if classifier_name == "PySRRegressor" :
                    with open(os.path.join(experiment_folder_name, classifier_name + "_best_equation_%d.tex" % index_fold), "w") as fp :
                        fp.write(classifier.latex())
                
            # end of loop on each classifier
        
        # end of loop on each fold
        
        # now, get the overall results in a CSV-readable form
        dict_final_results = { "regressor" : [] }
        for k in metrics :
            dict_final_results[k] = []
        
        for classifier_name, classifier_metrics in dict_results.items() :
            dict_final_results["regressor"].append(classifier_name)
        
            for metric_name in metrics :
                metric_scores = classifier_metrics[metric_name]
                dict_final_results[metric_name].append("%.4f +/- %.4f" % (np.mean(metric_scores), np.std(metric_scores)))
        
        print("Saving final results to CSV...")
        df_results = pd.DataFrame.from_dict(dict_final_results)
        df_results.to_csv(os.path.join(experiment_folder_name, "results.csv"), index=False)
        
        print("Creating predicted vs measured plots...")
        metric_plots = "Q2"
        for classifier_name, classifier_metrics in dict_results.items() :
            
            y_pred = classifier_metrics["y_test_pred"]
            y_true = classifier_metrics["y_test_true"]
            figure_title = classifier_name
            
            # find performance
            classifier_index = 0
            while dict_final_results["regressor"][classifier_index] != classifier_name :
                classifier_index += 1
            classifier_performance = dict_final_results[metric_plots][classifier_index]
            
            figure_title += " (Q2=" + classifier_performance + ")"
            
            create_predicted_vs_measured_plot(y_true, y_pred, figure_title, os.path.join(experiment_folder_name, classifier_name + ".png"))
            
            # also save results in a CSV file
            csv_file_name = os.path.join(experiment_folder_name, classifier_name + "_predicted_vs_observed.csv")
            predicted_vs_observed_dict = {"predicted" : [x[0] for x in y_pred], "observed" : [x[0] for x in y_true]}
            df_predicted_vs_observed = pd.DataFrame.from_dict(predicted_vs_observed_dict)
            df_predicted_vs_observed.to_csv(csv_file_name, index=False)
            
        print("Creating feature importance plots...")
        regressor_names = [n for n in dict_results if n != "SymbolicRegressor" and n != "PySRRegressor"] # SymbolicRegressor has no feature importance files
        create_feature_importance_barplots(regressor_names, experiment_folder_name)
        