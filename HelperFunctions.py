"""
Holds all of the helper functions used to perform building and analysis of the RDE performance models.
Scripts will leverage these functions.
"""

# native packages
import copy
from collections import OrderedDict
import os

# 3rd party packages
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, ExpSineSquared, Matern, RBF, RationalQuadratic, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error
import yaml


def parse_kriging_data_info(input_file_name):
    """
    Given a yaml file, parse the input data and output data definitions. Normalize the data given the yaml definitions.
    Also, the hyperparameters for the Kriging model are parsed.

    :param input_file_name: The yaml file containing model data subsets, and Kriging model hyperparameters.
    :return:
        Normalized input data in a pandas data frame
        Normalized output data in a pandas data frame
        Normalization scheme for the input data in a pandas data frame
        Normalization scheme for the output data in a pandas data frame
        Dictionary of Kriging hyper-parameters
        Dictionary of meta data for experiment
    """

    # try and open the yml file
    with open(input_file_name, "r") as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # extract the input, and output variables
    df_input_norm, df_output_norm, input_data_df, output_data_df = parse_training_data(data)

    # extract kriging hyper parameter information
    krig_data = data['model_params']

    # extract the meta data
    meta_data = data['meta_data']

    # make folders
    create_folders(meta_data)

    # save normalization information
    input_data_df.to_csv(os.path.join('Trial_'+str(meta_data['trial_num']),'input_norm_data.csv'))
    output_data_df.to_csv(os.path.join('Trial_' + str(meta_data['trial_num']), 'output_norm_data.csv'))

    return df_input_norm, df_output_norm, input_data_df, output_data_df, krig_data, meta_data


def parse_training_data(data):
    """
    Extracts and normalizes the input and output data that is defined vai the input file.

    :param data: The input yaml file that has been converted into a dictionary.
    :return:
        Normalized input data in a pandas data frame
        Normalized output data in a pandas data frame
        Normalization scheme for the input data in a pandas data frame
        Normalization scheme for the output data in a pandas data frame
    """

    input_data_dict = data['data']['input']
    output_data_dict = data['data']['output']

    # parse input data
    input_data_df = build_partial_norm_df(input_data_dict)

    # parse output data
    output_data_df = build_partial_norm_df(output_data_dict)

    # load data set and produce normalzied data frame
    df_input_norm, df_output_norm = load_data_from_csv(data['data']['file_name'], input_data_df, output_data_df)

    return df_input_norm, df_output_norm, input_data_df, output_data_df


def build_partial_norm_df(data_dict):
    """
    Initializes and creates the data frame that defines key values for normalizing the data. This data frame is
    used later when the normalization is created. The min and max data are not included in the data.

    :param data_dict: Dictionary that has user defined parameters for normalizing the data.
    :return: A built pandas data frame that contains the normalization
    """
    df_dicts = []
    for name, value in data_dict.items():
        tmp_dict = OrderedDict(value)
        # add min and max
        tmp_dict['min'] = np.nan
        tmp_dict['max'] = np.nan
        # store for future conversion into data frame
        df_dicts.append(tmp_dict)

    # convert to a data frame from all of the variables.
    data_df = pd.DataFrame(df_dicts)

    return data_df


def load_data_from_csv(data_file_name, input_norm_vars, output_norm_vars):
    """
    Give a file name of csv data that has the measured data, and initialized normalization data frames, the data is
    loaded into a data frame and the normalization data frames are also fully populated.

    :param data_file_name: Name of the csv file of the raw data. This is defined in an input yaml file.
    :param input_norm_vars: The partially initialized input variables normalization data frame
    :param output_norm_vars: The partially initialized output variable normalization data frame
    :return:
        Normalized input data for building the model. In a pandas data frame.
        Normalized output data for the model targets. In a pandas data frame.
    """
    # load the csv file
    df = pd.read_csv(data_file_name)

    # get the max and min values in the data set
    add_data_bounds(input_norm_vars, df)
    add_data_bounds(output_norm_vars, df)

    # create the normalized data set
    df_input_norm = normalize_data(input_norm_vars, df)
    df_output_norm = normalize_data(output_norm_vars, df)

    return df_input_norm, df_output_norm


def add_data_bounds(df, data):
    """
    Adds the maximum and minimum of the data set to the normalization data frame

    :param df: Normalization data frame holding key values for normalizing. This is modified in place.
    :param data: The data frame holding the data set.
    :return:
    """
    # supress warnings for save the max and min values
    pd.options.mode.chained_assignment = None  # default='warn'

    names = list(df['name'])
    for name in names:
        tmp_max = float(df['norm_max'][df['name'] == name].iloc[0])
        tmp_min = float(df['norm_min'][df['name'] == name].iloc[0])
        if tmp_min > tmp_max:
            raise ValueError('The norm min must be less than the norm max value. They are not for ' + str(name))
        df['max'].iloc[df['name'] == name] = data[name].max()
        df['min'].iloc[df['name'] == name] = data[name].min()


def normalize_data(df, data):
    """
    Normalizes the data by the values defined in a normalization data frame.
    This linearly scales the data in a column wise fashion between two values.

    :param df: The data frame with the normalization scheme.
    :param data: Data to be normalized.
    :return: A data frame of normalized data.
    """
    df_norm = pd.DataFrame()
    input_names = list(df['name'])
    for name in input_names:
        col_data = data[name]
        max_v = float(df['max'][df['name'] == name].iloc[0])
        min_v = float(df['min'][df['name'] == name].iloc[0])
        norm_max = float(df['norm_max'][df['name'] == name].iloc[0])
        norm_min = float(df['norm_min'][df['name'] == name].iloc[0])
        col_norm = ((col_data - min_v) / (max_v - min_v)) * (norm_max - norm_min) + norm_min

        df_norm[name] = col_norm
    return df_norm


def create_n_folds(x_data, y_data, kriging_hp):
    """
    Given the input data, it is randomized and split into N folds for n fold cross validation.

    :param x_data: input data dataframe
    :param y_data: output data dataframe
    :param kriging_hp: dictionary with kriging model hyper parameters
    :return:
        input data in n folds
        output data in n folds
    """
    # randomize the order of the data
    x_random = x_data.sample(frac=1,random_state=kriging_hp['seed'])
    y_random = y_data.sample(frac=1, random_state=kriging_hp['seed'])

    # break up the data into n folds
    n_folds = kriging_hp['n_folds']
    x_folds = np.array_split(x_random,n_folds)
    y_folds = np.array_split(y_random, n_folds)

    return x_folds, y_folds


def create_folders(meta_dict):
    """
    Create a folder for saving the models and graphs for the experiment

    :param meta_dict: Dictionary that has data about where to save the models
    :return:
    """

    # create base folder
    if not os.path.isdir('Trial_'+str(meta_dict['trial_num'])):
        os.mkdir('Trial_'+str(meta_dict['trial_num']))


def run_n_fold_cross_validation(x_data, y_data, kriging_hp, meta_dict):
    """
    Runs N-fold cross validation building N models. The error for each model and the standard deviation of the model
    at each of the sample points are found. Each model is also saved.

    :param x_data: A vector of N-packets of input data. Each packet is unique.
    :param y_data: A vector of N-packets of output data. Each packet is unique.
    :param kriging_hp: The hyper-parameters for building the Kriging model
    :param meta_dict: A dictionary that has information for where to save the model.
    :return:
    """

    rmse_vec = np.zeros(len(x_data))
    mae_vec = np.zeros_like(rmse_vec)
    avg_std = np.zeros_like(rmse_vec)
    for i, _ in enumerate(x_data):

        print('Training model {:d} of {:d}'.format(i+1,len(x_data)))

        # validation data
        x_valid = x_data[i]
        y_valid = y_data[i]

        # training data
        x_train = [data for j,data in enumerate(x_data) if j != i]
        y_train = [data for j,data in enumerate(y_data) if j != i]

        x_train_df = pd.concat(x_train)
        y_train_df = pd.concat(y_train)

        # build the model
        model = build_kriging_model(x_train_df, y_train_df, kriging_hp)

        # save the model
        joblib.dump(model,os.path.join('Trial_'+str(meta_dict['trial_num']),'Model_'+str(i+1)+'.pkl'))

        # get the error of the validation set
        rmse, mae, mean, std = get_error(model,x_valid,y_valid)

        # store the models results
        rmse_vec[i] = rmse
        mae_vec[i] = mae
        avg_std[i] = np.mean(std)

    print('The average RMSE over the N folds is: {0:.5f}'.format(np.mean(rmse_vec)))
    return rmse_vec, mae_vec, avg_std


def build_kriging_model(train_x, train_y, kriging_hp):
    """
    Builds a Kriging model.

    :param train_x: The normalized input data in a pandas data frame
    :param train_y: The normalized output data in a pandas data frame
    :param kriging_hp: Dictionary of hyper parameters for the Kriging model
    :return:
        The built Kriging model
    """

    # build the kernel of the Kriging model
    kernel = None
    kernel_dict = kriging_hp['kernel']
    if kernel_dict['type'] == 'DotProduct':
        bounds = kernel_dict['sigma_scale_bounds'].split(',')
        kernel = DotProduct(sigma_0=float(kernel_dict['sigma']), sigma_0_bounds=(float(bounds[0]), float(bounds[1])))
    elif kernel_dict['type'] == 'ExpSinSquared':
        bounds = kernel_dict['length_scale_bounds'].split(',')
        p_bounds = kernel_dict['periodicity_scale_bounds'].split(',')
        kernel = ExpSineSquared(length_scale=float(kernel_dict['length_scale']), periodicity=float(kernel_dict['periodicity']),
                        length_scale_bounds=(float(bounds[0]), float(bounds[1])), periodicity_bounds=(float(p_bounds[0]), float(p_bounds[1])))
    elif kernel_dict['type'] == 'Matern':
        bounds = kernel_dict['length_scale_bounds'].split(',')
        kernel = 1 * Matern(length_scale=float(kernel_dict['length_scale']),
                         length_scale_bounds=(float(bounds[0]), float(bounds[1])), nu=kernel_dict['nu'])
    elif kernel_dict['type'] == 'RationalQuadratic':
        bounds = kernel_dict['length_scale_bounds'].split(',')
        alpha_bounds = kernel_dict['alpha_scale_bounds'].split(',')
        kernel = RationalQuadratic(length_scale=float(kernel_dict['length_scale']), alpha=float(kernel_dict['alpha']), length_scale_bounds=(float(bounds[0]), float(bounds[1])),
                          alpha_bounds=(float(alpha_bounds[0]), float(alpha_bounds[1])))
    elif kernel_dict['type'] == 'RBF':
        bounds = kernel_dict['length_scale_bounds'].split(',')
        kernel = 1 * RBF(length_scale=float(kernel_dict['length_scale']), length_scale_bounds=(float(bounds[0]), float(bounds[1])))

    # initialize the model
    gaussian_process = GaussianProcessRegressor(kernel=kernel, alpha=float(kernel_dict['std'])
                                                , n_restarts_optimizer=int(kernel_dict['n_restarts']))

    # train the model
    gaussian_process.fit(train_x.values, train_y.values)

    return gaussian_process


def get_error(model, x_test, y_test):
    """
    Get the RMSE, MAE errors and the standard deviation at the test points

    :param model: A built Kriging model
    :param x_test: The normalized input test data
    :param y_test: The normalized output test data
    :return:
        RMSE for the sample points
        MAE for the sample points
        The average prediction at the sample points
        The models standard deviation at the sample points
    """

    mean, std = model.predict(x_test.values, return_std=True)

    mae = mean_absolute_error(y_test,mean)
    rmse = mean_squared_error(y_test,mean,squared=False)

    return rmse, mae, mean, std


def graph_error(rmse, mae, avg_std, meta_dict):
    """
    Graphs the RMSE, MAE, and average standard deviations measured at the validation data points from the n-fold cross
    validation.

    :param rmse: Vector of RMSE, one for each fold in the N-fold cross validation
    :param mae: Vector of MAE, one for each fold in the N-fold cross validation
    :param avg_std: Vector of average standard deviations, one for each fold in the N-fold cross validation
    :param meta_dict: Dictionary of information used for saving the graph
    :return:
    """

    sns.set_theme()
    fig_size = meta_dict['fig_size'].split(',')
    fig = plt.figure(0, figsize=(float(fig_size[0]), float(fig_size[1])))
    # create error graph
    ax1 = fig.add_subplot(121)
    bar_width = 0.33
    br1 = [i for i in range(len(rmse))]
    br2 = [i + bar_width for i in range(len(rmse))]
    ax1.bar(br1,rmse,label='rmse',width=bar_width)
    ax1.bar(br2, mae,label='norm mae',width=bar_width)
    ax1.set_xticks([r + bar_width/2 for r in range(len(rmse))],
               [i+1 for i in range(len(rmse))])
    ax1.legend()
    ax1.set_xlabel('Model Number of Fold [-]')
    ax1.set_ylabel('RMSE [-]')

    # add std graph
    ax2 = fig.add_subplot(122)
    br3 = [i for i in range(len(avg_std))]
    ax2.bar(br3,avg_std,label='avg $\sigma$',width=bar_width)
    ax2.legend()

    plt.suptitle('Error and Average Standard Deviation A Test Data Samples for N-Fold Cross Validation')
    plt.tight_layout()
    plt.savefig(os.path.join('Trial_'+str(meta_dict['trial_num']),'N_Fold_Error'+str(meta_dict['file_type'])))


def parse_graphing_data(file_name):

    # try and open the yml file
    with open(file_name, "r") as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # parse graphing constraints
    constant_vars_df, var_vars_df = parse_graph_constraints(data)

    # load kriging model
    meta_data = data['meta_data']
    model = load_kriging_model(meta_data['trial_num'],meta_data['model_num'])

    # load normalization schemes
    input_norm_df = pd.read_csv(os.path.join('Trial_' + str(meta_data['trial_num']), 'input_norm_data.csv'))
    output_norm_df = pd.read_csv(os.path.join('Trial_' + str(meta_data['trial_num']), 'output_norm_data.csv'))

    return model, constant_vars_df, var_vars_df, input_norm_df, output_norm_df, meta_data


def parse_graph_constraints(data):

    # parse variables to hold constant
    dict_lst = []
    for name, value in data['constant_vars'].items():
        tmp_dict = {'name':value['name'], 'value':value['value']}
        dict_lst.append(tmp_dict)
    constant_vars_df = pd.DataFrame(dict_lst)

    # parse the variables that vary
    dict_lst = []
    for name, value in data['vary_vars'].items():
        tmp_dict = {'name':value['name'], 'max':value['max'], 'min':value['min']}
        dict_lst.append(tmp_dict)

    if len(dict_lst) > 2:
        raise ValueError('Only 1 or two variables are allowed to vary while graphing a model')

    var_vars_df = pd.DataFrame(dict_lst)

    return constant_vars_df, var_vars_df


def load_kriging_model(trial_num, model_num):
    return joblib.load(os.path.join('Trial_'+str(trial_num),'Model_'+str(model_num)+'.pkl'))


def graph_model(model, constant_vars_df, var_vars_df, input_norm_df, output_norm_df, meta_data):

    if len(var_vars_df) == 1:
        graph_1d_slice(model, constant_vars_df, var_vars_df, input_norm_df, output_norm_df, meta_data)
    elif len(var_vars_df) == 2:
        graph_2d_slice(model, constant_vars_df, var_vars_df, input_norm_df, output_norm_df, meta_data)
    else:
        raise ValueError('Only 1 or two variables are allowed to vary while graphing a model')


def graph_1d_slice(model, constant_vars_df, var_vars_df, input_norm_df, output_norm_df, meta_data):

    # create the sample data
    x_data = np.linspace(var_vars_df['min'].iloc[0], var_vars_df['max'].iloc[0], meta_data['sample_density'])

    cols = list(input_norm_df['name'])
    data = np.zeros((len(x_data), len(cols)))
    sample_data = pd.DataFrame(data=data, columns=cols)
    # load in varing data
    sample_data[var_vars_df['name'].iloc[0]] = x_data

    # load in constant data
    for index, row in constant_vars_df.iterrows():
        sample_data[row['name']] = row['value']

    # normalize the data
    sample_data_norm = copy.deepcopy(sample_data)
    for col_name in sample_data_norm.columns:
        col_data = sample_data_norm[col_name]
        max_v = float(input_norm_df[input_norm_df['name'] == col_name]['max'])
        min_v = float(input_norm_df[input_norm_df['name'] == col_name]['min'])
        norm_max = float(input_norm_df[input_norm_df['name'] == col_name]['norm_max'])
        norm_min = float(input_norm_df[input_norm_df['name'] == col_name]['norm_min'])

        sample_data_norm[col_name] = ((col_data - min_v) / (max_v - min_v)) * (norm_max - norm_min) + norm_min

    # make predictions
    mean, std = model.predict(sample_data_norm, return_std=True)

    # un normalize output
    col_name = list(output_norm_df['name'])[0]
    # output_data_norm = pd.DataFrame(data=np.zeros((len(sample_data_norm),)),columns=[col_name])
    col_data = mean
    max_v = float(output_norm_df[output_norm_df['name'] == col_name]['max'].iloc[0])
    min_v = float(output_norm_df[output_norm_df['name'] == col_name]['min'].iloc[0])
    norm_max = float(output_norm_df[output_norm_df['name'] == col_name]['norm_max'].iloc[0])
    norm_min = float(output_norm_df[output_norm_df['name'] == col_name]['norm_min'].iloc[0])

    output_data_un_norm = (col_data - norm_min) / (norm_max - norm_min) * (max_v - min_v) + min_v
    col_data = std
    output_std_un_norm = (col_data - norm_min) / (norm_max - norm_min) * (max_v - min_v) + min_v

    sns.set_theme()
    plt.rcParams.update({'font.size': meta_data['font_size']})
    fig_size = meta_data['fig_size'].split(',')
    fig = plt.figure(0, figsize=(float(fig_size[0]), float(fig_size[1])))
    ax1 = fig.add_subplot(1, 1, 1)
    x_graph = np.array(sample_data[var_vars_df['name'].iloc[0]])
    ax1.plot(x_graph, output_data_un_norm)
    ax1.fill_between(x_graph,output_data_un_norm+output_std_un_norm,output_data_un_norm-output_std_un_norm,facecolor='tab:blue', alpha=0.3, label='1 $\sigma$')
    ax1.set_xlabel(input_norm_df['name'].iloc[0])
    #ax1.set_ylabel(input_norm_df['name'].iloc[1])
    ax1.set_ylabel(str(col_name))
    ax1.legend()
    ax1.set_title(meta_data['title'])
    plt.tight_layout()
    file_name = meta_data['file_name']
    plt.savefig(os.path.join('Trial_' + str(meta_data['trial_num']), file_name))

def graph_2d_slice(model, constant_vars_df, var_vars_df, input_norm_df, output_norm_df, meta_data):

    # create the sample data
    x_data = np.linspace(var_vars_df['min'].iloc[0],var_vars_df['max'].iloc[0],meta_data['sample_density'])
    y_data = np.linspace(var_vars_df['min'].iloc[1], var_vars_df['max'].iloc[1], meta_data['sample_density'])

    cols = list(input_norm_df['name'])
    data = np.zeros((len(x_data)*len(y_data),len(cols)))
    sample_data = pd.DataFrame(data=data,columns=cols)
    # load in varing data
    xd, yd = np.meshgrid(x_data,y_data)
    xd = np.reshape(xd,(len(xd[0,:])*len(xd[:,0]),))
    yd = np.reshape(yd, (len(yd[0, :]) * len(yd[:, 0]),))
    sample_data[var_vars_df['name'].iloc[0]] = xd
    sample_data[var_vars_df['name'].iloc[1]] = yd

    # load in constant data
    for index, row in constant_vars_df.iterrows():
        sample_data[row['name']] = row['value']

    # normalize the data
    sample_data_norm = copy.deepcopy(sample_data)
    for col_name in sample_data_norm.columns:

        col_data = sample_data_norm[col_name]
        max_v = float(input_norm_df[input_norm_df['name'] == col_name]['max'])
        min_v = float(input_norm_df[input_norm_df['name'] == col_name]['min'])
        norm_max = float(input_norm_df[input_norm_df['name'] == col_name]['norm_max'])
        norm_min = float(input_norm_df[input_norm_df['name'] == col_name]['norm_min'])

        sample_data_norm[col_name] = ((col_data - min_v) / (max_v - min_v)) * (norm_max - norm_min) + norm_min

    # make predictions
    mean, std = model.predict(sample_data_norm, return_std=True)

    # un normalize output
    col_name = list(output_norm_df['name'])[0]
    #output_data_norm = pd.DataFrame(data=np.zeros((len(sample_data_norm),)),columns=[col_name])
    col_data = mean
    max_v = float(output_norm_df[output_norm_df['name'] == col_name]['max'].iloc[0])
    min_v = float(output_norm_df[output_norm_df['name'] == col_name]['min'].iloc[0])
    norm_max = float(output_norm_df[output_norm_df['name'] == col_name]['norm_max'].iloc[0])
    norm_min = float(output_norm_df[output_norm_df['name'] == col_name]['norm_min'].iloc[0])

    output_data_un_norm = (col_data - norm_min)/(norm_max-norm_min)*(max_v-min_v)+min_v
    col_data = std
    output_std_un_norm = (col_data - norm_min) / (norm_max - norm_min) * (max_v - min_v) + min_v

    sns.set_theme()
    plt.rcParams.update({'font.size': meta_data['font_size']})
    fig_size = meta_data['fig_size'].split(',')
    fig = plt.figure(0, figsize=(float(fig_size[0]), float(fig_size[1])))
    ax1 = fig.add_subplot(1,2,1)
    x_graph = np.array(sample_data[var_vars_df['name'].iloc[0]])
    y_graph = np.array(sample_data[var_vars_df['name'].iloc[1]])
    cs = ax1.tricontourf(x_graph,y_graph,output_data_un_norm,cmap='plasma')
    plt.colorbar(cs)
    ax1.set_xlabel(var_vars_df['name'].iloc[0])
    ax1.set_ylabel(var_vars_df['name'].iloc[1])
    ax1.set_title('Mean Prediction of '+str(col_name))

    ax2 = fig.add_subplot(1,2,2)
    cs = ax2.tricontourf(x_graph, y_graph, output_std_un_norm, cmap='plasma')
    plt.colorbar(cs)
    ax2.set_xlabel(var_vars_df['name'].iloc[0])
    ax2.set_ylabel(var_vars_df['name'].iloc[1])
    ax2.set_title('Standard Deviation in Model of ' + str(col_name))

    plt.tight_layout()
    file_name = meta_data['file_name']
    plt.savefig(os.path.join('Trial_'+str(meta_data['trial_num']),file_name))


if __name__ == '__main__':

    # Don't allow running from here
    pass