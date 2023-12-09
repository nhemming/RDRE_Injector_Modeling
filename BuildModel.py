'''
This script is run to create a Kriging model from provided input data. A yaml file that defines the data and model
hyperparameters are needed to build the model.
'''

import HelperFunctions as hf

if __name__ == '__main__':

    input_yaml_file = 'build_model_input.yml'

    # load and parse the data needed for building a model
    df_input_norm, df_output_norm, input_data_df, output_data_df, kriging_hp, meta_data = hf.parse_kriging_data_info(input_yaml_file)

    # create N folds
    x_fold, y_fold = hf.create_n_folds(df_input_norm, df_output_norm, kriging_hp)

    # build kriging model
    rmse, mae, avg_std = hf.run_n_fold_cross_validation(x_fold, y_fold, kriging_hp, meta_data)
    #hf.build_kriging_model(df_input_norm, df_output_norm, kriging_hp)

    # graph and save N fold cross validation
    hf.graph_error(rmse, mae, avg_std, meta_data)