"""
Graphs a model with a user defined slice of the data. 1D or 2D slices are supported.
"""

import HelperFunctions as hf


if __name__ == '__main__':

    # input file name
    file_name = 'graph_model_input_2D.yml'

    # parse the input data
    model, constant_vars_df, var_vars_df, input_norm_df, output_norm_df, meta_data = hf.parse_graphing_data(file_name)

    # graph the data
    hf.graph_model(model, constant_vars_df, var_vars_df, input_norm_df, output_norm_df, meta_data)