"""
This script is for evaluating a model on the test data.
"""

import HelperFunctions as hf

if __name__ == '__main__':

    file_name = 'eval_model_input.yml'

    model, input_data_norm, output_data_norm, input_norm_df, output_norm_df, meta_data = hf.parse_eval_info(file_name)

    hf.graph_error(model, input_data_norm, output_data_norm, input_norm_df, output_norm_df, meta_data)

