# Introduction
This repository allows for the building of Kriging models. This does not require any
python experience or programming to use these scripts.

# Setup
To setup your environment, please follow the instructions in either "If new to python" or
"If not new to python".

## If new to python

- open a command prompt and navigate to the root folder where you have saved this repository.
- execute the following commands to create virtual environment
	and install required packages. Make sure you have the path to python.exe in your 
	environment variables:
	1. python -m venv venv
	2. venv\Scripts\activate
	3. python -m pip install -r requirements.txt

If you have already run the command "python setup_and_install.py" and you are returning 
to use the scripts at a later date, execute the following command from the root of this
repository to re-activate your python interpreter "venv\Scripts\activate".

## If not new to python
- The required packages are in the requirements.txt file.

# How to build a model
To build a model, please open the "build_model_input.yml" file. In the data section,
The "file_name" field points to a csv file that has RDE data. This csv file should 
include both training and validation data. The example included data is called
"RDRE_data.csv". 
Next, please specify what variables you want as inputs and outputs. The name for each
input variable should be the *exact* column name of the data in the .csv file. Please
leave norm_max and norm_min at there values, as they specify the data will be linearly
transformed to have a range from 0 to 1. Note the names for each variable counts up
in order start at 00, then 01, then 02, etc. These fields must be consecutive. The output
data is only allowed one variable and follows the same format as the inputs.
If you have empty cells in your data set for a variable you are using, you must delete
that row from the input data file that is being loaded.

After that the meta data section may be edited. The trial number denotes a folder to save 
the models and results in. If a folder with that number already exists, training a new 
model will overwrite it. For example if you set the value to 4, a folder called
"Trial_4" will be created and all of the models are supporting data will be saved
there. Do not edit any files here, as one could corrupt the model.
The remaining parts in the meta_data section are for controlling the appearance of the
saved graph.

The last section is the model_params. Here the kernel is defined for the Kriging model.
some example kernels are in "build_model_input.yml" and are commented out. The double
comment marks the different kernels. Please feel to change the values for the fields 
with exception of the 'type' field. These can be used to optimize the model. The 
'std' field is the estimated normalized standard deviation of the data. If this is not
known, please try a few values and note which value minimizes the error. The number of
restarts is how many times the Kriging model is built per fold. A higher number helps
prevent the model from being stuck in a local minima. 

'n_folds' is the number of folds to perform in the n-fold cross validation. It is 
recommended not to change this value, but one can use values between 6-10. Last is the
seed, which is the seed of the random number generator.

Once you are happy with the inputs, build the model by running the command 
"python BuildModel.py". The models, supporting data, and an error graph will be put into
the folder with the specified trial number. The average error (RMSE) will be printed
to the screen. This is the error to minimize when optimizing the Kriging model.

A copy of your input parameters are saved in a file called "Training_input.yml"
for ease of documentation.

# Graphing a model
After training a model, one can visualize it. To do so open either "graph_model_input_1D.yml"
or "graph_model_input_2D.yml". These are example files for graphing. The "vary_vars" section
is where you specify what variables to change in the output graph. Only one or two variables
are allowed to change. The max and min values are the raw values to change between. The
"constant_vars" must include all other input variables not included in "vary_vars" must
be included here. The value to hold constant shall be specified here.

The "meta_data" section has mainly parameters for changing the appearance of the graph. 
"trial_num" is the number of which trial you want to get a model from and graph. Ex.
If you have trials 0, 1, 4, and 10 and want to use a model in trial 4, set this field
to 4. "model_num" is which model in the selected trial to use. One should select this 
number at random between 0 and N with N being the number of folds performed. It grabs
the model with the same number form the specified trail folder as the model to graph.

Once you are happy with the setup, run the command "python GraphModel.py" to generate
the graphs of the model outputs. The graphs will be saved in the corresponding trial
folder.

# Evaluating the model
To evaluate how well the model does, please open the file "eval_model_input.yml".
In the "data" section, change the file (if necessary) to your test data .csv file.
This csv file should be the exact same format as your training .csv file, but should
contain completely unique data. The "meta_data" section works the same as the 
"meta_data" section described in the "Graphing a model" section of the ReadMe.md file.

Once you are happy with the file, run the command "python EvalModel.py" to run the 
analysis. The output graphs are again placed in your corresponding trial folder. The
RMSE and MAE are also printed out to console. This is a true measurement of the models
predictive performance.

