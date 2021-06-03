# -*- coding: utf-8 -*-

from pyfume import *
import pandas as pd
import os
    
low_cluster_nr = 2 # Lowest cluster number
high_cluster_nr = 3 # Highest cluster number
path_old_sales = './data/Previous_house_sales.csv'
path_new_sales = './data/Houses_to_be_sold.csv'
df_new_sales = pd.read_csv(path_new_sales)
df_predicted = pd.DataFrame(columns = [str(nc) + " clusters" for nc in range(low_cluster_nr, high_cluster_nr + 1)])
input_variable_names = list(df_new_sales.columns)[:-1]

# Work around FileExistError
for filename in os.listdir('./simpful_code/'):
    if "simpful" in filename:
        os.remove('./simpful_code/' + filename)
        
for nc in range(low_cluster_nr, high_cluster_nr + 1):
    
    #_______________ MODEL _______________ #
    # Generate Takagi-Sugeno FIS
    FIS = pyFUME(datapath=path_old_sales, nr_clus=nc) # , feature_selection='wrapper'
            
    # Calculate and print the accuracy of the generated model
    MAE=FIS.calculate_error(method="MAE")
    print ("The estimated error of the developed model is:", MAE)
    #____________________________________ #
    
    # Save Simpful code automatically in the right folder
    path = './'
    new_name = './simpful_code/simpful_{}_clusters.py'.format(nc)
    
    # Every time the model runs, the file 'Simpful.py' is automatically generated. Therefore, the following checks if there exist
    # a file named 'Simpful.py' and then changes its name and directory.
    for file in os.listdir(path):
        if file == 'Simpful_code.py':
            os.rename(path + file, new_name)
    
    # Making it easier to produce figures from the Simpful code by writing the following line in 'Simpful.py'
    with open(new_name, 'a') as file:
        file.write('\n\nFS.produce_figure("./figures/simpful_{}_clusters.png")'.format(nc))

    # Extract the model from the FIS object
    model=FIS.get_model()
    
    # Predict prices
    print("\nPredicted prices:")
    predicted = []
    for index, row in df_new_sales.iterrows():
        for name, value in zip(input_variable_names, row):
            model.set_variable(name, value)
        
        predicted.append(model.Sugeno_inference(['OUTPUT'])["OUTPUT"])
        print(model.Sugeno_inference(['OUTPUT'])["OUTPUT"])
    
    # Append to dataframe
    df_predicted[str(nc) + " clusters"] = predicted
# Save predicted values
df_predicted.to_csv("simpful_code/predicted/output.csv", index = False)
    
