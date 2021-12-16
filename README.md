# Drug_Recommender_System
Research Project on Recommender system used for OTC medicines recommendation based on content based filtering

# Steps for Building Recommender model

Download and install the lates version of Python to local system from https://www.python.org/downloads/

To run the recommender system python notebook file, Jupyter Notebook or Google Collab Notebook can be chosen accordingly.

If Jupyter Notebook is chosen, then need to install virual environment like Conda / Anaconda / Miniforge and after installing run the command in CLI "pip install -r requirements.txt"

If Google Collab is chosen, then nothing needs to be done as Google drive has built-in python and required libraries are already installed.

Import the " Drug_Recommender_model.ipynb" notebook file present in this directory and run the entire command.

Dataset csv files are already on Azure Blob Storage, hence no need of saving it again anywhere in local.
The notebook will directly fetch from cloud and process further.

After the model is created , drugs_dict.pkl and similarity.pkl file was downloaded and further fed to heroku cloud for making website.

At this point, all the required files are deployed on cloud and server is up and running fine. The Website can be acessed at 
https://drug-recommend.herokuapp.com/ (it may take 10 seconds for website to open)
