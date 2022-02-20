# Requirements
conda create -n syn python=3.8  
conda activate syn  
conda install numpy tqdm scikit-learn
pip install Levenshtein  
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install -c huggingface transformers


# Datasets
The datasets are from <http://www.obofoundry.org/>. This website consists hundred of sub datasets which list the [Term] information and its synonym labels. The synonym entries are labeled data that could be utilized for expermients.  
We can use ontologies.jsonld to catch up all the datasets on the website and filter the useful ones later.  
Also, for convenience, I implement the extraction of synonym entries for every Term on single dataset and the split of dataset(under two different settings).  
See data_process.py for more details.

# Code
I recommand you use pytorch because codes structure based on pytorch is very clear.

## datasets.py
implement the data process and constrcution of datasets. 
Use load_data function to get all the data(eg: name_array, query_array, mentions2id) you will need
the Dataset classes are for biosyn model. If you use torch, you could implement the dataset class for yourself here 

## models.py 
implement the neural networks here

## classifier.py
implement different kinds of classifiers 
here are edit_distance classifier and biosyn classifier

## main.py
set the args and run your algorithms,see args for more details

## evaluator.py
For evaluation. It is a simple version for evaluating the whole performance on all data. You can implement your evaluation based on the classifier(like I do in biosyn)

## criterion.py 
the loss function for biosyn method













