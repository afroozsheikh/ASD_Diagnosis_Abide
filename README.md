# Autism_Diagnosis_ABIDE

Autism Spectrum Disorder (ASD) is a psychiatric disorder that leads to communication impairment. My main purpose in this project was to identify ASD patients from healthy controls given fMRI data from ABIDE1 dataset. ABIDE I involved 17 international sites, sharing previously collected resting state fMRI including 539 from individuals with ASD and 573 from typical controls.

In this project, I focused on the automated detection of autism spectrum disorder using two distinct variants of  graph neural networks including Graph Attention Neural Network (GAT) and GraphSAGE. I utilized Nilearn library to fetch the ABIDE dataset and then I extracted fMRI time series from ROIs in the atlas.  Then, I constructed a graph using functional connectivity matrix obtained from calculating the correlation between the time series for each subject. 
![Image alt text](/img/2.PNG) 

Having built the graph, I took advantange of two architectures of Graph Neural Networks (Graph Attention Neural Networks and GraphSAGE) for classifying each subject as ASD or healthy control. For the construction of graph neural networks, I used PyTorch and PyTorch Geometric packages. 
![Image alt text](/img/4.PNG)
![Image alt text](/img/3.PNG)

Due to the lack of adequate resources for training the models on the whole dataset,  I decided to work on one of the site in the dataset.  The table below compares different metrics for both architectures. Obviously, GAT outperformed the GraphSAGE.

![Image alt text](/img/1.PNG)
