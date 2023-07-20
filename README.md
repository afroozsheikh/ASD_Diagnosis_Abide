# Autism_Diagnosis_ABIDE

Autism Spectrum Disorder (ASD) is a psychiatric disorder that leads to communication impairment. My main purpose in this project was to identify ASD patients from healthy controls given fMRI data from the ABIDE1 dataset. ABIDE I involved 17 international sites, sharing previously collected resting state fMRI including 539 from individuals with ASD and 573 from typical controls.

In this project, I focused on the automated detection of autism spectrum disorder using two distinct variants of  graph neural networks including Graph Attention Neural Network (GAT) and GraphSAGE. I utilized the Nilearn library to fetch the ABIDE dataset and then I extracted fMRI time series from ROIs in the atlas.  Then, I constructed a graph using a functional connectivity matrix obtained from calculating the correlation between the time series for each subject and used different centralities and  statistics  of  the  fMRI  time  series  as  the  node  features. The graph nodes represent  brain  ROIs  and  edges  denote  the  interactions  between the regions and their activities
![Image alt text](/img/2.PNG) 

## Baseline
Having built the graphs, I took advantage of two architectures of Graph Neural Networks (Graph Attention Neural Networks and GraphSAGE) for classifying each subject as ASD or healthy control. For the construction of graph neural networks, I used PyTorch and PyTorch Geometric packages. 
![Image alt text](/img/4.PNG)
![Image alt text](/img/3.PNG)

Due to the lack of adequate resources for training the models on the whole dataset,  I decided to use one site in the dataset (UCLA site) to compare the results of GAT and GraphSAGE. The table below compares different metrics for both architectures. Obviously, GAT outperformed GraphSAGE. 

![Image alt text](/img/1.PNG)

----
## Graph Data Augmentation Methods
Graph Data augmentation (GDA) has recently seen increased interest in graph machine learning given its ability to create extra training data and improve model generalization. I used two GDA methods in this project, feature, and subgraph augmentation. 
### Mixup
Mixup merges two images to generate new images with a weighted label





