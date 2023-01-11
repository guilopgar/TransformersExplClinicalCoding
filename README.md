# Transformers for Explainable Clinical Coding

We provide a description of the content of the different folders to facilitate the reproduction of the results obtained in this work.

- **datasets**: it contains the link to a zip folder with the corpora from the 3 explainable clinical coding tasks addressed in this study.
- **models**: it contains the link to a zip folder with the weights of all transformer-based models used in this work.
- **results**: it contains the link to a zip folder with all the predictions files obtained from the output of the transformer-based models at inference time.
- **scripts**: for each task, 9 different files have been created: the 5 *Eval-** jupyter notebooks evaluate the performance of the different models using the predictions files from **results** folder, while the 3 python scripts allow to reproduce the obtained results of the models following the *multi-task* and the *hierarchical-task* approaches (MER and MEN), respectively. Additionally, a **-Full_Filter* script has been created for each task to perform the supplementary evaluation of the hierarchical-task MEN approach on the full and filtering setups.
- **utils**: utils scripts in python containing auxiliary functions used in the files from the **scripts** folder.

For any additional information, please contact with guilopgar@uma.es
