# Projects in Data Science (2026) - Lizzard(s)
![unnamed](https://github.com/user-attachments/assets/134ff216-6cb9-44fd-af1b-7b25ef04fa5a)

#### Overview

This is the final project repository for group Lizzards in the "Projects in Data Science" course at IT University of Copenhagen, with the following contributors: Simon Friland, Hunor Szabó, Marcus Pedersen, Viky Kapičáková & Sofie Pedersen.

The goal of our project is to create a machine learning algortihm that could classify benign and cancerous lesions. This is a first year's project, therefore the algorithm might not have a good performance that could be used in medical assesment, however our goal    

#### Python environment 

Generally, we used Anaconda python enviroment, to make sure all modules work correctly use 'pip install -r requirements.txt'

### Instructions

- Open 'extract_features.py' file and change data path section to desired.
- In the dataframe section, 'df', change the path if needed, to extract the patient IDs
- Run 'extract_features.py' to generate 'features.csv' in 'data/' folder
- Open 'main.py' and change 'features_path'
- Run the code, which trains the model and saves it to 'predictions/model/name_of_the_model.joblib'
- Set 'load_model' to True and run the code on desired dataset, which should make prediction csv file and confusion matrix in 'results/predictions/' folder

#### File Hierarchy

```
ProjectInDataScience2026_ExamTemplate/
├── data/
│   ├─ features.csv                     # all image file names, ground-truth labels, and chosen features
│   ├─ annotations_combined.csv         # annotations of hair and penmarks
│   │
│   ├── imgs/                           # skin images (to not add on GitHub)
│   │    ├── img_XX1.png
│   │    ├── img_XX2.png
│   │     ......
│   │    └── img_XXX.png
│   │
│   └── masks/                          # masks images (to not add on GitHub)
│        ├── mask_XX1.png
│        ├── mask_XX2.png
│         ......
│        └── mask_XXX.png
│
├── src/
│   ├── __init__.py
│   ├── feature_A.py                    # code for feature A extraction
│   ├── feature_B.py                    # code for feature B extraction
│   ......
│   └── feature_X.py                    # code for feature X extraction
│ 
├── result/
│   ├── figures/                        # Figures used in your report
│   ├── models/                         # Trained models
│   ├── predictions/                    # Probabilities outputed by the models
│   └── reports                         # Files related to the Mandatory assignment
│        ├── report_GROUPEID.pdf
│        └── features_GROUPEID.csv
│ 
├── main.py                             # script to train or evaluate models
└── README.md
```
