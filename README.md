# Replication Package: What is the Carbon Footprint of ML Models on Hugging Face? A Repository Mining Study

## Overview

This paper analyzes the carbon emissions of ML models at Hugging Face, a popular repository for pretrained models. The research goal is to understand the general characteristics and energy efficiency of these models during training, focusing on two main research questions: (1) How do ML model creators measure and report carbon emissions on Hugging Face? and (2) What aspects impact the carbon emissions of training ML models? By analyzing energy reporting practices, correlations between energy consumption and other attributes, and proposing a classification system for energy efficiency, this paper provides insights and recommendations for AI practitioners seeking to optimize energy efficiency in their models.

## Data Collection and Preprocessing

Our first step involves collecting data from Hugging Face, primarily through an automated pipeline using the Hugging Face Hub API and HfApi class, which serves as a Python wrapper for the API. Following data collection, we carefully filter and refine the data. During data preprocessing, we focus on feature engineering, variable standardization/harmonization and one-hot encoding of tags. We create variables such as *co2_reported*, *auto*, *year_month*, and *domain* for filtering, splitting datasets, and analyzing model behavior across domains.  Next, we manually curate the resulting dataset by filtering and completing missing data that could not be automatically retrieved in an alternative dataset. Both the complete and the manually curated dataset will be used for the data analysis.

As a remark, p-values correction on Holm-Bonferroni, which affects the results for both RQs, can be found in HFGeneralAnalysis.ipynb last cell.

## Folder Structure

- `code/`: Contains the Jupyter notebooks with the data extraction, preprocessing, and analysis scripts.
- `datasets/`: Contains the raw, processed, and manually curated datasets used for the analysis.
- `metadata/`: Contains the `tags_metadata.yaml` file used during preprocessing.
- `requirements.txt`: Lists the required Python packages to set up the environment and run the code.

## Setup and Execution

1. Set up a Python virtual environment (optional, but recommended). We used Python 3.10.11 for this study.
2. Install the required Python packages using `pip install -r requirements.txt`.
3. Open the Jupyter notebooks in the `code/` folder and follow the instructions in each notebook to execute the data extraction, preprocessing, and analysis steps.

## Additional Information

This replication package is provided to help researchers and practitioners reproduce and extend the results of the paper. Please feel free to use the provided data and code for your own research, but remember to cite the original paper when doing so. If you encounter any issues or have any questions, please contact the authors for assistance.


