# About the Data

* The data was taken from 
https://m.rbi.org.in//scripts/PublicationReportDetails.aspx?ID=242

* Each rule is considered as a record under the the headings:<br>

  1. Enhancing Bank Transparency
  2. Best Practices for Credit Risk Disclosure
  3. Supervision of Financial Conglomerates 
  4. Risk Concentrations Principles
  5. Intra-Group Transactions and Exposures Principles
  6. Principles for the Supervision of Banks’ Foreign Establishments (The Basel Concordat)
  7. Information Flows Between Banking Supervisory Authorities
  8. Minimum Standards for the Supervision of 
International Banking Groups and their Cross-Border Establishments
  9. The Supervision of Cross-Border Banking 
  
* Rules are seperated by their rule number.

# Text Preprocessing

* The line break ‘\n’ is removed from the text, if there exist any.
* Round and square parantheses are removed, if there exist any.
* Text contatined within round paranthesis are removed, if there exist any.
* Text contatined within square paranthesis are removed, if there exist any.
* No summarization was done.
* Duplicate entries and text with word count less than 7 are removed.

# Notebooks:

1)The notebook containing the pre-processing of input 'Text' is **Part1_HuggingFace_Dataset.ipynb**

2A)The notebooks containing the fine-tuning of DistilBERT model for 5 epochs are:

  i.   **Part2_DistilBERT_5_epochs.ipynb**
  ii.  **Part3_Predicting_and_Inference_using_Pipeline_5_epochs.ipynb**

2B)The notebooks containing the fine-tuning of DistilBERT model for 15 epochs are:

  i.   **Part2_DistilBERT_15_epochs.ipynb**
  ii.  **Part3_Predicting_and_Inference_using_Pipeline_15_epochs.ipynb**

2C)The notebooks containing the fine-tuning of DistilBERT model for 30 epochs are:

  i.   **Part2_DistilBERT_30_epochs.ipynb**
  ii.  **Part3_Predicting_and_Inference_using_Pipeline_30_epochs.ipynb**
