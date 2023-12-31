# DocGPT
A sophisticated LLM-based conversational system will extract information from research paper PDFs to create a helpful assistant for researchers during brainstorming sessions.

> Clone Project 
1. Open a new empty directory .
2. Run following commands in cmd .
* git clone https://github.com/sankarapusamhith/DocGPT.git
* cd DocGPT

> Create Virtual Environment
1. Run following commands in cmd .
* pip install virtualenv 
* python -m venv env
* env\Scripts\activate

> Install required libraries
1. Run following command in cmd .
* python -m pip install -r requirements.txt

> Start streamlit app
1. Run following command in cmd .
* streamlit run DocGPT.py


## Limitations in Version 1

- Only PDF format supported 
- In some scenarios we’re getting trimmed form of actual answer because of token issues
- Tabular data extraction aren't providing good enough results
- Multi document option not available yet
- Math Computations won't work

## Version 1 working model

[streamlit-main-2023-07-11-22-07-45.webm](https://github.com/sankarapusamhith/DocGPT/assets/58435062/ca246543-ce01-49fb-bb17-3edd8e727d1b)

## Version 1 results on validation set

[streamlit-val-2023-07-11-22-07-96.webm](https://github.com/sankarapusamhith/DocGPT/assets/58435062/4db19642-1fa2-494a-a436-5621669a1ef0)

## Future Scope 

- Multi Phrase Results 
- Multiple Docs at same time 
- Improve efficiency on tabular data 
- Math Computations 
- Own Fine Tuned Model 
- Should highlight the output on the document 
