# PolStance - a Political Stance Detector

The PolStance project take advantage of the power of transformer-based models to classify political stances in text.

It works by a fine-tuned BERT model on a dataset of political statements labeled with their corresponding stances. The labeling is done by Gemini flash lite model.

This repository contains both the training and inference code, as well as how the dataset is obtained. The model is trained on Chinese and classifies the stances into "KMT", "DPP", and "Neutral".

The result has the accuracy of 72%, but the labeling quality is off, and the model seems to be unable to understand against statements. More work is needed to improve the model performance.

The project is currently in update phase, upgrading from analyzing title to analyze article content. Legacy version of the project is deployed onto Huggingface spaces for easy access. Click [here](https://huggingface.co/spaces/abcd1234davidchen/PolStance) to try it out.

## Status
 - [x] Implemented article crawling from multiple news websites
 - [x] Implemented data cleaning
 - [ ] Implemented data labeling using "Voting" mechanism from multiple LLMs
 - [ ] Implemented model training using BERT
 - [ ] Setup command line interface for managing the pipeline
 - [ ] Setup web app for easy access

## Crawlers and Data cleaning
The crawlers are implemented in `getArticle.py`. The script uses selenium for web scraping and BeautifulSoup for HTML parsing. It includes functions to crawl titles from multiple news websites. The data cleaning functions are also included in this script. The base cleaning is implemented by removing short or empty titles and articles. Roughly 30k titles can be obtained after crawling and cleaning.

## Data Labeling(legacy: Title Labeling)
The data labeling is done using Gemini flash lite model. The labeling function is implemented in `getTitleLabel.py`. The script reads the cleaned data and uses the Gemini flash model to label each title with its stances. The labeled data is saved into the same db. The labeling turns out the data has the ratio of 7:5:7 for KMT/DPP/Neutral.

## Model Training(legacy: Title-based Model Training)
The model training is implemented in `trainModel.py`. The script uses the Hugging Face Transformers library to fine-tune a BERT model on the labeled dataset. The model adds layers for classification and uses cross-entropy loss for training. The trained model is saved for later use. The process is optimized for GPU, and is done on my base line M3 Pro MacBook Pro with MPS backend. The training achieves an accuracy of around 72% on the validation set.

## Inference(legacy: Title-based Inference)
The inference pipeline is implemented in `inference.py`. The script loads the trained model and provides a function to predict the stance of a given sentence.

## Web App(legacy: Title-based Web App)
The web app is implemented using Gradio in `app.py`. The app provides a simple interface for users to input a sentence and get the predicted stance from the model. The same file is also used on huggingface spaces for web deployment.

## Requirements
The project is managed through UV. `pyproject.toml` contains the project dependencies. To install the dependencies, run: `uv sync`. `.env` file should contain the Gemini API key for data labeling and huggingface API key for model inference.