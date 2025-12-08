# PolStance - a Political Stance Detector

The PolStance project take advantage of the power of transformer-based models to classify political stances in text.

It works by a fine-tuned BERT model on a dataset of political statements labeled with their corresponding stances. The labeling is done by Gemini flash lite model.

This repository contains both the training and inference code, as well as how the dataset is obtained. The model is trained on Chinese and classifies the stances into "KMT", "DPP", and "Neutral".

The result has the accuracy of 72%, but the labeling quality is off, and the model seems to be unable to understand criticism and sarcasm. More work is needed to improve the model performance.

The project is currently in update phase, upgrading from analyzing title to analyze article content. Legacy version of the project is deployed onto Huggingface spaces for easy access. Click [here](https://huggingface.co/spaces/abcd1234davidchen/PolStance) to try it out.

## Status
 - [x] Implemented article crawling from multiple news websites
 - [x] Implemented data cleaning
 - [x] Implemented data labeling using "Voting" mechanism from multiple LLMs
 - [x] Implemented model training using BERT
 - [ ] Setup command line interface for managing the pipeline
 - [x] Setup web app for easy access

## Crawlers and Data cleaning
The crawlers are implemented in `getArticle.py`. The script uses selenium for web scraping and BeautifulSoup for HTML parsing. It includes functions to crawl titles from multiple news websites. The data cleaning functions are also included in this script. The base cleaning is implemented by removing short or empty titles and articles. Roughly 30k titles can be obtained after crawling and cleaning. The cleaned data is saved into a local SQLite database for later use.

## Database
The project uses SQLite as the database to store the crawled and cleaned data. The database schema is defined as follows:
```sql
CREATE TABLE IF NOT EXISTS articleTable (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT UNIQUE,
    title TEXT,
    article TEXT,
    labelA INTEGER,
    labelB INTEGER,
    labelC INTEGER,
    label INTEGER
)
```

## Data Labeling
The data labeling is done with three LLMs and a voting mechanism to improve the labeling quality. The three models are Gemini 2.5 Flash, GPT-OSS 120B and Claude Haiku 4.5. The final label is determined by majority voting among the three models. The labeling process is implemented in `Labeling/autoLabelingWorker.py`. The script reads the cleaned data from the database, sends each title to the three LLMs for labeling, and stores the individual labels and final label back into the database.

## Model Training
The model training is implemented in `trainModel.py`. The script uses the Hugging Face Transformers library to fine-tune a BERT model on the labeled dataset. The model adds layers for classification and uses cross-entropy loss for training. The trained model is saved for later use. The training achieves an accuracy of around 72% on the validation set.

## Inference(legacy: Title-based Inference)
The inference pipeline is implemented in `inference.py`. The script loads the trained model and provides a function to predict the stance of a given sentence.

## Web App(legacy: Title-based Web App)
The web app is implemented using Gradio in `app.py`. The app provides a simple interface for users to input a sentence and get the predicted stance from the model. The same file is also used on huggingface spaces for web deployment.

## Requirements
The project is managed through UV. `pyproject.toml` contains the project dependencies. To install the dependencies, run: `uv sync`. `.env` file should contain the Gemini API key for data labeling and huggingface API key for model inference.