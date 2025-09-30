# PolStance - a Political Stance Detector

The PolStance project take advantage of the power of transformer-based models to classify political stances in text. It works by a fine-tuned BERT model on a dataset of political statements labeled with their corresponding stances. The labeling is done by Gemini 2.5 model. This repository contains both the training and inference code, as well as the dataset used for training.

## Status
 - [x] Implemented title crawling
   - [x] Crawl from chinatimes.com
   - [x] Crawl from thenewslens.com
 - [x] Implemented data cleaning
 - [ ] Implemented data labeling using Gemini 2.5
 - [ ] Implemented model training using Gemma 3
 - [ ] Setup inference pipeline
 - [ ] Documented code and usage instructions

## Planned
 - [ ] Find more sources

## Crawlers
The crawlers are implemented in `getTitle.py`. The script uses selenium for web scraping and BeautifulSoup for HTML parsing. It includes functions to crawl titles from chinatimes.com and thenewslens.com.

## Requirements
`pyproject.toml` contains the project dependencies. Make sure to install them.