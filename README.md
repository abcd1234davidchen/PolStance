# PolStance - a Political Stance Detector

The PolStance project take advantage of the power of transformer-based models to classify political stances in text. It works by a fine-tuned BERT model on a dataset of political statements labeled with their corresponding stances. The labeling is done by Gemini flash model. This repository contains both the training and inference code, as well as the dataset used for training.

## Status
 - [x] Implemented title crawling
   - [x] Crawl from chinatimes.com
   - [x] Crawl from thenewslens.com
   - [x] Crawl from ctinews.com/tags/國民黨
 - [x] Implemented data cleaning
 - [x] Implemented data labeling using Gemini flash
 - [x] Implemented model training using BERT
 - [x] Setup simplified inference pipeline
 - [ ] Setup web app for easy access
 - [ ] Always something to improve

## Crawlers
The crawlers are implemented in `getTitle.py`. The script uses selenium for web scraping and BeautifulSoup for HTML parsing. It includes functions to crawl titles from chinatimes.com and thenewslens.com.

## Requirements
`pyproject.toml` contains the project dependencies. Make sure to install them.