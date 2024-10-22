# Amazon Product Review Sentiment Analysis

## Overview
The **Amazon Product Review Sentiment Analysis** project leverages the **RoBERTa** model for classifying the sentiment of Amazon product reviews. It scrapes reviews from Amazon using **Selenium**, processes them for sentiment analysis, and visualizes the sentiment distribution using bar and pie charts. Additionally, it integrates with the **Gemini API** to generate a summary of the pros and cons based on customer reviews.

### Features
- Scrapes Amazon product reviews automatically using Selenium and Helium 10 extension.
- Classifies reviews into **Positive**, **Neutral**, or **Negative** categories using the **RoBERTa** model.
- Visualizes the sentiment distribution with both **Bar Charts** and **Pie Charts**.
- Summarizes customer feedback (pros and cons) using the **Gemini API**.

## Requirements
Ensure the following Python packages are installed:

```plaintext
torch
transformers
pandas
matplotlib
google-generativeai
selenium
```

## How to Use
### 1. Clone the Repository
You can clone this repository using Git:
```bash
git clone https://github.com/YourUsername/YourRepositoryName.git
```
### 2. Navigate to the Project Directory
Use `cd` to navigate into the cloned project directory:
```bash
cd amazon-product-review-sentiment-analysis
```

### 3. Install Dependencies
Make sure you have Python installed. Then, use pip to install the dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Run the Project
To run the scraping and sentiment analysis program, use the following command:
```bash
python sentiment_analysis.py
```

### Customize the Script 
Ensure you have ChromeDriver installed and update the script with the correct path:
```bash
driver_path = r"Path\to\your\chromedriver.exe"
```
## Note 
Make sure all your chrome browsers are off when u start the program 

Enable Helium 10 extension in your chrome browser

Ensure that the Chrome profile used for scraping reviews is correctly set (ie. use the chrome profile with Helium 10 extension enabled). In the script, you will find the following lines that configure the Chrome profile:
```bash
chrome_profile_path = r"C:\Users\YourUsername\AppData\Local\Google\Chrome\User Data"
profile_name = "Profile 10"  # Replace with the actual profile you use for browsing Amazon
```
### Locate Chrome User Data Directory
Open Chrome and type `chrome://version/` in the address bar.

Press Enter. This will open a page that shows information about your Chrome installation.

Look for the Profile Path section. It will show the full path to the Chrome profile currently in use, such as:
```bash
C:\Users\YourUsername\AppData\Local\Google\Chrome\User Data\Default
```






