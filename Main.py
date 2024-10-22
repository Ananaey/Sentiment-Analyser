import os
import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import matplotlib.pyplot as plt
import google.generativeai as genai
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time

# Configure the API key for Gemini
API_KEY = 'AIzaSyAxsNb8T2UhRWycVyuVKPLoqzTcDMGgbSA'  # Replace with your actual Gemini API key
genai.configure(api_key=API_KEY)

# Load the RoBERTa model and tokenizer for sentiment analysis
tokenizer = RobertaTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
model = RobertaForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')

# Chrome WebDriver setup (Selenium)
driver_path = r"C:\Users\Asus\Downloads\chromedriver-win64\chromedriver-win64\chromedriver.exe"
chrome_profile_path = r"C:\Users\Asus\AppData\Local\Google\Chrome\User Data"
profile_name = "Profile 10"  # Specific Chrome profile with the extensions installed


# Function to extract reviews from Amazon using Selenium and Helium
def scrape_amazon_reviews():
    # Prompt the user for the Amazon product review page
    product_link = input("Please enter the Amazon product review page URL: ")

    # Set Chrome options for the Selenium WebDriver, including using the specific profile
    chrome_options = Options()
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument(f"user-data-dir={chrome_profile_path}")  # Use the Chrome user data folder
    chrome_options.add_argument(f"--profile-directory={profile_name}")  # Use the specific profile (Profile 10)
    chrome_options.add_argument("--remote-debugging-port=9222")  # Set remote debugging port to avoid DevTools issue

    service = Service(driver_path)
    driver = webdriver.Chrome(service=service, options=chrome_options)

    # Open the Amazon product review page
    driver.get(product_link)
    time.sleep(5)  # Wait for the page to load fully

    print("Amazon page opened with your Chrome profile. Helium 10 should now automatically download the reviews CSV.")

    # Wait for Helium 10 to download the CSV automatically
    time.sleep(60)  # Adjust this based on how long the download takes

    # Check the Downloads folder for the most recent CSV file
    downloads_folder = os.path.join(os.path.expanduser('~'), 'Downloads')
    latest_file = get_latest_downloaded_file(downloads_folder)
    driver.quit()

    return latest_file

# Function to find the latest downloaded file in the Downloads folder
def get_latest_downloaded_file(downloads_folder):
    files = os.listdir(downloads_folder)
    paths = [os.path.join(downloads_folder, basename) for basename in files if basename.endswith('.csv')]
    latest_file = max(paths, key=os.path.getctime)  # Get the most recently downloaded file
    return latest_file

# Function to calculate and display sentiment percentages
def calculate_sentiment_percentage(sentiment_counts, total_reviews):
    for sentiment, count in sentiment_counts.items():
        percentage = (count / total_reviews) * 100
        print(f"{sentiment}: {percentage:.2f}%")

# Function to analyze sentiment of reviews and visualize the results
def analyze_sentiment_and_visualize(csv_file):
    try:
        # Load the CSV into a pandas DataFrame
        df = pd.read_csv(csv_file)

        # Select the 'Body' column for sentiment analysis
        reviews = df['Body'].dropna().tolist()  # Drop missing values (if any)

        # Initialize counters for sentiment categories
        sentiment_counts = {'Positive': 0, 'Neutral': 0, 'Negative': 0}
        review_texts = []  # Collecting review texts for Gemini summary

        print(f"CSV file found: {csv_file}")

        # Analyze sentiment for each review
        print("\nPerforming sentiment analysis on reviews...")
        for i, review in enumerate(reviews):  # Process all reviews
            try:
                # Tokenize and analyze sentiment
                inputs = tokenizer(review, return_tensors="pt", truncation=True, padding=True, max_length=512)
                outputs = model(**inputs)
                sentiment = torch.argmax(outputs.logits, dim=-1).item()

                # Convert numeric sentiment to label (0: Negative, 1: Neutral, 2: Positive)
                sentiment_label = ['Negative', 'Neutral', 'Positive'][sentiment]
                sentiment_counts[sentiment_label] += 1
                review_texts.append(review)

            except Exception as e:
                # Handle error in sentiment analysis
                print(f"Review {i + 1} is too long or caused an error. Skipping...")

        # Calculate sentiment percentages
        total_reviews = len(reviews)
        calculate_sentiment_percentage(sentiment_counts, total_reviews)

        # Visualization - Bar chart
        plt.figure(figsize=(8, 6))
        labels = list(sentiment_counts.keys())
        counts = list(sentiment_counts.values())
        plt.bar(labels, counts, color=['green', 'gray', 'red'])
        plt.title("Sentiment Distribution for Product Reviews", fontsize=14)
        plt.xlabel("Sentiment", fontsize=12)
        plt.ylabel("Number of Reviews", fontsize=12)
        plt.show()

        # Additional Visualization - Pie Chart
        plt.figure(figsize=(6, 6))
        plt.pie(counts, labels=labels, autopct='%1.1f%%', colors=['green', 'gray', 'red'])
        plt.title("Sentiment Pie Chart for Product Reviews")
        plt.show()

        # Gemini API to generate pros and cons summary
        pros_cons_summary = generate_gemini_summary(review_texts)
        print("\nGemini Pros and Cons Summary:\n", pros_cons_summary)

    except Exception as e:
        print(f"Error processing sentiment analysis: {e}")

# Function to generate pros and cons using Gemini API
def generate_gemini_summary(reviews):
    try:
        # Provide a detailed, structured prompt for Gemini API to follow
        prompt = """
        Summarize the customer reviews for a product into three sections: Pros, Cons, and Overall Customer Sentiment.
        Please make the summary concise and to the point. Use bullet points for the pros and cons.

        Reviews:
        """ + " ".join(reviews)  # Using all reviews for generating summary

        # Create the conversation history for Gemini
        conversation_history = [{"role": "user", "content": prompt}]

        model = genai.GenerativeModel('gemini-pro')
        chat = model.start_chat()
        response = chat.send_message(prompt).text

        return response

    except Exception as e:
        print(f"Error generating summary with Gemini API: {e}")
        return "Summary generation failed."

# Main function to scrape reviews, analyze sentiment, and visualize
def main():
    # First, scrape the reviews from Amazon using Selenium and Helium
    csv_file = scrape_amazon_reviews()

    # Then, perform sentiment analysis and visualize results
    if csv_file:

        analyze_sentiment_and_visualize(csv_file)
    else:
        print("No CSV file found. Please try again.")


if __name__ == "__main__":
    main()
