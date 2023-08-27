from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain
import tiktoken
import pandas as pd


def call_transcripts_etl(raw_trascripts):
    import re
    f = open(raw_trascripts, 'r')
    content = f.read()
    content = content.replace('\n',' ')
    return re.findall(r'(Order.Number:.*?)Order.Number', content) + re.findall(r'(Transcript.[0-9][0-9]:.*?)Transcript', content) + re.findall(r'(Transcript.*?:.*?)Transcript', content)

def get_openai_api_key():
    import os
    load_dotenv()
    return os.getenv('openai_api_key')

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def get_input_price_estimate(string: str, gpt_version_number: int) -> int:
    import tiktoken
    """Returns the number of tokens in a text string."""
    if gpt_version_number == 3:
        encoding_name = 'r50k_base'
    else: encoding_name = 'cl100k_base'
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    cost = 0
    if gpt_version_number == 4:
        cost = 0.03
    elif gpt_version_number == 3.5:
        cost = 0.0015
    elif gpt_version_number == 3:
        cost = 	0.0016 

    return round(cost*(num_tokens_from_string(processed_transcripts[0], encoding_name)/1000),5)

def get_output_price_estimate(string: str, gpt_version_number: int) -> int:
    import tiktoken
    """Returns the number of tokens in a text string."""
    if gpt_version_number == 3:
        encoding_name = 'r50k_base'
    else: encoding_name = 'cl100k_base'
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    cost = 0
    if gpt_version_number == 4:
        cost = 0.06
    elif gpt_version_number == 3.5:
        cost = 0.002
    elif gpt_version_number == 3:
        cost = 	0.0016 
    
    

    return round(cost*(num_tokens_from_string(processed_transcripts[0], encoding_name)/1000),5)

def generate_llm_sentiment_analysis(open_ai_api_key, model, text):
    from langchain import PromptTemplate
    template = """Based on the input below, provide a sentiment score. This score should be formatted as
    a float between -1 to 1 where -1 is highly negative, 0 is neutral, and 1 is highly positive. The output should consist of only a number.

    Context: LLM's are superior models for determining the sentiment of text.

    Question: {query}

    Answer: """

    prompt_template = PromptTemplate(
        input_variables=["query"],
        template=template
    )

    openai = OpenAI(openai_api_key=openai_api_key, model_name=model)

    return openai(prompt_template.format(query=text))

def get_vader_sentiment(text):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)['compound']

def generate_follow_up_flag(open_ai_api_key, model, text):
    from langchain import PromptTemplate
    template = """Based on the input below, determine whether the customer support representative needs to follow up with the customer.
    Output a 1 if a follow up is needed and a 0 if a follow up is not needed. The only possible outputs are a 1 or 0. A follow up is required if the customer's sentiment is negative or
    if they threaten to cancel their service or if they are extremely frustrated.

    Context: LLM's are superior models for determining the need to follow up with customers.

    Question: {query}

    Answer: """

    prompt_template = PromptTemplate(
        input_variables=["query"],
        template=template
    )

    openai = OpenAI(openai_api_key=openai_api_key, model_name=model)

    return openai(prompt_template.format(query=text))

def generate_pain_points(open_ai_api_key, model, text):
    from langchain import PromptTemplate
    template = """Based on the input below, generate 1 pain point the customer is facing. Examples include "long wait time", "defective product" etc.
    These pain points should be at most 3 words long.

    Context: LLM's are superior models for determining customer needs.

    Question: {query}

    Answer: """

    prompt_template = PromptTemplate(
        input_variables=["query"],
        template=template
    )

    openai = OpenAI(openai_api_key=openai_api_key, model_name=model)

    return openai(prompt_template.format(query=text))

processed_transcripts = call_transcripts_etl('customer_service_transcripts.txt')
openai_api_key = get_openai_api_key()

df = pd.DataFrame({'CALL_TRANSCRIPTS' : processed_transcripts})


df['INPUT_PRICE_GPT3.5'] = df['CALL_TRANSCRIPTS'].apply(lambda x: get_input_price_estimate(x, 3.5))
df['SENTIMENT_3_5'] = df['CALL_TRANSCRIPTS'].apply(lambda x: generate_llm_sentiment_analysis(openai_api_key, model = 'gpt-3.5-turbo', text = x))
df['FOLLOW_UP_FLAG'] = df['CALL_TRANSCRIPTS'].apply(lambda x: generate_follow_up_flag(openai_api_key, model = 'gpt-3.5-turbo', text = x))
df['VADER_OUTPUT'] = df['CALL_TRANSCRIPTS'].map(get_vader_sentiment)
df['OUTPUT_PRICE_SENTIMENT'] = df['SENTIMENT_3_5'].apply(lambda x: get_output_price_estimate(str(x), 3.5))
df['OUTPUT_PRICE_FOLLOW_UP'] = df['FOLLOW_UP_FLAG'].apply(lambda x: get_output_price_estimate(str(x), 3.5))
df['PAIN_POINTS'] = df['CALL_TRANSCRIPTS'].apply(lambda x: generate_pain_points(openai_api_key, model = 'gpt-3.5-turbo', text = x))
df.to_csv('processed_data.csv')

