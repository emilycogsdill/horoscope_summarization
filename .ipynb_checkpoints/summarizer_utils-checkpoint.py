from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel

def load_xsum():

    print("loading xsum model...")
    xsum_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-xsum")
    print("loading xsum tokenizer...")
    xsum_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-xsum")
    print("creating xsum summarizer...")
    xsum_summarizer = pipeline("summarization", model=xsum_model, tokenizer=xsum_tokenizer)
    
    return xsum_summarizer

def load_cnn():
    
    print("loading cnn model...")
    cnn_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    print("loading cnn tokenizer...")
    cnn_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    print("creating cnn summarizer...")
    cnn_summarizer = pipeline("summarization", model=cnn_model, tokenizer=cnn_tokenizer)
    
    return cnn_summarizer

def summarize(summarizer,text,min_length,max_length):
    
    result = summarizer(text, min_length=min_length, max_length=max_length)
    summary = result[0]['summary_text']
    return summary
