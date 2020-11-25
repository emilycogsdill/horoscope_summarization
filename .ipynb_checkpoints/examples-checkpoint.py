from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel

# Monthly horoscope - https://www.vice.com/en/article/epddvj/monthly-horoscope-aquarius-november-2020
text = '''Welcome to Scorpio season, dear Aquarius! The sun is lighting up the sector of your chart that rules your career and life in public, and reward and recognition are sure to come your way! You’re famously an inventive genius, but it’s not just your brilliance that wins your success; it’s your tenacity and focus on achieving the results you’re looking for, and Scorpio season supports you in exactly that. Mercury retrograde in Libra clashes with your ruling planet Saturn, currently in Capricorn, on November 1, and will do so again on November 6, after Mercury ends its retrograde on November 3. The end of Mercury retrograde means we can finally move forward with conversations that have been brewing over the last few weeks, especially regarding travel, education, and your career. You’re figuring out that there’s some information missing, and need to make some changes to get everything together. This can be frustrating, since there are more delays than you’d like to deal with, but at least things are meeting a standard. You’re thinking back to September 23 as Mercury clashes with Saturn, struggling with similar delays in communication or a heavy mental atmosphere. Venus in Libra has found you having some philosophical breakthroughs, and as it opposes Mars retrograde on November 9, you’re ready to have some intense conversations. You might be admitting your feelings for someone or simply sharing your beliefs with a friend. The sun connects with mystical Neptune in Pisces on November 10, bringing a whimsical, imaginative energy, and a boost of abundance. When Netpune’s involved, resources accumulate over time. Mercury enters Scorpio on November 10, kicking up communication about your career—this is a great time to update your résumé or to make it public. Jupiter meets Pluto in Capricorn for the third and final time this year on November 12. Think back to April 4 and June 30, as similar breakthroughs are taking place. This also begins a new spiritual journey for you, Aquarius—profound messages may arrive in your dreams, making this a powerful time to start a dream journal. This is also a brilliant time for therapy and for shadow work, as you’re ready to explore unknown parts of yourself . It may feel like there’s a lot happening at once and like there’s no information about how things will end up, so stay present in your body, don’t rush to making decisions, and most importantly, reach out for help!'''

# Xsum

print("loading xsum model...")
xsum_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-xsum")
print("loading xsum tokenizer...")
xsum_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-xsum")
print("creating xsum summarizer...")
xsum_summarizer = pipeline("summarization", model=xsum_model, tokenizer=xsum_tokenizer)

print("summarizing with Xsum...")
xsum_result = xsum_summarizer(text, min_length=5, max_length=100)
print("Xsum summary: \n")
print(xsum_result[0]['summary_text'])
print("\n\n")

# CNN

print("loading cnn model...")
cnn_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
print("loading cnn tokenizer...")
cnn_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
print("creating cnn summarizer...")
cnn_summarizer = pipeline("summarization", model=cnn_model, tokenizer=cnn_tokenizer)

print("summarizing with CNN...")
cnn_result = summarizer(text, min_length=5, max_length=100)
print("Large CNN summary: \n")
print(cnn_result[0]['summary_text'])
print("\n\n")