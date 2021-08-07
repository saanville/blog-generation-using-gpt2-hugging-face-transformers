from transformers import GPT2Tokenizer, GPT2LMHeadModel

# for using larger models, use "gpt2-large" instead of "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id)

sentence = "Any input text that you wanna start your blog with"

length = None   #enter the maximum length of the string that you want to generate

input_ids = tokenizer.encode(sentence, return_tensors="pt")

output = model.generate(input_ids, max_length=length, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)

output_text = tokenizer.decode(output[0])
output_text