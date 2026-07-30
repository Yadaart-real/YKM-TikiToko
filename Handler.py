from YKMtikitoko import Tikitoko

tokenizer = Tikitoko()

# ----------------------------
# Training Phase
# ----------------------------

training_text = """
Byte Pair Encoding (BPE) is a subword tokenization algorithm used in
many modern Natural Language Processing systems.
"""

tokenizer.initialize_tokenizer(required_vocab_size=100)

original_token_stream, compressed_token_stream = tokenizer.training(text_by_user=training_text) # calling the training function on the text
generated_vocab_table = tokenizer.entire_vocabulary

print("=" * 60)
print("TRAINING")
print("=" * 60)

print("Original UTF-8 Token Stream:")
print(original_token_stream)
print("Length: " + str(len(original_token_stream)))

print("Compressed Token Stream:")
print(compressed_token_stream)
print("Length: " + str(len(compressed_token_stream)))

print("\n Compression Ratio:")
print(f"{len(compressed_token_stream)/len(original_token_stream):.2f}")

print("starting decoding..")
print("here is the decoded string " + "\n" + tokenizer.decoding_to_str(token_ids=compressed_token_stream))

print("Learned New Vocabulary : ")
for token_id in generated_vocab_table:
    print(f"Token ID:{token_id} from pair of :{generated_vocab_table[token_id]}")

new_text = """
The tokenizer should now encode this completely new sentence using the
previously learned vocabulary without creating any new merge rules.
"""

encoded_inference = tokenizer.inference_handling(raw_text=new_text)

print("\n\n" + "=" * 60)
print("INFERENCE")
print("=" * 60)

print("\nInput Text:")
print(new_text)

print("Encoded Uncompressed Token Stream:")
og_stream = tokenizer.utf_encoding(precoded_text=new_text)
print(og_stream)
print(len(og_stream))

print("\nEncoded and compressed Token Stream:")
print(encoded_inference)
print(len(encoded_inference))

print("\nDecoded Back:")
print(tokenizer.decoding_to_str(encoded_inference))

print("\n Compression Ratio:")
print(f"{len(encoded_inference)/len(og_stream):.2f}")
