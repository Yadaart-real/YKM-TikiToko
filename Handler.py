import YKMtikitoko

tiktoko = YKMtikitoko.Tikitoko() # instantiating my tokenizer
tiktoko.initialize_tokenizer(required_vocab_size=4) # intializing the tokenizer with the vocab size here
og, mg = tiktoko.training(text_by_user="The original BPE algorithm operates by iteratively replacing the most common contiguous sequences of characters in a target text with unused 'placeholder' bytes. The iteration ends when no sequences can be found, leaving the target text effectively compressed. Decompression can be performed by reversing this process, querying known placeholder terms against their corresponding denoted sequence, using a lookup table. In the original paper, this lookup table is encoded and stored alongside the compressed text.") # calling the training function on the text
generated_vocab_table = tiktoko.entire_vocabulary
print("this is the og token list before BPE encoding ")
print(og)
print("Length: " + str(len(og)))

print("this is the mg list after BPE encoding ")
print(mg)
print("Length: " + str(len(mg)))

print("starting decoding..")
print("here is the decoded string " + "\n" + tiktoko.decoding_to_str(token_ids=mg))


print("the generated vocabulary is as follows")
print(generated_vocab_table)