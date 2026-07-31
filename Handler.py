from YKMtikitoko import Tikitoko

tokenizer = Tikitoko()

# ----------------------------
# Training Phase
# ----------------------------

training_text = ("<BOS> Patrick Jane is one of the most fascinating fictional detectives in modern television. Unlike traditional investigators, he rarely relies on forensic science alone. Instead, he studies people, emotions, habits, and tiny behavioral details. His extraordinary observation skills allow him to uncover truths that others completely overlook. Before joining the California Bureau of Investigation, " +
                 "Patrick worked as a professional psychic and openly admitted that his psychic abilities were nothing more than carefully crafted deception. That past became both his greatest strength and his deepest regret after the notorious serial killer Red John murdered his wife and daughter. From that moment onward, every investigation carried the weight of personal revenge. Although he often appears relaxed and playful, " +
                 "he constantly analyzes everyone around him. He notices body posture, breathing patterns, eye movements, subtle pauses, and changes in tone. These observations allow him to construct remarkably accurate psychological profiles. His methods frequently frustrate his fellow investigators because they seem unconventional, yet despite the chaos he creates, his deductions are almost always correct. <CLS> Every crime scene becomes a puzzle rather than merely a collection of evidence. Patrick enjoys misleading suspects into revealing information " +
                 "voluntarily instead of forcing confessions. He often sets elaborate psychological traps instead of relying solely on physical evidence. His understanding of human nature proves more valuable than expensive technology. Many suspects unknowingly confess simply because Patrick predicts exactly how they will react. His confidence occasionally borders on arrogance, but it is supported by exceptional reasoning and years of experience reading people. Despite his intelligence, he carries enormous emotional pain throughout the series, " +
                 "and the memory of his family influences nearly every major decision he makes. His relationship with Teresa Lisbon gradually evolves from professional partnership into deep trust and genuine friendship. The rest of the investigative team slowly learns to appreciate his unusual methods. " +
                 "Cho contributes calm discipline and quiet confidence, Rigsby provides determination and practical field experience, while Van Pelt offers empathy and technical expertise. <SEP> Red John remains the central mystery connecting many otherwise independent investigations. Every new clue raises additional questions instead of providing simple answers. Patrick patiently assembles countless small observations into a larger picture and understands that criminals frequently reveal themselves through overconfidence. " +
                 "Patience becomes one of his greatest investigative tools. Occasionally witnesses provide incomplete or contradictory information, but rather than forcing immediate conclusions, Patrick accepts temporary uncertainty. <UNK> Missing information rarely discourages him because he knows patterns eventually emerge. Some investigations require disguises, elaborate performances, or carefully planned social experiments, while others depend entirely on a single overlooked conversation. " +
                 "His success demonstrates that intelligence involves observation just as much as knowledge. <MASK> Even when evidence appears overwhelming, Patrick continues questioning every assumption because he believes the obvious explanation is often the least interesting one. His remarkable memory allows him to connect details separated by months or even years, and every solved case reinforces his belief that people inevitably reveal their true nature. <PAD> Although the search for Red John defines much of his journey, " +
                 "Patrick ultimately discovers that healing requires more than revenge, proving that understanding people can sometimes be more powerful than seeking justice alone. <EOS>")


#print(training_text)

tokenizer.initialize_tokenizer(required_vocab_size=1000) # initialized required vocab to 1000, will create only if compression further possible

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



print("starting decoding..")
print("here is the decoded string " + "\n" + tokenizer.decoding_to_str(token_ids=compressed_token_stream))

print("Learned New Vocabulary : ")
for token_id in generated_vocab_table:
    print(f"Token ID:{token_id} from pair of :{generated_vocab_table[token_id]}")

if training_text == tokenizer.decoding_to_str(token_ids=compressed_token_stream):
    print("\n~|DECODING WAS COMPLETELY SUCCESSFULL BOTH TEXTS MATCH|~")

print("\nCompression Ratio:")
print(f"{len(compressed_token_stream)/len(original_token_stream):.2f}")

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

if new_text == tokenizer.decoding_to_str(encoded_inference):
    print("\n~|DECODING WAS COMPLETELY SUCCESSFULL BOTH TEXTS MATCH|~")

print("\nCompression Ratio:")
print(f"{len(encoded_inference)/len(og_stream):.2f}")

