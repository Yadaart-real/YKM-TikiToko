"""
YKM TikiToko
============

A from-scratch implementation of the Byte Pair Encoding (BPE)
algorithm used by modern Natural Language Processing systems.

----------------------------------------------------------------
Why this project?
----------------------------------------------------------------

Modern language models such as GPT do not process raw text directly.
Instead, text is converted into tokens using tokenization algorithms,
one of the most popular being Byte Pair Encoding (BPE).

Rather than using an existing tokenizer library, this project rebuilds
the complete Byte Pair Encoding (BPE) pipeline manually, including
training, vocabulary serialization, inference-time encoding, and
recursive decoding.

The goal is to understand how production tokenizers learn reusable
vocabularies, persist them, and later tokenize previously unseen text
without retraining.

----------------------------------------------------------------
Implementation Philosophy
----------------------------------------------------------------

Before writing the implementation, the algorithm was decomposed into
smaller independent problems:

1. Encode text into UTF-8 bytes.
2. Generate adjacent byte pairs.
3. Count pair frequencies.
4. Select the most frequent pair.
5. Merge every occurrence into a new token.
6. Repeat until the desired vocabulary size is reached.
7. Recognize the pre-set special tokens during both encoding and decoding.
8. Recursively decode merged tokens back into the original text.

This modular approach made every stage independently testable before
combining them into a complete tokenizer capable of training,
serialization, inference, and decoding.

The implementation intentionally prioritizes readability and algorithmic
clarity over production-level optimisation.

Author:
Yatharth Keshavamurthy
(YKM)
"""
from shlex import join
from unittest import main

from pygments.lexers import special

"""
Initial design notes
--------------------

Before implementation, the algorithm was divided into
small independent components.

1. UTF-8 encoding

2. Pair frequency analysis

3. Merge operation

4. Recursive decoding

5. Training loop

Exact reference (very messy) : # functions I can make : 1) an utf encoding function that returns an array of all tokens (encoded letters), takes : entire text doc
                                                        2) a get_hook function that returns the pair having the most count, takes : the array of tokens, generates the pairs and counts
                                                        3) a merging function that replaces the get_hook's return with a newly merged token. and replaces each and every instance of that particularly max count pair
                                                        4) a decoding function that takes in the byte or the integer ID and gets its byte and decodes it back into the string
                                                        5) use the functions in a while loop based on the required vocab size (modified BPE)
Breaking the algorithm into modular pieces made
debugging and verification significantly easier.
"""
import json
import re
class Tikitoko:
    """
    A minimal Byte Pair Encoding (BPE) tokenizer implemented from scratch.

    The tokenizer operates primarily on UTF-8 byte tokens while supporting
    reserved special tokens commonly used by modern language models.

    During training, frequently occurring adjacent byte pairs are merged into
    new token IDs to construct a reusable vocabulary. During inference, the
    stored merge rules are replayed without modification to encode previously
    unseen text.

    The tokenizer additionally supports:

    • Vocabulary serialization to JSON
    • Vocabulary loading for inference
    • Recursive token decoding
    • Reserved special token handling
    • UTF-8 byte level tokenization

    The learned vocabulary maps every generated token ID to the pair of
    tokens used to construct it, allowing merged tokens to be recursively
    expanded back into their original byte representation.
    """

    def initialize_tokenizer(self,  required_vocab_size : int):
        """
        Initialise the tokenizer.

        Parameters
        ----------
        required_vocab_size : int
            Maximum number of merge operations the tokenizer
            is allowed to perform during training.

        Notes
        -----
        Every successful merge creates one new token ID,
        starting immediately after the standard UTF-8 byte range
        (0–255).

        Notes
        -----
        Initialization prepares the tokenizer for both training and inference by
        creating:

        • Merge vocabulary
        • Special token library
        • Reverse lookup table for decoding
        • Precompiled regular expression used to recognize special tokens

        Special tokens are assigned integer IDs beginning after the learned BPE
        vocabulary to ensure they never collide with UTF-8 bytes or generated
        merge tokens.
        """
        self.vocab_size = required_vocab_size
        # Keeps a complete history of every learned token.
        # Example:
        # 256 -> (97,98)
        # 257 -> (256,99)
        self.entire_vocabulary = {}
        self.special_token_library = {}
        self.special_set = set(self.special_token_library.values())
        self.special_tokens = ["<BOS>", "<EOS>", "<PAD>", "<UNK>", "<MASK>", "<CLS>", "<SEP>"] # intializing a list of all special tokens this tokenizer should be able to recognize

        for num,special_token in enumerate(self.special_tokens): # populating the special token library with the key as the special token itself
            self.special_token_library[special_token] = 1000+self.vocab_size+num  # we dont wanna smudge tokens within our created vocabulary or our utf 8 exisitng vocabulary aswell
        #inverting the dictionary so i can access the original string using token id as key during decoding
        self.inverted_special_token_library = {value: key for key, value in self.special_token_library.items()}
        specialized_str = f"({"|".join(map(re.escape, self.special_tokens))})" # making the proper string that re.split() needs "(BLAH | BLAH2)" type
        self.one_time_special_state = re.compile(specialized_str) # pre intializing a one time exclusion state for re.split() to recognize the special tokens

    def utf_encoding(self, precoded_text:str):
        """
        Convert raw text into the tokenizer's internal token representation.

        Normal text is encoded as UTF-8 byte tokens while reserved special
        tokens (such as <BOS>, <EOS>, and <MASK>) are preserved as individual
        token IDs rather than being decomposed into bytes.

        This allows special tokens to participate in inference and decoding
        without losing their semantic meaning.

        ...
        Returns
        -------
        list[int]

            A token stream consisting of both

            • UTF-8 byte tokens

            • Reserved special token IDs

        while preserving the original ordering of the input text.
        """
        # encoding happens character by character # so i have to exclude the spcial characters or even the entire special token word, maybe by splitting the string ?
        complete_stream = [] # the final complete stream of tokens
        splitted_list = self.one_time_special_state.split(precoded_text) # splitting the text into a list containing the delimitters (the special tokens) in correct positions (betwwen or wherever) relative to the normal text
        for split in splitted_list: # iterating over each split/part
            if split in self.special_tokens: #if we find the delimitter/ the special token itself
                # we append it's token id (taken by passing the key as the special string itself to get token id) to our complete stream.
                # since the positions are properly taken care off each append also follows proper order thus preserving the original order of the text
                complete_stream.append(self.special_token_library[split])
            else:
                complete_stream.extend(list(split.encode('utf-8'))) # if the part is a normal text we get the corresponding integer id's of characters by encoding using utf-8 (base) and
                # converts the encoded bytes into a list of integer id's with each id having different byte value
                # and append it to our complete stream again in proper order as preserved
                # beautifully programmed right ?

        return complete_stream # returns the text's fully encoded token stream or list

    """
    A ghost function written to have an entire vocabulary even of the original utf-8 encodings,
    was planning something, but decided against the idea.. 
    still left it there, as a GHOST....
    """
    #def initialize_vocab_og(self, encoded_array:list):
        #for idx in encoded_array:
            #self.entire_vocabulary[idx] = idx
                                                            #  !
    def get_hook(self, token_array:list):                   # aaabccc
        """
        Find the most frequently occurring adjacent token pair.

        During every training iteration, all neighboring token
        pairs are generated and counted.

        A check is also made before increasing each pair's count
        to ensure the hooked pair that is sent for merging
        does not contain a special token ( since we don't want to merge our special tokens at all)
        if such a pair is formed its count is never incremented, and it remains 1 (edge case handled during training)
        If a genuine pair occurs (no special tokens in it) its count is incremented.

        The pair with the highest frequency becomes the
        "hooked pair"—the pair selected for merging during
        the current iteration.

        Returns
        -------
        tuple

        (
            (left_token, right_token),
            frequency
        )
        """
        # Maps each adjacent pair to the number of times
        # it appears in the current token stream.
        pair_dictionary = {} # dictionary with tuple pair as key and value as its count
        for i in range(0, len(token_array)-1):  # iterates over the encoded token array to generate all pairs
            current_byte = token_array[i] # current byte value
            ahead_byte = token_array[i+1] # one step ahead byte value
            pair = (current_byte,ahead_byte) # makes the pair
            if pair in pair_dictionary:# checks if pair already exists in hash
                if pair[0] in self.special_set or pair[1] in self.special_set: # checking if pair has either child as special token
                    continue # if yes it doesnt increase its count and lets it stay 1, so it never gets chosen as hooked pair, and edge case of possibly being chosen
                            # if count all pairs is 1 is taken care of by stopping training upon reaching that circumstance
                else: # if the pair is genuine and does not contain any special tokens
                    # its count can be incremented so it has infused possibility of being chosen as hooked pair
                    pair_dictionary[pair] += 1 # if yes it increments its count
            else:
                pair_dictionary[pair] = 1 # if no it adds that pair to dictionary and sets original count to 1
        hooked_pair = max(pair_dictionary.items(), key=lambda item: item[1]) # finds the most occurring pair, by getting the max count in the dictionary and returning its key i.e the pair
        # "Hooked pair" is simply the pair selected
        # for merging during this training iteration.
        return hooked_pair

    def merging(self, token_array:list, hooked_pair:tuple, new_token_id:int, is_inference:bool):
        """
        Merge every occurrence of the selected pair into a new token.

        This function performs the core compression step of the
        Byte Pair Encoding algorithm.

        Every occurrence of the most frequent adjacent pair is replaced
        with a newly generated token ID.

        The relationship between the new token and the pair used to
        construct it is permanently stored inside the vocabulary.

        Example
        -------

        Before

            [65, 66, 65, 66]

        Pair selected

            (65, 66)

        After

            [256, 256]

        Vocabulary

            256 -> (65, 66)

        Parameters
        ----------
        token_array : list
            Current token sequence.

        hooked_pair : tuple
            Most frequently occurring pair.

        new_token_id : int
            Integer ID assigned to the merged token.

        Returns
        -------
        list

            Updated token stream after every merge.
        """
        # Stores the updated token stream after merges.
        updated_array = [] # intialzing empty list that will be populated with the new token(for merged tokens) and current tokens (for non merging tokens)
        # Manual indexing is used because merges consume two tokens
        # at once, making a standard for-loop unsuitable.
        i = 0 # setting the iteration counter to 0
        # Record how the new token was formed.
        # This information is later used by in the recursive decoder.
        if not is_inference:
            self.entire_vocabulary[new_token_id] = hooked_pair # logs into the global vocab that the new token is formed from what pair

        while i < len(token_array)-1: # iterates over token array but stops one early to prevent indexing error
            current_token = token_array[i] # the current token byte
            ahead_token = token_array[i+1] # the ahead token byte
            if current_token == hooked_pair[0] and ahead_token == hooked_pair[1]: # checks if it found the pair in the array
                updated_array.append(new_token_id) # appends to the updated array with the new token id if it found the pair
                i += 2 # since pair found it skips the middle element
            else:
                updated_array.append(current_token) # if pair not found add the current single token to updated array
                i += 1 # go ahead to next element
        if i == len(token_array)-1: # if the last element is never part of a pair then we append it here to avoid the indexing error earlier stated
            updated_array.append(token_array[i]) # add the last element to final array
        # If no merges occurred, the returned list is
        # just basically identical to the input.
        return updated_array


    # decoding should have recursion as there is a possibility that a token formed after merging may consist of the tokens that are again also formed after merging
    # and we can maintain a list of the bytes, and basically add both the left and the right children as part of that list, thus if it branches like a merged token inside a merged token the list will expand and eventually get flattened at the end into one single non nested list i.e the virgin bytes

    def decoding_to_str(self, token_ids:list):
        """
        Decode a sequence of BPE tokens back into the original text.

        Unlike the encoder, decoding cannot rely on a simple lookup.

        Newly generated tokens may themselves contain previously
        merged tokens, producing a hierarchy of token relationships.

        Example

            259
            ├──256
            │   ├──104
            │   └──101
            └──258
                ├──108
                └──111

        Recovering the original text therefore requires recursively
        expanding every merged token until only the original UTF-8
        byte values remain.

        Those bytes are then reconstructed into the original string.
        Notes
        -----
        Decoding is performed in two stages :
        1. Every learned BPE token is recursively expanded until only
            UTF-8 byte values remain.
        2. Reserved special token IDs are translated directly back into
        their original string representations.

        This allows mixed token streams containing both learned BPE tokens
        and special tokens to be reconstructed exactly.
        """
        def decode_special_tokens(special_token_id):
            decoded_special_str = self.inverted_special_token_library[special_token_id]
            return decoded_special_str

        def get_bytes(token_idx): # a helper function to allow for recursion
            """
               Recursively expand one token into its original bytes.
            """
            decoded_str = ""
            if token_idx <= 255: # checks if the token isn't a new token and is one of the already single token ids(BASE CASE)
                return [token_idx] # just returns that id as list since it isnt a new token at all so no merges
            if token_idx in self.entire_vocabulary: # if it wasnt part of the 0 to 255 tokens, we check if the token even exists in our vocab
                left_child, right_child = self.entire_vocabulary[token_idx] # if it does we get the left child and the right child of this new token since it returns a tuple
                #so the below line will branch each pair (tuple) into its respective children
                #and do so again if the children also form from a merge
                return get_bytes(left_child) + get_bytes(right_child) # returns the recursively byte concatenation of both lists(the left child if contains another tuples inside it again takes and branches it aswell)
                # basically returns a list again but of the entire constituency of the bytes ( the addition basically appends to a list nothing else)
            # Unknown token.
            # Return an empty list instead of raising an exception.
            return [] # returns empty list if it got some weird ass value or an unknown token

        # both of the below variables are used to kinda split the token list basically or seperate it in chunks of original utf-8 decodable tokens and special tokens in order
        completely_decoded_parts = []  # Fully reconstructed string sequence
        running_part = [] # this list contains only the original UTF-8 bytes.

        for num,token_id in enumerate(token_ids): # iterates throught the token sequence that needs to be decoded
            if token_id in self.inverted_special_token_library: # checks if the current token_id is a special token
                if running_part: # checks if the running part/split has accumulated original utf-8 token id's or not
                    completely_decoded_parts.append(bytes(running_part).decode("utf-8", errors="replace")) # appends the utf-8 decoded string of that original utf-8 split to list of complete decoded strings
                    running_part = [] # resets running part/split to be able to accumulate original utf-8 token id's again
                completely_decoded_parts.append(decode_special_tokens(token_id)) # once the running part is decoded we append the decoded string of the current token id (special one) to our complete decoded string list
                                                                                 # thus the append always happens in order preserving the position of the special token in the sequence aswell
            else: # if the current token_id is not a special token
                # we add the complete flattened tree(if there) of the token to our running part and await the next special token
                running_part.extend(get_bytes(token_id)) # flattens it, using extend to get a non nested list, and not append or we would get a nested list
                # the below check is done only if we are sure that no special token is present at the last token in token list if it is, it will already have been taken care off in above logic
                if num == len(token_ids)-1: # checks if we reach the last tokenid in the given token_list,
                    # if not a special token and we reach the end (final token in caller's token list) then we append running part to complete decoded parts list
                    completely_decoded_parts.append(bytes(running_part).decode("utf-8", errors="replace"))
        # returns the concatenated final string by joining all the seperate decoded strings of original utf-8 decodables and special tokens
        # since all appends are done in token arised order it is all positionally correct
        return "".join(completely_decoded_parts)# returns the concatenated final string by joining all the seperate decoded strings of original utf-8 decodables and special tokens, since all appends are done in token arised order it is all positionally correct

    def inference_handling(self, raw_text:str):
        """
        Encode previously unseen text using a saved BPE vocabulary.

        Unlike `training()`, this method performs no learning and creates no
        new merge rules. Instead, it loads a previously trained vocabulary from
        disk and replays the learned merge operations in their original order.

        The supplied text is converted into the tokenizer's internal token
        representation, preserving any reserved special tokens while encoding
        ordinary text as UTF-8 byte tokens.

        The learned merge rules are then replayed exactly as they were created
        during training.

        Parameters
        ----------
        raw_text : str
            Previously unseen text to encode.

        Returns
        -------
        list
            The compressed token sequence generated using the stored
            pre-trained vocabulary.

        Notes
        -----
        This method assumes that a vocabulary has already been created
        using `training()` and saved as a JSON file.
        """
        # loading stored pre-trained vocabulary from the json file
        try:
            with open("pretrained_vocab_token_merges.json", "r") as hot_file: # loads the stored json file
                stored_vocab_data = json.load(hot_file) # fetches the pre-trained vocabulary of merged tokens and new token id's
                print(f"Yay! vocabulary has been fetched../")
        except FileNotFoundError: # if file is not found
            return f"The file {hot_file} was not found anywhere, womp womp../"
        except json.JSONDecodeError: # if the file is not in correct JSON format
            return f"The file {hot_file} is corrupted dude, clean it up../"

        if not stored_vocab_data: # if all above checks are good, but the loaded dict is empty return empty error
            return f"The file {hot_file} is entirely empty, some error occured while saving../"
        else: # if not continue with inference
            token_list = self.utf_encoding(precoded_text=raw_text) # generates the original encoded token stream from raw user provided text
            mutable_token_stream = token_list.copy() # creates a copy of the original stream to make a changable stream that will be worked on
            sorted_data_of_merges_and_ids = sorted(stored_vocab_data) # sorts the Integer ID's in order, since the original merges were made in that same order
            for tid in sorted_data_of_merges_and_ids: # iterates over those sorted ID's
                pair = stored_vocab_data[tid] # gets the pair from the vocabulary map for that Integer ID/token
                token_id = int(tid) # stores the token ID/merged token number from vocab map
                mutable_token_stream = self.merging(mutable_token_stream, pair, token_id, is_inference=True) # merges all instances of the current pair into the currently provided new token id
            return mutable_token_stream


    def training(self, text_by_user:str): # the training function that actually initiates the BPE algo
        """
        Train the tokenizer on the supplied text.

        Reserved special tokens are never merged into the learned vocabulary.
        Instead, they remain fixed identifiers throughout training and are
        treated as immutable boundaries between ordinary UTF-8 token sequences.

        The training loop repeatedly

            1. Finds the most frequent pair.
            2. Merges every occurrence.
            3. Creates one new vocabulary entry.

        Training terminates when either

        • the requested vocabulary size is reached

        or

        • no pair occurs more than once.

        Upon successful completion, the learned vocabulary is automatically
        serialized to a JSON file, allowing future encoding sessions to reuse
        the same merge rules without retraining.

        Returns
        -------
        tuple

        (
            original_token_stream,
            compressed_token_stream
        )
        """

        original_token_list = self.utf_encoding(precoded_text=text_by_user) # Preserve the original token stream so the caller
                                                                            # can compare it against the compressed version.
        mutable_token_list = original_token_list.copy() # creating a copy of the original as mutable, so every merge will mutate this sequence and not original
        counter = 1 # setting counter to start, counts how many vocabulary additions/entries where made
        while counter <= self.vocab_size: # ensuring training produces required vocabulary size only
            hooked_pair,countofpair = self.get_hook(token_array=mutable_token_list) #getting the most occurring adjacent pair from the token array
            if countofpair == 1: # checking to see if we arent forceully creating vocab, making sure if pair has count more than 1 thus occurring more than once
                break # if the max pair only occurs once it means no pair is occurring more than once thus no tokens need to be merged so no new token or vocab needs to be generated so we stop training
            else: # if the max pair has count more than 1 it means this pair needs to be merged and new token needs to generate, Continue training
                #print(f"Merge {counter}")
                mutable_token_list = self.merging(token_array=mutable_token_list, hooked_pair=hooked_pair, new_token_id= 255+counter, is_inference=False) # merging the new token and replacing all instances of the merging pair from the token list and setting it to mutable list
                counter+=1 # incrementing the vocab_counter                                                           # Generated token IDs begin immediately after the UTF-8 byte range (0–255).

        #saving the trained vocabulary consisting of merged tokens as new tokens with integer id's
        with open("pretrained_vocab_token_merges.json", "w") as sexy_file:
            json.dump(self.entire_vocabulary, sexy_file, indent=4) # indent for readability

        return original_token_list,mutable_token_list # returns both the original and updated list of tokens


"""
------------------------------------------------------------------
End of implementation.

Author :
Yatharth Keshavamurthy
(YKM)

This tokenizer was intentionally implemented from first principles
to understand how Byte Pair Encoding constructs vocabularies used
by modern Large Language Models.

Current features
----------------
✓ BPE vocabulary training
✓ Vocabulary serialization (JSON)
✓ Vocabulary loading
✓ Inference-time encoding
✓ Recursive decoding
✓ Special token support (recognize pre-set special tokens)

Future versions will focus on production-oriented improvements such as
performance optimisation, larger training corpora,
and additional tokenizer utilities.
------------------------------------------------------------------
"""
