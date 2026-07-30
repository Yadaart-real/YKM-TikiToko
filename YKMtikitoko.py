#import numpy as np


# a standard modified BPE tokenizer
# break a given text data into tokens, based on Byte Pair Encoding

# Take an example text data, get unicode encoding to convert into bytes, generate byte pairs, maintain a dictionary with
# key as the pair tuple itself and count as value, get the pair with max count out of all, and merge it into a new token
#replace all instances of the pair (now merged) by the new token

# functions I can make : 1) an utf encoding function that returns an array of all tokens (encoded letters), takes : entire text doc
                        #2) a get_hook function that returns the pair having the most count, takes : the array of tokens, generates the pairs and counts
                        #3) a merging function that replaces the get_hook's return with a newly merged token. and replaces each and every instance of that particularly max count pair
                        #4) a decoding function that takes in the byte or the integer ID and gets its byte and decodes it back into the string
                        #5) use the functions in a while loop based on the required vocab size (modified BPE)

class Tikitoko:
    def initialize_tokenizer(self,  required_vocab_size : int):
        #self.input_text = text_input
        self.vocab_size = required_vocab_size # initializing the required vocabulary size according to user (how many merges or new merged vocabulary the user would like)
        self.entire_vocabulary = {} # setting a global vocab to have a history of all merges and custom integer id's apart from utf integers ids (0-255)

    def utf_encoding(self, precoded_text:str): # encodes the given text into a list of all its integer id with returnable bytes (the integer id's correspond to the utf-8 bytes)
        encoded_array = list(precoded_text.encode('utf-8')) # converts the encoded bytes into a list of integer id's with each id having different byte value
        return encoded_array # returns that string's integer id encodings or also called tokens

    #def initialize_vocab_og(self, encoded_array:list):
        #for idx in encoded_array:
            #self.entire_vocabulary[idx] = idx
                                           #  !
    def get_hook(self, token_array:list): # aaabccc  # generates adjacent pairs from utf encoded token list and returns the most occuring pair along with its count for training continuation check
        pair_dictionary = {} # dictionary with tuple pair as key and value as its count
        for i in range(0, len(token_array)-1):  # iterates over the encoded token array to generate all pairs
            current_byte = token_array[i] # current byte value
            ahead_byte = token_array[i+1] # one step ahead byte value
            pair = (current_byte,ahead_byte) # makes the pair
            if pair in pair_dictionary: # checks if pair already exists in hash
                pair_dictionary[pair] += 1 # if yes it increments its count
            else:
                pair_dictionary[pair] = 1 # if no it adds that pair to dictionary and sets original count to 1
        most_occuring_pair = max(pair_dictionary.items(), key=lambda item: item[1]) # finds the most occuring pair, by getting the max count in the dictionary and returning its key i.e the pair
        # hooked pair = most occuring pair
        return most_occuring_pair # returns the hooked/most occuring pair

    def merging(self, token_array:list, hooked_pair:tuple, new_token_id:int): # merges the most occuring pair into a new token and replaces its instances in main and adds to voacbulary
        updated_array = [] # intialzing empty list that will be populated with the new token(for merged tokens) and current tokens (for non merging tokens)
        i = 0 # setting the iteration counter to 0
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
        if i == len(token_array)-1: # if the last element was not part of a pair then we append it here to avoid the indexing error earlier stated
            updated_array.append(token_array[i]) # add the last element to final array
        # if no pairs were found at all it will return the updated array populated entirely by the else block, thus literally
        # returning the normal token array only
        return updated_array


    # decoding should have recursion as there is a possibility that a token formed after merging may consist of the tokens that are again also formed after merging
    # and we can maintain a list of the bytes, and basically add both the left and the right children as part of that list, thus if it branches like a merged token inside a merged token the list will expand and eventually get flattened at the end into one single non nested list i.e the virgin bytes
    def decoding_to_str(self, token_ids:list): # decodes BPE encoded list of tokens it refers to global vocabulary populated with custom tokens/integerids that branch into existing tokens that are part of original (0-255) integer id's bytes
        def get_bytes(token_idx): # a helper function to allow for recursion
            if token_idx <= 255: # checks if the token isnt a new token and is one of the already single token ids
                return [token_idx] # just returns that id since it isnt a pair at all so no merges
            if token_idx in self.entire_vocabulary: # if it wasnt part of the 0 to 255 tokens, we check if the token even exists in our vocab
                left_child, right_child = self.entire_vocabulary[token_idx] # if it does we get the left child and the right child of this new token since it returns a tuple
                return get_bytes(left_child) + get_bytes(right_child) # returns the recursively byte concatination of both lists(the left child if contains another tuples inside it again takes and branches it aswell)
                # basically returns a list again but of the entire constituency of the bytes ( the addition basically appends to a list nothing else)
            return [] # returns empty list if it got some weird ass value

        virgin_bytes = [] # the array of all the bytes in order since spaces are left concatanation is whats only needed
        for token_id in token_ids: # iterates throught the token sequence that needs to be decoded
            virgin_bytes.extend(get_bytes(token_id)) # flattens it, using extend to get a non nested array, and not append or we would get a nested array
        return bytes(virgin_bytes).decode("utf-8", errors="replace") # returns the concatenated final string

    def training(self, text_by_user:str): # the training function that actually initiates the BPE algo
        original_token_list = self.utf_encoding(precoded_text=text_by_user) #storing the original unmutated tokens
        mutable_token_list = original_token_list # creating a copy of the original as mutable
        counter = 1 # setting counter to start
        while counter <= self.vocab_size: # ensuring training produces required vocabulary size only
            hooked_pair,countofpair = self.get_hook(token_array=mutable_token_list) #getting the most occuring adjacent pair from the token array
            if countofpair == 1: # checking to see if we arent forceully creating vocab, making sure if pair has count more than 1 thus occuring more than once
                break # if the max pair only occurs once it means no pair is occuring more than once thus no tokens need to be merged so no new token or vocab needs to be generated so we stop training
            else: # if the max pair has count more than 1 it means this pair needs to be merged and new token needs to generate, Continue training
                mutable_token_list = self.merging(token_array=mutable_token_list, hooked_pair=hooked_pair, new_token_id= 255+counter) # merging the new token and replacing all instances of the merging pair from the token list and setting it to mutable list
                counter+=1 # incrementing the vocab_counter                                                            # | new token generated must have a integer id other than the existing 0 to 255 integer ids, hence for each new vocab we make the new token with integer id equal to that vocab's count on top of the exisiting 255


        return original_token_list,mutable_token_list # returns both the original and updated list of tokens