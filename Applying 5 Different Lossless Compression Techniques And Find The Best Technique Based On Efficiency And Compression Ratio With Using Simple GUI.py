#!/usr/bin/env python
# coding: utf-8

# ## Run Length Encoding Technique

# In[26]:


from collections import Counter
import math

def run_length_encoding(text):
    encoded_text = ''
    count = 1

    for i in range(1, len(text)):
        if text[i] == text[i - 1]:
            count += 1
        else:
            encoded_text += text[i - 1] + str(count)
            count = 1

    encoded_text += text[-1] + str(count)

    return encoded_text


def entropy(probabilities):
    entropy_value = 0
    
    for prob in probabilities.values():
        entropy_value += prob * math.log2(prob)
    
    return -entropy_value


def probability_of_occurrence(text):
    char_count = Counter(text)
    total_chars = len(text)
    probabilities = {}

    for char, count in char_count.items():
        probabilities[char] = round(count / total_chars, 3)

    return probabilities, total_chars

def main():
    text = input("Enter a text message: ")
    encoded_text = run_length_encoding(text)
    
    original_bits = sum(4 if char.isdigit() else 8 for char in text)
    encoded_bits = sum(4 if char.isdigit() else 8 for char in encoded_text)

    compression_ratio = original_bits / encoded_bits 

    probabilities, total_chars = probability_of_occurrence(text)
    text_entropy = entropy(probabilities)
    
    # Calculate the average length of the encoded symbols
    average_length = sum(prob * ((1 if prob == 1 else len(char) + math.ceil(math.log2(total_chars)))) for char, prob in probabilities.items())
    
    efficiency = (text_entropy / average_length) * 100

    print("Original text:", text)
    print("Encoded text:", encoded_text)
    print("Bits before encoding:", original_bits)
    print("Bits after encoding:", encoded_bits)
    print("Compression ratio: ", round(compression_ratio, 4))
    print("Probability of occurrence for each character:", probabilities)
    print("Entropy:", round(text_entropy, 4))
    print("Average length:", round(average_length, 4))
    print("Efficiency:", round(efficiency, 4) , ("%"))

if __name__ == "__main__":
    main()


# In[8]:


def RLE_ratio(text):
    encoded_text = run_length_encoding(text)
    
    original_bits = sum(4 if char.isdigit() else 8 for char in text)
    encoded_bits = sum(4 if char.isdigit() else 8 for char in encoded_text)

    compression_ratio = original_bits / encoded_bits 
    return round(compression_ratio, 4)
    

def RLE_efficiency(text):
    encoded_text = run_length_encoding(text)
    probabilities, total_chars = probability_of_occurrence(text)
    text_entropy = entropy(probabilities)
    
    # Calculate the average length of the encoded symbols
    average_length = sum(prob * ((1 if prob == 1 else len(char) + math.ceil(math.log2(total_chars)))) for char, prob in probabilities.items())
    
    efficiency = (text_entropy / average_length) * 100
    return round(efficiency, 4)


# ## Huffman Encoding Technique

# In[27]:


import heapq
import math

class Node:
    def __init__(self, char, freq, left=None, right=None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right
        
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(char_freq):
    heap = []
    for char, freq in char_freq.items():
        heapq.heappush(heap, Node(char, freq))
    
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        parent = Node(None, left.freq + right.freq, left, right)
        heapq.heappush(heap, parent)
    
    return heap[0]

def build_huffman_codes(root, code, codes):
    if root is None:
        return
    
    if root.char is not None:
        codes[root.char] = code
    
    build_huffman_codes(root.left, code + "0", codes)
    build_huffman_codes(root.right, code + "1", codes)

def huffman_encode(message):
    char_freq = {}
    for char in message:
        if char in char_freq:
            char_freq[char] += 1
        else:
            char_freq[char] = 1
    
    root = build_huffman_tree(char_freq)
    codes = {}
    build_huffman_codes(root, "", codes)
    
    encoded_message = ""
    for char in message:
        encoded_message += codes[char]
    
    return encoded_message, codes

def calculate_metrics(message, encoded_message, codes):
    total_chars = len(message)
    char_freq = {}
    
    for char in message:
        if char in char_freq:
            char_freq[char] += 1
        else:
            char_freq[char] = 1
    
    bits_before_encoding = total_chars * 8
    bits_after_encoding = sum(char_freq[char] * len(codes[char]) for char in char_freq)
    compression_ratio = bits_before_encoding / bits_after_encoding
    
    entropy = 0
    avg_length = 0
    for char, count in char_freq.items():
        probability = count / total_chars
        entropy -= probability * math.log2(probability)
        avg_length += probability * len(codes[char])
    
    efficiency = (entropy / avg_length) * 100

    print("Original Message:", message)
    print("Encoded Message:", encoded_message)
    print("Bits before encoding:", bits_before_encoding)
    print("Bits after encoding:", bits_after_encoding)
    print("Compression ratio:", round(compression_ratio, 4)) 
    print("Probability of occurrence of each character:")
    for char, count in char_freq.items():
        print(char, ":", count / total_chars)
    print("Entropy:", round(entropy, 4))
    print("Average length:", round(avg_length, 4))
    print("Efficiency:", round(efficiency, 4), "%")

# Get user input
message = input("Enter the text message: ")

# Encode the message using Huffman encoding
encoded_message, codes = huffman_encode(message)

# Calculate and print the metrics
calculate_metrics(message, encoded_message, codes)


# In[10]:


def huff_ratio(message, encoded_message, codes):
    total_chars = len(message)
    char_freq = {}
    
    for char in message:
        if char in char_freq:
            char_freq[char] += 1
        else:
            char_freq[char] = 1
    
    bits_before_encoding = total_chars * 8
    bits_after_encoding = sum(char_freq[char] * len(codes[char]) for char in char_freq)
    compression_ratio = bits_before_encoding / bits_after_encoding
    return round(compression_ratio,4)


def huff_efficiency(message, encoded_message, codes):
    total_chars = len(message)
    char_freq = {}
    
    for char in message:
        if char in char_freq:
            char_freq[char] += 1
        else:
            char_freq[char] = 1
    
    entropy = 0
    avg_length = 0
    for char, count in char_freq.items():
        probability = count / total_chars
        entropy -= probability * math.log2(probability)
        avg_length += probability * len(codes[char])
    
    efficiency = (entropy / avg_length) * 100
    return round(efficiency, 4)


# ## Arithmatic Encoding Technique

# In[28]:


import math

def arithmetic_encode(message, char_freq):
    lower = 0.0
    upper = 1.0
    
    for char in message:
        range_size = upper - lower
        upper = lower + range_size * char_freq[char][1]
        lower = lower + range_size * char_freq[char][0]
        
    # Ensure that upper is always greater than lower
    if upper == lower:
        upper += 1e-10
    
    return (lower + upper) / 2, lower, upper

def calculate_metrics(message, encoded_value, lower, upper):
    total_chars = len(message)
    char_freq = {}
    
    for char in message:
        if char in char_freq:
            char_freq[char] += 1
        else:
            char_freq[char] = 1
    
    char_prob = {}
    for char, count in char_freq.items():
        char_prob[char] = count / total_chars
        
    
    bits_before_encoding = total_chars * 8
    bits_after_encoding = math.ceil(-math.log2(upper - lower))
    compression_ratio = bits_before_encoding / bits_after_encoding
    
    entropy = 0
    for char, prob in char_prob.items():
        probability = prob
        entropy -= probability * math.log2(probability)
        
    avg_length = 0
    for char, prob in char_prob.items():
        avg_length += prob * math.ceil(-math.log2(prob))
        
    print("Original Message:", message)
    print("Encoded Value:", encoded_value)
    print("Bits before encoding:", bits_before_encoding)
    print("Bits after encoding:", bits_after_encoding)
    print("Compression ratio:", round(compression_ratio, 4))
    print("Probability of occurrence of each character:")
    for char, prob in char_prob.items():
        print(char, ":", prob)
    print("Entropy:", round(entropy, 4))
    print("Average length:", round(avg_length , 4))
    print("Efficiency:", round((entropy / avg_length)*100 , 4) , ("%"))

# Get user input
message = input("Enter the text message: ")

# Calculate character frequencies and ranges
total_chars = len(message)
char_freq = {}
for char in message:
    if char not in char_freq:
        char_freq[char] = 0
    char_freq[char] += 1

char_prob = {char: count / total_chars for char, count in char_freq.items()}
char_range = {}
start = 0.0
for char, prob in sorted(char_prob.items()):
    char_range[char] = (start, start + prob)
    start += prob

# Encode the message using arithmetic encoding
encoded_value, lower, upper = arithmetic_encode(message, char_range)

# Calculate and print the metrics
calculate_metrics(message, encoded_value, lower, upper)


# In[12]:


def arith_efficiency(message, encoded_value, lower, upper):
    total_chars = len(message)
    char_freq = {}
    
    for char in message:
        if char in char_freq:
            char_freq[char] += 1
        else:
            char_freq[char] = 1
    
    char_prob = {}
    for char, count in char_freq.items():
        char_prob[char] = count / total_chars
        
    
    bits_before_encoding = total_chars * 8
    bits_after_encoding = math.ceil(-math.log2(upper - lower))
    compression_ratio = bits_before_encoding / bits_after_encoding
    
    entropy = 0
    for char, prob in char_prob.items():
        probability = prob
        entropy -= probability * math.log2(probability)
        
    avg_length = 0
    for char, prob in char_prob.items():
        avg_length += prob * math.ceil(-math.log2(prob))
    efficiency = (entropy/avg_length ) *100
    return round(efficiency, 4)
def arith_ratio(message, encoded_value, lower, upper):
    total_chars = len(message)
    char_freq = {}
    
    for char in message:
        if char in char_freq:
            char_freq[char] += 1
        else:
            char_freq[char] = 1
    
    char_prob = {}
    for char, count in char_freq.items():
        char_prob[char] = count / total_chars
        
    
    bits_before_encoding = total_chars * 8
    bits_after_encoding = math.ceil(-math.log2(upper - lower))
    compression_ratio = bits_before_encoding / bits_after_encoding
    
    return round(compression_ratio,4)


# ## Golomb Encoding Technique

# In[29]:


import math
from collections import Counter


def golomb_encode(message, m):
    encoded_message = ""
    for char in message:
        char_code = ord(char)
        quotient = char_code // m
        remainder = char_code % m
        # Unary coding for quotient
        unary_code = "1" * quotient + "0"
        # Binary coding for remainder
        b = math.ceil(math.log2(m))
        if remainder < 2 ** b - m:
            binary_code = bin(remainder - 1)[2:].zfill(b)
        else:
            binary_code = bin(remainder)[2:].zfill(b)
    
        encoded_message += unary_code + binary_code

    return encoded_message

def calculate_metrics(message, encoded_message):
    char_count = {}
    total_chars = len(message)
    
    for char in message:
        if char in char_count:
            char_count[char] += 1
        else:
            char_count[char] = 1
    
    bits_before_encoding = total_chars * 8
    bits_after_encoding = len(encoded_message)
    compression_ratio = bits_before_encoding / bits_after_encoding
    
    entropy = 0
    avg_length = 0
    for char, count in char_count.items():
        probability = count / total_chars
        entropy -= probability * math.log2(probability)
        encoded_char = golomb_encode(char, m)
        avg_length += probability * len(encoded_char)
        efficiency = (entropy / avg_length) * 100
    
    print("Original Message:" , message)
    print("Encoded Message:" ,encoded_message)
    print("Bits before encoding:", bits_before_encoding)
    print("Bits after encoding:", bits_after_encoding)
    print("Compression ratio:", round(compression_ratio , 4))
    print("Probability of occurrence of each character:")
    for char, count in char_count.items():
        print(char, ":", count / total_chars)
    print("Entropy:", round(entropy , 4))
    print("Average length:", round(avg_length , 4))
    print("Efficiency:", round(efficiency , 4) , ("%"))

# Get user input
message = input("Enter the text message: ")
m = int(input("Enter the Golomb coding parameter m: "))

encoded_message = golomb_encode(message, m).replace('b', '')
# Calculate and print the metrics
calculate_metrics(message, encoded_message)


# In[14]:


def golomb_efficiency(message, encoded_message):
    char_count = {}
    total_chars = len(message)
    
    for char in message:
        if char in char_count:
            char_count[char] += 1
        else:
            char_count[char] = 1
    
    bits_before_encoding = total_chars * 8
    bits_after_encoding = len(encoded_message)
    compression_ratio = bits_before_encoding / bits_after_encoding
    
    entropy = 0
    avg_length = 0
    for char, count in char_count.items():
        probability = count / total_chars
        entropy -= probability * math.log2(probability)
        encoded_char = golomb_encode(char, m)
        avg_length += probability * len(encoded_char)
    efficiency = (entropy / avg_length) * 100
        
    return round(efficiency, 4)

def golomb_ratio(message, encoded_message):
    char_count = {}
    total_chars = len(message)
    
    for char in message:
        if char in char_count:
            char_count[char] += 1
        else:
            char_count[char] = 1
    
    bits_before_encoding = total_chars * 8
    bits_after_encoding = len(encoded_message)
    compression_ratio = bits_before_encoding / bits_after_encoding
    
    return round(compression_ratio,4)


# ## LZW Encoding Technique

# In[30]:


import math

def lzw_encode(message):
    dictionary = {chr(i): i for i in range(256)}
    next_code = 256
    encoded_data = []
    current_string = ""
    
    for char in message:
        new_string = current_string + char
        if new_string in dictionary:
            current_string = new_string
        else:
            encoded_data.append(dictionary[current_string])
            dictionary[new_string] = next_code
            next_code += 1
            current_string = char
    
    if current_string:
        encoded_data.append(dictionary[current_string])
    
    return encoded_data

def calculate_metrics(message, encoded_data):
    total_chars = len(message)
    char_freq = {}
    
    for char in message:
        if char in char_freq:
            char_freq[char] += 1
        else:
            char_freq[char] = 1
    
    bits_before_encoding = total_chars * 8
    bits_after_encoding = len(encoded_data) * math.ceil(math.log2(len(encoded_data) + 256))
    compression_ratio = bits_before_encoding / bits_after_encoding
    
    entropy = 0
    avg_length = 0
    codes = {}  # Define the codes dictionary
    for char, count in char_freq.items():
        probability = count / total_chars
        entropy -= probability * math.log2(probability)
        codes[char] = math.ceil(math.log2(len(codes) + 256))  # Assign the correct code length based on encoding
        avg_length += probability * codes[char]  # Calculate the average length
    
    print("Original Message:" , message)
    print("Encoded Data:" , encoded_data)
    print("Bits before encoding:", bits_before_encoding)
    print("Bits after encoding:", bits_after_encoding)
    print("Compression ratio:", round(compression_ratio , 4))
    print("Probability of occurrence of each character:")
    for char, count in char_freq.items():
        print(char, ":", count / total_chars)
    print("Entropy:", round(entropy , 4))
    print("Average length:", round(avg_length , 4))
    print("Efficiency:", round((entropy/avg_length) * 100, 4) , ("%"))  # Calculate the efficiency

# Get user input
message = input("Enter the text message: ")

# Encode the message using LZW encoding
encoded_data = lzw_encode(message)

# Calculate and print the metrics
calculate_metrics(message, encoded_data)


# In[16]:


def lzw_ratio(message, encoded_data):
    total_chars = len(message)
    char_freq = {}
    
    for char in message:
        if char in char_freq:
            char_freq[char] += 1
        else:
            char_freq[char] = 1
    
    bits_before_encoding = total_chars * 8
    bits_after_encoding = len(encoded_data) * math.ceil(math.log2(len(encoded_data) + 256))
    compression_ratio = bits_before_encoding / bits_after_encoding
    
    return round(compression_ratio,4)

def lzw_efficiency(message, encoded_data):
    total_chars = len(message)
    char_freq = {}
    
    for char in message:
        if char in char_freq:
            char_freq[char] += 1
        else:
            char_freq[char] = 1
    
    
    entropy = 0
    avg_length = 0
    codes = {}  # Define the codes dictionary
    for char, count in char_freq.items():
        probability = count / total_chars
        entropy -= probability * math.log2(probability)
        codes[char] = math.ceil(math.log2(len(codes) + 256))  # Assign the correct code length based on encoding
        avg_length += probability * codes[char]  # Calculate the average length
    
    efficiency = (entropy/avg_length)*100
    return round(efficiency, 4)


# ## Finding Optimal Technique Based on Compression Ratio

# In[17]:


message = input("Enter the text message: ")

# RLE
rle = RLE_ratio(message)

# Huffman
h_encoded_message, codes = huffman_encode(message)
huffman = huff_ratio(message, h_encoded_message, codes)

# Arithmetic
total_chars = len(message)
char_freq = {}
for char in message:
    if char not in char_freq:
        char_freq[char] = 0
    char_freq[char] += 1

char_prob = {char: count / total_chars for char, count in char_freq.items()}
char_range = {}
start = 0.0
for char, prob in sorted(char_prob.items()):
    char_range[char] = (start, start + prob)
    start += prob
a_encoded_value, lower, upper = arithmetic_encode(message, char_range)
arithmatic = arith_ratio(message, a_encoded_value, lower, upper)

# Golomb
g_encoded_message = golomb_encode(message, m).replace('b', '')
golomb = golomb_ratio(message, g_encoded_message)

# LZW
l_encoded_data = lzw_encode(message)
lzw = lzw_ratio(message, l_encoded_data)

# Print compression ratios and determine optimal model(s)
compression_ratios = {
    "RLE": rle,
    "Huffman": huffman,
    "Arithmetic": arithmatic,
    "Golomb": golomb,
    "LZW": lzw
}

max_ratio = max(compression_ratios.values())
optimal_models = [model for model, ratio in compression_ratios.items() if ratio == max_ratio]

print("Compression ratios:")
for model, ratio in compression_ratios.items():
    print(f"{model}: {ratio}")

print("\nOptimal model(s):")
for model in optimal_models:
    print(f"{model} with compression ratio {max_ratio}")


# <h1>Finding optimal model using efficiency</h1>

# In[18]:


message = input("Enter the text message: ")

# RLE
rle = RLE_efficiency(message)

# Huffman
h_encoded_message, codes = huffman_encode(message)
huffman = huff_efficiency(message, h_encoded_message, codes)

# Arithmetic
total_chars = len(message)
char_freq = {}
for char in message:
    if char not in char_freq:
        char_freq[char] = 0
    char_freq[char] += 1

char_prob = {char: count / total_chars for char, count in char_freq.items()}
char_range = {}
start = 0.0
for char, prob in sorted(char_prob.items()):
    char_range[char] = (start, start + prob)
    start += prob
a_encoded_value, lower, upper = arithmetic_encode(message, char_range)
arithmatic = arith_efficiency(message, a_encoded_value, lower, upper)

# Golomb
g_encoded_message = golomb_encode(message, m).replace('b', '')
golomb = golomb_efficiency(message, g_encoded_message)

# LZW
l_encoded_data = lzw_encode(message)
lzw = lzw_efficiency(message, l_encoded_data)

# Print efficiencies and determine optimal model(s)
model_efficiency = {
    "RLE": rle,
    "Huffman": huffman,
    "Arithmetic": arithmatic,
    "Golomb": golomb,
    "LZW": lzw
}

max_efficiency = max(model_efficiency.values())
optimal_models = [model for model, efficiency in model_efficiency.items() if efficiency == max_efficiency]

print("Model Efficiencies:")
for model, efficiency in model_efficiency.items():
    print(f"{model}: {efficiency} %")

print("\nOptimal model(s):")
for model in optimal_models:
    print(f"{model} with efficiency {max_efficiency} %")


# In[22]:


#########################################
 ## Compression Techniques
#########################################
import tkinter as tk
from tkinter import ttk, messagebox
import math
import heapq
from collections import Counter

class Node:
    def __init__(self, char, freq, left=None, right=None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right
        
    def __lt__(self, other):
        return self.freq < other.freq

def entropy(probabilities):
    return sum(-p * math.log2(p) for p in probabilities.values())

def arithmetic_encode(message, char_freq):
    lower = 0.0
    upper = 1.0
    
    for char in message:
        range_size = upper - lower
        upper = lower + range_size * char_freq[char][1]
        lower = lower + range_size * char_freq[char][0]
        
    # Ensure that upper is always greater than lower
    if upper == lower:
        upper += 1e-10
    
    return (lower + upper) / 2, lower, upper

def golomb_encode(message, m):
    encoded_message = ""
    for char in message:
        char_code = ord(char)
        quotient = char_code // m
        remainder = char_code % m
        # Unary coding for quotient
        unary_code = "1" * quotient + "0"
        # Binary coding for remainder
        b = math.ceil(math.log2(m))
        if remainder < 2 ** b - m:
            binary_code = bin(remainder - 1)[2:].zfill(b)
        else:
            binary_code = bin(remainder)[2:].zfill(b)
    
        encoded_message += unary_code + binary_code

    return encoded_message

def lzw_encode(message):
    dictionary = {chr(i): i for i in range(256)}
    next_code = 256
    encoded_data = []
    current_string = ""
    
    for char in message:
        new_string = current_string + char
        if new_string in dictionary:
            current_string = new_string
        else:
            encoded_data.append(dictionary[current_string])
            dictionary[new_string] = next_code
            next_code += 1
            current_string = char
    
    if current_string:
        encoded_data.append(dictionary[current_string])
    
    return encoded_data

def run_length_encoding(text):
    encoded_text = ''
    count = 1

    for i in range(1, len(text)):
        if text[i] == text[i - 1]:
            count += 1
        else:
            encoded_text += text[i - 1] + str(count)
            count = 1

    encoded_text += text[-1] + str(count)

    return encoded_text

def build_huffman_tree(char_freq):
    heap = []
    for char, freq in char_freq.items():
        heapq.heappush(heap, Node(char, freq))
    
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        parent = Node(None, left.freq + right.freq, left, right)
        heapq.heappush(heap, parent)
    
    return heap[0]

def build_huffman_codes(root, code, codes):
    if root is None:
        return
    
    if root.char is not None:
        codes[root.char] = code
    
    build_huffman_codes(root.left, code + "0", codes)
    build_huffman_codes(root.right, code + "1", codes)

def huffman_encode(message):
    char_freq = {}
    for char in message:
        if char in char_freq:
            char_freq[char] += 1
        else:
            char_freq[char] = 1
    
    root = build_huffman_tree(char_freq)
    codes = {}
    build_huffman_codes(root, "", codes)
    
    encoded_message = ""
    for char in message:
        encoded_message += codes[char]
    
    return encoded_message, codes

def calculate_metrics(message, encoded_value, lower, upper):
    total_chars = len(message)
    char_freq = {}
    
    for char in message:
        if char in char_freq:
            char_freq[char] += 1
        else:
            char_freq[char] = 1
    
    char_prob = {}
    for char, count in char_freq.items():
        char_prob[char] = count / total_chars
        
    bits_before_encoding = total_chars * 8
    bits_after_encoding = math.ceil(-math.log2(upper - lower))
    compression_ratio = bits_before_encoding / bits_after_encoding
    
    entropy = 0
    for char, prob in char_prob.items():
        probability = prob
        entropy -= probability * math.log2(prob)
        
    avg_length = 0
    for char, prob in char_prob.items():
        avg_length += prob * math.ceil(-math.log2(prob))
        
    efficiency = (entropy / avg_length) * 100
    
    return bits_before_encoding, bits_after_encoding, compression_ratio, char_prob, entropy, avg_length, efficiency

def arith_efficiency(message, encoded_value, lower, upper):
    bits_before_encoding, bits_after_encoding, _, char_prob, entropy, avg_length, _ = calculate_metrics(message, encoded_value, lower, upper)
    efficiency = (entropy / avg_length ) * 100
    return efficiency

def arith_ratio(message, encoded_value, lower, upper):
    bits_before_encoding, bits_after_encoding, _, _, _, _, _ = calculate_metrics(message, encoded_value, lower, upper)
    compression_ratio = bits_before_encoding / bits_after_encoding
    return round(compression_ratio , 4)

def golomb_efficiency(message, encoded_message, m):
    char_count = {}
    total_chars = len(message)
    
    for char in message:
        if char in char_count:
            char_count[char] += 1
        else:
            char_count[char] = 1
    
    bits_before_encoding = total_chars * 8
    bits_after_encoding = len(encoded_message)
    compression_ratio = bits_before_encoding / bits_after_encoding
    
    entropy = 0
    avg_length = 0
    for char, count in char_count.items():
        probability = count / total_chars
        entropy -= probability * math.log2(probability)
        encoded_char = golomb_encode(char, m)
        avg_length += probability * len(encoded_char)
    efficiency = (entropy / avg_length) * 100
        
    return round(efficiency, 4)

def golomb_ratio(message, encoded_message):
    char_count = {}
    total_chars = len(message)
    
    for char in message:
        if char in char_count:
            char_count[char] += 1
        else:
            char_count[char] = 1
    
    bits_before_encoding = total_chars * 8
    bits_after_encoding = len(encoded_message)
    compression_ratio = bits_before_encoding / bits_after_encoding
    
    return round(compression_ratio, 4)

def lzw_efficiency(message, encoded_data):
    total_chars = len(message)
    char_freq = {}
    
    for char in message:
        if char in char_freq:
            char_freq[char] += 1
        else:
            char_freq[char] = 1
    
    bits_before_encoding = total_chars * 8
    bits_after_encoding = len(encoded_data) * math.ceil(math.log2(len(encoded_data) + 256))
    compression_ratio = bits_before_encoding / bits_after_encoding
    
    entropy = 0
    avg_length = 0
    codes = {}  
    for char, count in char_freq.items():
        probability = count / total_chars
        entropy -= probability * math.log2(probability)
        codes[char] = math.ceil(math.log2(len(codes) + 256))
        avg_length += probability * codes[char]
    efficiency = (entropy / avg_length) * 100
    
    return round(efficiency, 4)

def lzw_ratio(message, encoded_data):
    total_chars = len(message)
    bits_before_encoding = total_chars * 8
    bits_after_encoding = len(encoded_data) * math.ceil(math.log2(len(encoded_data) + 256))
    compression_ratio = bits_before_encoding / bits_after_encoding
    
    return round(compression_ratio, 4)

def rle_efficiency(text):
    char_count = Counter(text)
    total_chars = len(text)
    probabilities = {}

    for char, count in char_count.items():
        probabilities[char] = round(count / total_chars, 3)

    text_entropy = entropy(probabilities)
    
    average_length = sum(prob * ((1 if prob == 1 else len(char) + math.ceil(math.log2(total_chars)))) for char, prob in probabilities.items())
    
    efficiency = (text_entropy / average_length) * 100

    return round(efficiency, 4)

def rle_ratio(text, encoded_text):
    original_bits = sum(4 if char.isdigit() else 8 for char in text)
    encoded_bits = sum(4 if char.isdigit() else 8 for char in encoded_text)
    compression_ratio = original_bits / encoded_bits 

    return round(compression_ratio, 4)

def huff_ratio(message, encoded_message, codes):
    total_chars = len(message)
    char_freq = {}
    
    for char in message:
        if char in char_freq:
            char_freq[char] += 1
        else:
            char_freq[char] = 1
    
    bits_before_encoding = total_chars * 8
    bits_after_encoding = sum(char_freq[char] * len(codes[char]) for char in char_freq)
    compression_ratio = bits_before_encoding / bits_after_encoding
    return round(compression_ratio,4)

def huff_efficiency(message, encoded_message, codes):
    total_chars = len(message)
    char_freq = {}
    
    for char in message:
        if char in char_freq:
            char_freq[char] += 1
        else:
            char_freq[char] = 1
    
    entropy = 0
    avg_length = 0
    for char, count in char_freq.items():
        probability = count / total_chars
        entropy -= probability * math.log2(probability)
        avg_length += probability * len(codes[char])
    
    efficiency = (entropy / avg_length) * 100
    return round(efficiency, 4)


def on_encode():
    selected_technique = technique_var.get()
    message = input_text.get("1.0", "end-1c")
    
    if selected_technique == "Arithmetic Coding":
        total_chars = len(message)
        char_freq = {}
        for char in message:
            if char not in char_freq:
                char_freq[char] = 0
            char_freq[char] += 1

        char_prob = {char: count / total_chars for char, count in char_freq.items()}
        char_range = {}
        start = 0.0
        for char, prob in sorted(char_prob.items()):
            char_range[char] = (start, start + prob)
            start += prob

        encoded_value, lower, upper = arithmetic_encode(message, char_range)
        metrics = calculate_metrics(message, encoded_value, lower, upper)
        
        result_str = "Original Message:\n{}\n".format(message)
        result_str += "Encoded Message:\n{}\n".format(encoded_value)
        result_str += "Bits before encoding: {}\n".format(metrics[0])
        result_str += "Bits after encoding: {}\n".format(metrics[1])
        result_str += "Compression ratio: {}\n".format(round(metrics[2], 4))
        result_str += "Probability of occurrence of each character:\n"
        for char, prob in char_prob.items():
            result_str += "{}: {}\n".format(char, prob)
        result_str += "Entropy: {}\n".format(round(metrics[4], 4))
        result_str += "Average length: {}\n".format(round(metrics[5], 4))
        result_str += "Efficiency: {}%\n".format(round(metrics[6], 4))


    elif selected_technique == "Golomb Coding":
        m_value = int(m_entry.get())
        encoded_message = golomb_encode(message, m_value).replace('b', '')
        
        result_str = "Original Message:\n{}\n".format(message)
        result_str += "Encoded Message:\n{}\n".format(encoded_message)
        result_str += "Bits before encoding: {}\n".format(len(message) * 8)
        result_str += "Bits after encoding: {}\n".format(len(encoded_message))
        result_str += "Compression ratio: {}\n".format(round(golomb_ratio(message, encoded_message), 4))
        result_str += "Probability of occurrence of each character:\n"
        char_count = Counter(message)
        total_chars = len(message)
        char_probabilities = {char: count / total_chars for char, count in char_count.items()}
        entropy = 0
        avg_length = 0
        for char, count in char_count.items():
            probability = count / total_chars
            entropy -= probability * math.log2(probability)
            encoded_char = golomb_encode(char, m)
            avg_length += probability * len(encoded_char)
            efficiency = (entropy / avg_length) * 100
        for char, prob in char_probabilities.items():
            result_str += "{}: {}\n".format(char, prob)
            
        result_str += "Entropy: {}\n".format(round(entropy, 4))
        result_str += "Average length: {}\n".format(round(avg_length, 4))
        result_str += "Efficiency: {}%\n".format(round(efficiency, 4))
        

        
    elif selected_technique == "LZW Coding":
        encoded_data = lzw_encode(message)
    
        result_str = "Original Message:\n{}\n".format(message)
        result_str += "Encoded Message:\n{}\n".format(encoded_data)
        result_str += "Bits before encoding: {}\n".format(len(message) * 8)
        result_str += "Bits after encoding: {}\n".format(len(encoded_data) * math.ceil(math.log2(len(encoded_data) + 256)))
        result_str += "Compression ratio: {}\n".format(round(lzw_ratio(message, encoded_data), 4))
        result_str += "Probability of occurrence of each character:\n"
        char_count = Counter(message)
        total_chars = len(message)
        char_probabilities = {char: count / total_chars for char, count in char_count.items()}
        for char, prob in char_probabilities.items():
            result_str += "{}: {}\n".format(char, prob)
        entropy = 0
        avg_length = 0
        codes = {}  # Define the codes dictionary
        for char, count in char_count.items():
            probability = count / total_chars
            entropy -= probability * math.log2(probability)
            codes[char] = math.ceil(math.log2(len(codes) + 256))  # Assign the correct code length based on encoding
            avg_length += probability * codes[char]  # Calculate the average length
            encoded_data = lzw_encode(message)
        result_str += "Entropy: {}\n".format(round(entropy, 4))
        result_str += "Average length: {}\n".format(round(avg_length, 4))
        result_str += "Efficiency: {}%\n".format(round(lzw_efficiency(message, encoded_data), 4))

        
    elif selected_technique == "Run Length Encoding (RLE)":
        encoded_text = run_length_encoding(message)
        
        result_str = "Original Message:\n{}\n".format(message)
        result_str += "Encoded Message:\n{}\n".format(encoded_text)
        result_str += "Bits before encoding: {}\n".format(sum(4 if char.isdigit() else 8 for char in message))
        result_str += "Bits after encoding: {}\n".format(sum(4 if char.isdigit() else 8 for char in encoded_text))
        result_str += "Compression ratio: {}\n".format(round(rle_ratio(message, encoded_text), 4))
        result_str += "Probability of occurrence of each character:\n"
        char_count = Counter(message)
        total_chars = len(message)
        probabilities = {}
        for char, count in char_count.items():
            probabilities[char] = round(count / total_chars, 3)
        char_probabilities = {char: count / total_chars for char, count in char_count.items()}
        for char, prob in char_probabilities.items():
            result_str += "{}: {}\n".format(char, round(prob,3))
        def entropy(probabilities):
            entropy_value = 0
            for prob in probabilities.values():
                entropy_value += prob * math.log2(prob)
            return -entropy_value

        result_str += "Entropy: {}\n".format(round(entropy(probabilities), 4))
        result_str += "Average length: {}\n".format(round(sum(prob * ((1 if prob == 1 else len(char) + math.ceil(math.log2(total_chars)))) for char, prob in probabilities.items()) ,4))
        result_str += "Efficiency: {}%\n".format(round(rle_efficiency(message), 4))

        
    elif selected_technique == "Huffman Coding":
        encoded_message, codes = huffman_encode(message)
    
        result_str = "Original Message:\n{}\n".format(message)
        result_str += "Encoded Message:\n{}\n".format(encoded_message)
        result_str += "Bits before encoding: {}\n".format(len(message) * 8)
        result_str += "Bits after encoding: {}\n".format(len(encoded_message))
        result_str += "Compression ratio: {}\n".format(round(huff_ratio(message, encoded_message, codes), 4))
        result_str += "Probability of occurrence of each character:\n"
        char_count = Counter(message)
        total_chars = len(message)
        char_probabilities = {char: count / total_chars for char, count in char_count.items()}
        for char, prob in char_probabilities.items():
            result_str += "{}: {}\n".format(char, prob)
       
        char_freq = {}
    
        for char in message:
            if char in char_freq:
                char_freq[char] += 1
            else:
                char_freq[char] = 1
                
        entropy = 0
        avg_length = 0
        for char, count in char_freq.items():
            probability = count / total_chars
            entropy -= probability * math.log2(probability)
            avg_length += probability * len(codes[char])
            efficiency = (entropy / avg_length) * 100
            
        result_str += "Entropy: {}\n".format(round(entropy, 4))
        result_str += "Average length: {}\n".format(round(avg_length, 4))
        result_str += "Efficiency: {}%\n".format(round(efficiency, 4))

        
    else:
        result_str = "Invalid encoding technique selection."
        
    messagebox.showinfo("Encoding Results", result_str)

# GUI
root = tk.Tk()
root.title("Encoding Techniques")

# Input text
input_frame = ttk.Frame(root)
input_frame.pack(pady=10 , padx=20)

input_label = ttk.Label(input_frame, text="Enter the text message :")
input_label.grid(row=0, column=0)

input_text = tk.Text(input_frame, width=40, height=1)
input_text.grid(row=0, column=1)

# Encoding Technique selection
technique_var = tk.StringVar()
technique_var.set("Run Length Encoding (RLE)")

technique_frame = ttk.Frame(root)
technique_frame.pack(pady=10)

technique_label = ttk.Label(technique_frame, text="Select Encoding Technique :")
technique_label.grid(row=0, column=0)

technique_menu = ttk.OptionMenu(technique_frame, technique_var, "Run Length Encoding (RLE)", "Run Length Encoding (RLE)", "Huffman Coding", "Arithmetic Coding", "Golomb Coding", "LZW Coding")

technique_menu.grid(row=0, column=1)

# Parameters for Golomb and LZW Coding
param_frame = ttk.Frame(root)
param_frame.pack(pady=5)

param_label = ttk.Label(param_frame, text="Enter the parameter m for Golomb Coding : ")
param_label.grid(row=0, column=0)

m_entry = ttk.Entry(param_frame)
m_entry.grid(row=0, column=1)

# Function to show parameter entry for Golomb Encoding
def show_golomb_params():
    param_label.grid(row=0, column=0)
    m_entry.grid(row=0, column=1)

# Function to hide parameter entry for Golomb Encoding
def hide_golomb_params():
    param_label.grid_forget()
    m_entry.grid_forget()

# Encode Button
encode_btn = ttk.Button(root, text="Encode", command=on_encode)
encode_btn.pack(pady=10)

# Initially hide Golomb parameters
hide_golomb_params()

# Binding the functions to show/hide parameters based on selection
technique_var.trace_add("write", lambda *args: show_golomb_params() if technique_var.get() == "Golomb Coding" else hide_golomb_params())

root.mainloop()


# In[23]:


import tkinter as tk
from tkinter import ttk
import math
import heapq
from collections import Counter

# Run Length Encoding (RLE)
def run_length_encoding(text):
    encoded_text = ''
    count = 1

    for i in range(1, len(text)):
        if text[i] == text[i - 1]:
            count += 1
        else:
            encoded_text += text[i - 1] + str(count)
            count = 1

    encoded_text += text[-1] + str(count)

    return encoded_text

def RLE_ratio(text):
    encoded_text = run_length_encoding(text)
    
    original_bits = sum(4 if char.isdigit() else 8 for char in text)
    encoded_bits = sum(4 if char.isdigit() else 8 for char in encoded_text)

    compression_ratio = original_bits / encoded_bits 
    return round(compression_ratio, 4)

def RLE_efficiency(text):
    encoded_text = run_length_encoding(text)
    probabilities, total_chars = probability_of_occurrence(text)
    text_entropy = entropy(probabilities)
    
    # Calculate the average length of the encoded symbols
    average_length = sum(prob * ((1 if prob == 1 else len(char) + math.ceil(math.log2(total_chars)))) for char, prob in probabilities.items())
    
    efficiency = (text_entropy / average_length) * 100
    return round(efficiency, 4)

# Huffman Encoding
class Node:
    def __init__(self, char, freq, left=None, right=None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right
        
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(char_freq):
    heap = []
    for char, freq in char_freq.items():
        heapq.heappush(heap, Node(char, freq))
    
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        parent = Node(None, left.freq + right.freq, left, right)
        heapq.heappush(heap, parent)
    
    return heap[0]

def build_huffman_codes(root, code, codes):
    if root is None:
        return
    
    if root.char is not None:
        codes[root.char] = code
    
    build_huffman_codes(root.left, code + "0", codes)
    build_huffman_codes(root.right, code + "1", codes)

def huffman_encode(message):
    char_freq = {}
    for char in message:
        if char in char_freq:
            char_freq[char] += 1
        else:
            char_freq[char] = 1
    
    root = build_huffman_tree(char_freq)
    codes = {}
    build_huffman_codes(root, "", codes)
    
    encoded_message = ""
    for char in message:
        encoded_message += codes[char]
    
    return encoded_message, codes

def huff_ratio(message, encoded_message, codes):
    total_chars = len(message)
    char_freq = {}
    
    for char in message:
        if char in char_freq:
            char_freq[char] += 1
        else:
            char_freq[char] = 1
    
    bits_before_encoding = total_chars * 8
    bits_after_encoding = sum(char_freq[char] * len(codes[char]) for char in char_freq)
    compression_ratio = bits_before_encoding / bits_after_encoding
    
    return round(compression_ratio, 4)

def huff_efficiency(message, encoded_message, codes):
    total_chars = len(message)
    char_freq = {}
    
    for char in message:
        if char in char_freq:
            char_freq[char] += 1
        else:
            char_freq[char] = 1
    
    entropy = 0
    avg_length = 0
    for char, count in char_freq.items():
        probability = count / total_chars
        entropy -= probability * math.log2(probability)
        avg_length += probability * len(codes[char])
    
    efficiency = (entropy / avg_length) * 100
    return round(efficiency, 4)

# Arithmetic Encoding
def arithmetic_encode(message, char_freq):
    lower = 0.0
    upper = 1.0
    
    for char in message:
        range_size = upper - lower
        upper = lower + range_size * char_freq[char][1]
        lower = lower + range_size * char_freq[char][0]
        
    # Ensure that upper is always greater than lower
    if upper == lower:
        upper += 1e-10
    
    return (lower + upper) / 2, lower, upper

def arith_ratio(message, encoded_value, lower, upper):
    total_chars = len(message)
    char_freq = {}
    
    for char in message:
        if char in char_freq:
            char_freq[char] += 1
        else:
            char_freq[char] = 1
    
    bits_before_encoding = total_chars * 8
    bits_after_encoding = math.ceil(-math.log2(upper - lower))
    compression_ratio = bits_before_encoding / bits_after_encoding
    
    return round(compression_ratio , 4)

def arith_efficiency(message, encoded_value, lower, upper):
    total_chars = len(message)
    char_freq = {}
    
    for char in message:
        if char in char_freq:
            char_freq[char] += 1
        else:
            char_freq[char] = 1
    
    entropy = 0
    avg_length = 0
    for char, count in char_freq.items():
        probability = count / total_chars
        entropy -= probability * math.log2(probability)
        avg_length += probability * math.ceil(-math.log2(probability))
    
    efficiency = (entropy / avg_length) * 100
    return round(efficiency, 4)

# Golomb Encoding
def golomb_encode(message, m):
    encoded_message = ""
    for char in message:
        char_code = ord(char)
        quotient = char_code // m
        remainder = char_code % m
        # Unary coding for quotient
        unary_code = "1" * quotient + "0"
        # Binary coding for remainder
        b = math.ceil(math.log2(m))
        if remainder < 2 ** b - m:
            binary_code = bin(remainder - 1)[2:].zfill(b)
        else:
            binary_code = bin(remainder)[2:].zfill(b)
    
        encoded_message += unary_code + binary_code

    return encoded_message

def golomb_ratio(message, encoded_message):
    total_chars = len(message)
    bits_before_encoding = total_chars * 8
    bits_after_encoding = len(encoded_message)
    compression_ratio = bits_before_encoding / bits_after_encoding
    
    return round(compression_ratio, 4)

def golomb_efficiency(message, encoded_message):
    total_chars = len(message)
    char_count = Counter(message)
    
    entropy = 0
    avg_length = 0
    for char, count in char_count.items():
        probability = count / total_chars
        entropy -= probability * math.log2(probability)
        avg_length += probability * len(encoded_message) / total_chars
    
    efficiency = (entropy / avg_length) * 100
    return round(efficiency, 4)

# LZW Encoding
def lzw_encode(message):
    dictionary = {chr(i): i for i in range(256)}
    next_code = 256
    encoded_data = []
    current_string = ""
    
    for char in message:
        new_string = current_string + char
        if new_string in dictionary:
            current_string = new_string
        else:
            encoded_data.append(dictionary[current_string])
            dictionary[new_string] = next_code
            next_code += 1
            current_string = char
    
    if current_string:
        encoded_data.append(dictionary[current_string])
    
    return encoded_data

def lzw_ratio(message, encoded_data):
    total_chars = len(message)
    char_freq = {}
    
    for char in message:
        if char in char_freq:
            char_freq[char] += 1
        else:
            char_freq[char] = 1
    
    bits_before_encoding = total_chars * 8
    bits_after_encoding = len(encoded_data) * math.ceil(math.log2(len(encoded_data) + 256))
    compression_ratio = bits_before_encoding / bits_after_encoding
    
    return round(compression_ratio,4)

def lzw_efficiency(message, encoded_data):
    total_chars = len(message)
    char_freq = {}
    
    for char in message:
        if char in char_freq:
            char_freq[char] += 1
        else:
            char_freq[char] = 1
    
    
    entropy = 0
    avg_length = 0
    codes = {}  # Define the codes dictionary
    for char, count in char_freq.items():
        probability = count / total_chars
        entropy -= probability * math.log2(probability)
        codes[char] = math.ceil(math.log2(len(codes) + 256))  # Assign the correct code length based on encoding
        avg_length += probability * codes[char]  # Calculate the average length
    
    efficiency = (entropy/avg_length)*100
    return round(efficiency, 4)

# Function to calculate the probability of occurrence of each character
def probability_of_occurrence(text):
    char_count = Counter(text)
    total_chars = len(text)
    probabilities = {char: count / total_chars for char, count in char_count.items()}
    return probabilities, total_chars

# Function to calculate the entropy of a text
def entropy(probabilities):
    entropy = -sum(prob * math.log2(prob) for prob in probabilities.values())
    return entropy

# Function to calculate compression ratios
def calculate_ratios():
    message = input_entry.get()

    # RLE
    rle = RLE_ratio(message)

    # Huffman
    h_encoded_message, codes = huffman_encode(message)
    huffman = huff_ratio(message, h_encoded_message, codes)

    # Arithmetic
    total_chars = len(message)
    char_freq = {}
    for char in message:
        if char not in char_freq:
            char_freq[char] = 0
        char_freq[char] += 1

    char_prob = {char: count / total_chars for char, count in char_freq.items()}
    char_range = {}
    start = 0.0
    for char, prob in sorted(char_prob.items()):
        char_range[char] = (start, start + prob)
        start += prob
    a_encoded_value, lower, upper = arithmetic_encode(message, char_range)
    arithmatic = arith_ratio(message, a_encoded_value, lower, upper)

    # Golomb
    g_encoded_message = golomb_encode(message, m).replace('b', '')
    golomb = golomb_ratio(message, g_encoded_message)

    # LZW
    l_encoded_data = lzw_encode(message)
    lzw = lzw_ratio(message, l_encoded_data)

    # Print compression ratios and determine optimal model(s)
    compression_ratios = {
        "RLE": rle,
        "Huffman": huffman,
        "Arithmetic": arithmatic,
        "Golomb": golomb,
        "LZW": lzw
    }

    max_ratio = max(compression_ratios.values())
    optimal_models = [model for model, ratio in compression_ratios.items() if ratio == max_ratio]

    output_text = ""
    output_text += "Compression ratios:\n"
    for model, ratio in compression_ratios.items():
        output_text += f"{model}: {ratio}\n"

    output_text += "\nOptimal model(s):\n"
    for model in optimal_models:
        output_text += f"{model} with compression ratio {max_ratio}\n"

    output_label.config(text=output_text)

    output_label.config(text=output_text)

# Function to calculate model efficiencies
def calculate_efficiencies():
    message = input_entry.get()

    # RLE
    rle_eff = RLE_efficiency(message)

    # Huffman
    h_encoded_message, codes = huffman_encode(message)
    huffman_eff = huff_efficiency(message, h_encoded_message, codes)

    # Arithmetic
    total_chars = len(message)
    char_freq = {}
    for char in message:
        if char not in char_freq:
            char_freq[char] = 0
        char_freq[char] += 1

    char_prob = {char: count / total_chars for char, count in char_freq.items()}
    char_range = {}
    start = 0.0
    for char, prob in sorted(char_prob.items()):
        char_range[char] = (start, start + prob)
        start += prob
    a_encoded_value, lower, upper = arithmetic_encode(message, char_range)
    arithmatic_eff = arith_efficiency(message, a_encoded_value, lower, upper)

    # Golomb
    g_encoded_message = golomb_encode(message, m)
    golomb_eff = golomb_efficiency(message, g_encoded_message)

    # LZW
    l_encoded_data = lzw_encode(message)
    lzw_eff = lzw_efficiency(message, l_encoded_data)

    # Calculate and display model efficiencies
    model_efficiency = {
        "RLE": rle_eff,
        "Huffman": huffman_eff,
        "Arithmetic": arithmatic_eff,
        "Golomb": golomb_eff,
        "LZW": lzw_eff
    }

    max_efficiency = max(model_efficiency.values())
    optimal_models = [model for model, eff in model_efficiency.items() if eff == max_efficiency]

    output_text = "Model Efficiencies:\n"
    for model, eff in model_efficiency.items():
        output_text += f"{model}: {eff}%\n"

    # Display all optimal models
    output_text += "\nOptimal model(s) with the highest efficiency:\n"
    for model in optimal_models:
        output_text += f"{model} with efficiency {model_efficiency[model]}%\n"

    output_label.config(text=output_text)

# Create main window
root = tk.Tk()
root.title("Compression Analysis")

# Create input label and entry
input_label = ttk.Label(root, text="Enter the text message:")
input_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
input_entry = ttk.Entry(root, width=50)
input_entry.grid(row=0, column=1, padx=5, pady=5)

# Create buttons for calculating compression ratios and model efficiencies
ratio_button = ttk.Button(root, text="Calculate Ratios", command=calculate_ratios)
ratio_button.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

efficiency_button = ttk.Button(root, text="Calculate Efficiencies", command=calculate_efficiencies)
efficiency_button.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

# Create output label
output_label = ttk.Label(root, text="")
output_label.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="w")

root.mainloop()


# In[ ]:




