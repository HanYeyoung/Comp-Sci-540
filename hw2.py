import sys
import math
import string

def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    described in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    '''
    #Implementing vectors e,s as lists (arrays) of length 26
    #with p[0] being the probability of 'A' and so on
    e=[0]*26
    s=[0]*26

    with open('e.txt',encoding='utf-8') as f:
        for line in f:
            #strip: removes the newline character
            #split: split the string on space character
            char,prob=line.strip().split(" ")
            #ord('E') gives the ASCII (integer) value of character 'E'
            #we then subtract it from 'A' to give array index
            #This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char)-ord('A')]=float(prob)
    f.close()

    with open('s.txt',encoding='utf-8') as f:
        for line in f:
            char,prob=line.strip().split(" ")
            s[ord(char)-ord('A')]=float(prob)
    f.close()

    return (e,s)

def shred(filename):
    #Using a dictionary here. You may change this to any data structure of
    #your choice such as lists (X=[]) etc. for the assignment
    X = dict.fromkeys(string.ascii_uppercase, 0)

    with open (filename,encoding='utf-8') as f:
        # TODO: add your code here
        data = f.read().upper()  # Convert all characters to uppercase
        for i in data:
            if i in string.ascii_uppercase:  # Check if character is an alphabet
                X[i] += 1

    return X

# TODO: add your code here for the assignment
# You are free to implement it as you wish!
# Happy Coding!

def main():
    # Print "Q1" followed by the 26 character counts for letter.txt
    print("Q1")
    X = shred("letter.txt")
    for key, value in X.items():
        print(f"{key} {value}")

    # Compute X1loge1 and X1logs1
    # Print "Q2" then these values up to 4 decimal places on two separate lines

    e, s = get_parameter_vectors()

    X1 = X['A']
    value_e = X1 * math.log(e[0])
    value_s = X1 * math.log(s[0])
    print("Q2")
    print("{:.4f}".format(value_e))
    print("{:.4f}".format(value_s))

    # Compute F(English) and F(Spanish)
    # Print "Q3" followed by their values up to 4 decimal places on two separate lines
    F_English = sum([X[char] * math.log(e[i]) for i, char in enumerate(string.ascii_uppercase)]) + math.log(0.6)
    F_Spanish = sum([X[char] * math.log(s[i]) for i, char in enumerate(string.ascii_uppercase)]) + math.log(0.4)
    print("Q3")
    print("{:.4f}".format(F_English))
    print("{:.4f}".format(F_Spanish))

    # Compute P(Y = English | X)
    # Print "Q4" then this value up to 4 decimal places
    difference = F_Spanish - F_English
    if difference >= 100:
        probability = 0
    elif difference <= -100:
        probability = 1
    else:
        probability = 1 / (1 + math.exp(difference))
    print("Q4")
    print("{:.4f}".format(probability))

if __name__ == "__main__":
    main()