import re


def convert_roman_to_num(roman_num):
    roman_num = roman_num.upper()
    roman_to_num = {'I': 10, 'V': 50, 'X': 100, 'L': 500, 'C': 1000, 'D': 5000, 'M': 10000}
    alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    num = 0
    for i in range(len(roman_num)):
        romain_char = roman_num[i]
        if romain_char not in roman_to_num.keys():
            num += alphabet.index(romain_char) + 1
            continue
        if i > 0 and roman_to_num[romain_char] > roman_to_num[roman_num[i - 1]]:
            num += roman_to_num[romain_char] - 2 * roman_to_num[roman_num[i - 1]]
        else:
            num += roman_to_num[romain_char]
    return num


def extract_input(input_string):
    # Define a regular expression pattern to match the content inside parentheses
    pattern = r"\((.*?)\)"

    # Use re.search to find the first match in the input string
    match = re.search(pattern, input_string)

    # Check if a match is found
    if match:
        # Extract and return the content inside parentheses
        return match.group(1)
    else:
        # Return None if no match is found
        return None