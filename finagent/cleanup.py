import re

def remove_think_tags(filename):
    # Open the file for reading and reading the content
    with open(filename, 'r') as file:
        content = file.read()
    
    # Use regular expression to remove everything between <think> and </think>
    cleaned_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    
    # Optionally, write the cleaned content back to the file or a new file
    with open('cleaned_file_MSFTv2.txt', 'w') as file:
        file.write(cleaned_content)

# Example usage
remove_think_tags('/Users/subrat_roy/Documents/projects/finagent/MSFT_analysis.txt')
