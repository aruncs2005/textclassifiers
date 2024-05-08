import pandas as pd
import re
from sklearn.model_selection import train_test_split

def remove_urls(text, replacement_text=""):
    # Define a regex pattern to match URLs
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
 
    # Use the sub() method to replace URLs with the specified replacement text
    text_without_urls = url_pattern.sub(replacement_text, text)
 
    return text_without_urls

def main():
    mapping = {"alzheimer":1,"parkinson":2,"cancer":3,"diabetes":4}
    file = open("../data/illness/data.txt",encoding="utf-8")

    Lines = file.readlines()
    dataset = []
    for line in Lines:
        input = line.split("\t")
        item = {}
        item["text"] = remove_urls(input[2]).replace("\"","")
        item["label"] = mapping[input[0]]
        dataset.append(item)

    dataDf = pd.DataFrame(dataset)

    

    #print(dataDf.head())
    traindf, testdf = train_test_split(dataDf, test_size=0.2, random_state=42, shuffle=True)
    print(f"Number of samples in train dataset {len(traindf.index)}")
    print(f"Number of samples in test dataset {len(testdf.index)}")

    traindf.to_csv("../data/illness/train.csv",index=False)
    testdf.to_csv("../data/illness/test.csv",index=False)
        


if __name__ == "__main__":
    main()