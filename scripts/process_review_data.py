import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    trainData = pd.read_csv("../data/patient_reviews/reviews_small.csv")

    trainData = trainData[["rating","comment"]]
    print(len(trainData.index))
    trainData = trainData.rename({'rating': 'label', 'comment': 'text'}, axis=1)

    trainData["text_length"] = trainData["text"].str.split(" ")
    trainData = trainData[trainData['text_length'].map(len) < 100]
    trainData = trainData.drop('text_length', axis=1)
    
    columns_titles = ["text","label"]
    trainData=trainData.reindex(columns=columns_titles)


    sample = 1500
    col_name = "label"

    probs = trainData[col_name].map(trainData[col_name].value_counts())
    trainData = trainData.sample(n=sample, weights=probs)

    print(len(trainData.index))
    
    traindf, testdf = train_test_split(trainData, test_size=0.2, random_state=42, shuffle=True)

    print(f"Number of samples in train dataset {len(traindf.index)}")
    print(f"Number of samples in test dataset {len(testdf.index)}")

    traindf.to_csv("../data/patient_reviews/train.csv",index=False)
    testdf.to_csv("../data/patient_reviews/test.csv",index=False)
    


if __name__ == "__main__":
    main()