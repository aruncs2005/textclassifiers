import pandas as pd

def main():
    data = pd.read_csv("../data/medabstracts/medical_tc_train.csv")
    data = data.rename({'condition_label': 'label', 'medical_abstract': 'text'}, axis=1)

    print(len(data.index))
    data["text_length"] = data["text"].str.split(" ")
    data = data[data['text_length'].map(len) < 100]
    data = data.drop('text_length', axis=1)
    print(len(data.index))
    columns_titles = ["text","label"]
    data=data.reindex(columns=columns_titles)
    data.to_csv("../data/medabstracts/train.csv",index=False)

    # process test data

    data = pd.read_csv("../data/medabstracts/medical_tc_test.csv")
    data = data.rename({'condition_label': 'label', 'medical_abstract': 'text'}, axis=1)

    print(len(data.index))
    print(data['label'].unique())
    data["text_length"] = data["text"].str.split(" ")
    data = data[data['text_length'].map(len) < 100]
    data = data.drop('text_length', axis=1)
    print(len(data.index))
    columns_titles = ["text","label"]
    data=data.reindex(columns=columns_titles)
    data.to_csv("../data/medabstracts/test.csv",index=False)


if __name__ == "__main__":
    main()