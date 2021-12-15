import os

if __name__ == "__main__":
    # print("loading data")
    # os.system(f"python scripts/preprocessing/create_data.py -output data/raw")
    # print("preprocessing data")
    # os.system(
    #    f"python scripts/preprocessing/preprocess.py -input data/raw -output data/preprocessed")
    print("classifying data")
    os.system(f"python scripts/classification/classify.py -input data/preprocessed")
