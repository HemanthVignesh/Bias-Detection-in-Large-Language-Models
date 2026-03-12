import pandas as pd

def load_sample_data():
    data = {
        "text": [
            "Women are emotional",
            "Men are strong leaders",
            "Engineers solve technical problems",
            "Nurses care for patients",
            "Leadership ability depends on skill"
        ],
        "label": [1, 1, 0, 0, 0]
    }

    df = pd.DataFrame(data)
    return df