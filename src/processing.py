import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def processing(Data):
    color_mapped = {'Blue White': 3, 'Blue white': 3, 'Blue-White': 3, 'Blue-white': 3,
                    'white': 5, 'Whitish': 5, 'White': 5,
                    'Yellowish White': 4, 'yellowish': 4, 'Yellowish': 4, 'White-Yellow': 4, 'yellow-white': 4,
                    'Pale yellow orange': 6, 'Orange-Red': 6, 'Orange': 6,
                    'Red': 1,
                    'Blue': 2}

    Data['Color'] = Data['Color'].map(color_mapped)

    spectral_class_mapped = {'M': 1, 'B': 2, 'O': 3, 'A': 4, 'F': 5, 'K': 6, 'K': 7, 'G': 8}

    Data['Spectral_Class'] = Data['Spectral_Class'].map(spectral_class_mapped)

    numerical = Data[['Temperature', 'L', 'R', 'A_M']]
    categorical = Data[['Color', 'Spectral_Class']]

    scaler = MinMaxScaler()
    standardized_numerical = pd.DataFrame(scaler.fit_transform(numerical),
                                          columns=numerical.columns, index=numerical.index)
    standardized_Data = pd.concat([standardized_numerical, categorical], axis=1)
    return standardized_Data
