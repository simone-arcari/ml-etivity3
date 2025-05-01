from sklearn.datasets import load_iris
from Etivity3 import DataAnalyzer
import pandas as pd

if __name__ == "__main__":
    print(
    "\n      ______ _   _       _ _           ____  "
    "\n     |  ____| | (_)     (_) |         |___ \ "
    "\n     | |__  | |_ ___   ___| |_ _   _    __) |"
    "\n     |  __| | __| \ \ / / | __| | | |  |__ < "
    "\n     | |____| |_| |\ V /| | |_| |_| |  ___) |"
    "\n     |______|\__|_| \_/ |_|\__|\__, | |____/ "
    "\n                                __/ |        "
    "\n                               |___/         "
    )

    # Caricamento dati (esempio con dataset Iris)
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target_names[iris.target]
    
    # Creazione analizzatore
    #analyzer = DataAnalyzer(df, target_variable='species')
    analyzer = DataAnalyzer(df)
    # Preprocessing
    analyzer.preprocess_data()
    
    # Analisi automatica
    analyzer.analyze()
    
    # Analisi specifica tra due variabili categoriche
    if 'species' in df.columns and 'petal length (cm)' in df.columns:
        # Per questo esempio, convertiamo una variabile numerica in categorica
        df['petal_length_cat'] = pd.cut(df['petal length (cm)'], bins=3, 
                                       labels=['short', 'medium', 'long'])
        analyzer._analyze_categorical_relationship('species', 'petal_length_cat')