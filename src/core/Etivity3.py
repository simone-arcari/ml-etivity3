# ======================================================================================
# Autore:     Simone Arcari
# Data:       01/05/2025
# Descrizione: 
#   Questo script implementa una classe `DataAnalyzer` in grado di eseguire un'analisi 
#   automatica di dataset forniti in formato CSV o DataFrame. Include rilevamento e 
#   classificazione automatica dei tipi di variabili, preprocessamento dei dati, 
#   analisi esplorativa, clustering (K-Means e gerarchico) e modelli di classificazione 
#   supervisionata (regressione logistica e albero decisionale), con visualizzazioni 
#   grafiche per una migliore interpretazione dei risultati.
# ======================================================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from scipy.cluster.hierarchy import dendrogram, linkage


class DataAnalyzer:
    def __init__(self, data_path, target_variable=None):
        """
        Inizializza l'analizzatore di dati.
        
        Args:
            data_path (str): Percorso al file CSV o DataFrame
            target_variable (str, optional): Nome della variabile target per analisi supervisionate
        """
        if isinstance(data_path, pd.DataFrame):
            self.df = data_path
        else:
            self.df = pd.read_csv(data_path)
        
        self.target = target_variable
        self.features = [col for col in self.df.columns if col != target_variable]
        self.var_types = {}
        self._detect_variable_types()
        
    def _detect_variable_types(self):
        """
        Rileva automaticamente i tipi di variabili nel dataset:
        - Numeriche continue
        - Numeriche discrete
        - Categoriche nominali
        - Categoriche ordinali
        """
        for col in self.df.columns:
            # Controllo per variabili categoriche
            if self.df[col].dtype == 'object' or self.df[col].nunique() < 10:
                if self.df[col].nunique() < 5:
                    self.var_types[col] = 'categorical_nominal'
                else:
                    self.var_types[col] = 'categorical_ordinal'
            else:
                # Controllo per variabili numeriche
                if np.issubdtype(self.df[col].dtype, np.integer):
                    if self.df[col].nunique() < 20:
                        self.var_types[col] = 'numeric_discrete'
                    else:
                        self.var_types[col] = 'numeric_continuous'
                else:
                    self.var_types[col] = 'numeric_continuous'
                    
        print("Tipi di variabili rilevati:")
        for var, typ in self.var_types.items():
            print(f"- {var}: {typ}")
    
    def preprocess_data(self):
        """
        Preprocessa i dati in base ai tipi di variabili rilevati.
        """
        self.encoded_df = self.df.copy()
        
        # Codifica variabili categoriche
        self.encoders = {}
        for col in self.encoded_df.columns:
            if self.var_types[col].startswith('categorical'):
                le = LabelEncoder()
                self.encoded_df[col] = le.fit_transform(self.encoded_df[col])
                self.encoders[col] = le
        
        # Normalizzazione variabili numeriche
        if any(t in ['numeric_continuous', 'numeric_discrete'] for t in self.var_types.values()):
            numeric_cols = [col for col, typ in self.var_types.items() 
                           if typ in ['numeric_continuous', 'numeric_discrete']]
            scaler = StandardScaler()
            self.encoded_df[numeric_cols] = scaler.fit_transform(self.encoded_df[numeric_cols])
    
    def analyze(self):
        """
        Esegue l'analisi automatica dei dati in base ai tipi rilevati.
        """
        print("\n=== ANALISI AUTOMATICA IN CORSO ===")
        
        # 1. Analisi esplorativa di base
        self._basic_exploratory_analysis()
        
        # 2. Analisi non supervisionata (se nessuna target variable)
        if not self.target:
            self._unsupervised_analysis()
        else:
            # 3. Analisi supervisionata (se target variable specificata)
            self._supervised_analysis()
            
        print("\n=== ANALISI COMPLETATA ===")
    
    def _basic_exploratory_analysis(self):
        """Analisi esplorativa di base"""
        print("\n1. ANALISI ESPLORATIVA:")
        
        # Statistiche descrittive
        print("\nStatistiche descrittive:")
        print(self.df.describe(include='all'))
        
        # Matrice di correlazione (solo per variabili numeriche)
        numeric_cols = [col for col, typ in self.var_types.items() 
                       if typ in ['numeric_continuous', 'numeric_discrete']]
        if len(numeric_cols) > 1:
            print("\nMatrice di correlazione:")
            corr_matrix = self.df[numeric_cols].corr()
            print(corr_matrix)
            
            # Heatmap della correlazione
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
            plt.title("Matrice di correlazione")
            plt.show()
        
        # Analisi delle frequenze per variabili categoriche
        categorical_cols = [col for col, typ in self.var_types.items() 
                           if typ.startswith('categorical')]
        for col in categorical_cols:
            print(f"\nDistribuzione di frequenza per {col}:")
            print(self.df[col].value_counts())
            
            # Grafico a barre
            plt.figure(figsize=(8, 5))
            self.df[col].value_counts().plot(kind='bar')
            plt.title(f"Distribuzione di {col}")
            plt.show()
    
    def _unsupervised_analysis(self):
        """Esegue analisi non supervisionata automatica basata sui tipi di variabili.
    
            Strategie applicate:
            1. K-Means clustering con selezione ottimale del k usando silhouette score
            2. Clustering gerarchico con identificazione automatica del punto di taglio
            3. Visualizzazione comparativa dei risultati
            
            Output:
            - Grafico silhouette scores per K-Means
            - Scatter plot dei cluster K-Means
            - Dendrogramma con taglio ottimale
            - Scatter plot dei cluster gerarchici
        """
        print("\n2. ANALISI NON SUPERVISIONATA:")
        
        # Seleziona solo variabili numeriche
        numeric_cols = [col for col, typ in self.var_types.items() 
                    if typ in ['numeric_continuous', 'numeric_discrete']]
        
        if len(numeric_cols) < 2:
            print("Attenzione: Almeno 2 variabili numeriche richieste per il clustering.")
            return
        
        X = self.encoded_df[numeric_cols]
        
        # 1. K-MEANS CLUSTERING
        print("\n[1/2] CLUSTERING K-MEANS")
        
        MIN_CLUSTERS = 2
        MAX_CLUSTERS = min(10, len(X)//2)
        
        silhouette_scores = []
        k_values = range(MIN_CLUSTERS, MAX_CLUSTERS+1)
        
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            silhouette_scores.append(silhouette_score(X, labels))
            print(f" - k={k}: Silhouette={silhouette_scores[-1]:.3f}")

        best_k = k_values[np.argmax(silhouette_scores)]
        print(f"\nNumero ottimale di cluster: {best_k} (silhouette={max(silhouette_scores):.3f})")
        
        final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        self.df['kmeans_cluster'] = final_kmeans.fit_predict(X)
        
        # 2. HIERARCHICAL CLUSTERING
        if len(self.df) < 1000:
            print("\n[2/2] CLUSTERING GERARCHICO")
            
            Z = linkage(X, method='ward')
            
            # Identificazione punto di taglio
            last_merges = Z[-10:, 2]
            merge_diffs = np.diff(last_merges[::-1])
            relative_diffs = merge_diffs[:-1]/merge_diffs[1:]
            optimal_idx = np.argmax(relative_diffs) + 1
            cutoff = last_merges[::-1][optimal_idx]
            n_clusters = optimal_idx + 1
            
            print(f"Taglio ottimale a distanza: {cutoff:.2f} ({n_clusters} cluster)")
            
            # Plot dendrogramma
            plt.figure(figsize=(15, 6))
            dendrogram(Z,
                    truncate_mode='lastp',
                    p=12,
                    color_threshold=cutoff)
            
            plt.axhline(y=cutoff, color='r', linestyle='--')
            plt.title(f"Dendrogramma (Taglio a {cutoff:.2f}, {n_clusters} cluster)")
            plt.show()
            
            agg = AgglomerativeClustering(n_clusters=n_clusters, 
                                        metric='euclidean',
                                        linkage='ward')
            self.df['hierarchical_cluster'] = agg.fit_predict(X)
            
            # Visualizzazione comparativa
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
            
            sns.scatterplot(data=self.df, 
                            x=numeric_cols[0], 
                            y=numeric_cols[1],
                            hue='kmeans_cluster',
                            palette='viridis',
                            ax=ax1)
            ax1.set_title(f"K-Means (k={best_k})")
            
            sns.scatterplot(data=self.df, 
                            x=numeric_cols[0], 
                            y=numeric_cols[1],
                            hue='hierarchical_cluster',
                            palette='viridis',
                            ax=ax2)
            ax2.set_title(f"Gerarchico (k={n_clusters})")
            
            plt.tight_layout()
            plt.show()
    
    def _supervised_analysis(self):
        """Analisi supervisionata"""
        print(f"\n3. ANALISI SUPERVISIONATA (target: {self.target}):")
        
        # Preparazione dati
        X = self.encoded_df.drop(columns=[self.target])
        y = self.encoded_df[self.target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Tipo di problema (classificazione o regressione)
        problem_type = 'classification' if self.var_types[self.target].startswith('categorical') else 'regression'
        
        if problem_type == 'classification':
            print("\nProblema di classificazione rilevato")
            
            # Regressione logistica
            print("\nRegressione Logistica:")
            lr = LogisticRegression(max_iter=1000)
            lr.fit(X_train, y_train)
            y_pred = lr.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Accuracy: {accuracy:.3f}")
            
            # Albero decisionale
            print("\nAlbero Decisionale:")
            dt = DecisionTreeClassifier(max_depth=3)
            dt.fit(X_train, y_train)
            y_pred = dt.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Accuracy: {accuracy:.3f}")
            
            # Feature importance
            if hasattr(dt, 'feature_importances_'):
                print("\nImportanza delle feature (Albero Decisionale):")
                for col, importance in zip(X.columns, dt.feature_importances_):
                    print(f"- {col}: {importance:.3f}")
                
                plt.figure(figsize=(8, 5))
                plt.barh(X.columns, dt.feature_importances_)
                plt.title("Importanza delle Feature")
                plt.show()
        
        else:
            print("\nProblema di regressione rilevato")
        
            # Regressione Lineare
            print("\nRegressione Lineare:")
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            y_pred = lr.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            print(f"Mean Squared Error: {mse:.3f}")
            print(f"R² (coefficiente di determinazione): {r2:.3f}")
            
            # Albero Regressore
            print("\nAlbero Regressore:")
            dt = DecisionTreeRegressor(max_depth=3)
            dt.fit(X_train, y_train)
            y_pred = dt.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            print(f"Mean Squared Error: {mse:.3f}")
            print(f"R² (coefficiente di determinazione): {r2:.3f}")
            
            # Importanza delle feature
            if hasattr(dt, 'feature_importances_'):
                print("\nImportanza delle feature (Albero Regressore):")
                for col, importance in zip(X.columns, dt.feature_importances_):
                    print(f"- {col}: {importance:.3f}")
                
                plt.figure(figsize=(8, 5))
                plt.barh(X.columns, dt.feature_importances_)
                plt.title("Importanza delle Feature (Regressore)")
                plt.show()
    
    def _analyze_categorical_relationship(self, var1, var2):
        """Analizza la relazione tra due variabili categoriche"""
        contingency_table = pd.crosstab(self.df[var1], self.df[var2])
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        
        print(f"\nTest Chi-Quadro tra {var1} e {var2}:")
        print(f"Chi2: {chi2:.3f}, p-value: {p:.4f}")
        print(f"Gradi di libertà: {dof}")
        
        alpha = 0.05
        if p < alpha:
            print(f"Risultato: Dipendenza significativa (p < {alpha})")
        else:
            print(f"Risultato: Nessuna evidenza di dipendenza (p >= {alpha})")
        
        # Visualizzazione
        plt.figure(figsize=(10, 6))
        sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Tabella di contingenza: {var1} vs {var2}")
        plt.show()
