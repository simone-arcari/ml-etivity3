# ======================================================================================
# Autore:     Simone Arcari
# Data:       01/05/2025
#
# Descrizione:
#   Questo script implementa una finestra principale in PyQt5 che consente all'utente 
#   di selezionare un dataset da un elenco e una variabile target da analizzare. 
#   L'utente può anche analizzare la relazione tra due variabili del dataset selezionato 
#   utilizzando un test di statistica (Chi-Quadro). I risultati dell'analisi vengono 
#   visualizzati direttamente nell'interfaccia grafica.
#
#   - La finestra principale consente la selezione di un dataset e la variabile target.
#   - Fornisce un pulsante per avviare l'analisi sul dataset selezionato.
#       <> [variabile target selezionata]      --> ANALISI SUPERVISIONATA
#       <> [variabile target NON selezionata]  --> ANALISI NON SUPERVISIONATA
#   - Un secondo pulsante permette di analizzare la relazione tra due variabili tramite
#     un dialogo di selezione (Test Chi-Quadro).
# ======================================================================================

import sys
from PyQt5.QtWidgets import (
    QDialog, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QPushButton, QTextEdit
)

import pandas as pd
from core.Etivity3 import DataAnalyzer

from sklearn.datasets import (
    load_iris,
    load_wine,
    load_breast_cancer,
    load_diabetes,
    fetch_california_housing,
)

import io
from contextlib import redirect_stdout
from core.Tee import Tee

# Mappatura dei loader sklearn
DATASETS = {
    "Iris": load_iris,
    "Wine": load_wine,
    "Breast Cancer": load_breast_cancer,
    "Diabetes": load_diabetes,
    "California Housing": fetch_california_housing,
}

class DatasetSelectorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Selettore Dataset e Target")
        self.resize(600, 400)

        # Widget centrale e layout
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Primo layout: selezione dataset
        dataset_layout = QHBoxLayout()
        dataset_label = QLabel("Dataset:")
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(DATASETS.keys())
        self.dataset_combo.currentTextChanged.connect(self.update_target_combo)
        dataset_layout.addWidget(dataset_label)
        dataset_layout.addWidget(self.dataset_combo)
        layout.addLayout(dataset_layout)

        # Secondo layout: selezione target
        target_layout = QHBoxLayout()
        target_label = QLabel("Variabile target:")
        self.target_combo = QComboBox()
        target_layout.addWidget(target_label)
        target_layout.addWidget(self.target_combo)
        layout.addLayout(target_layout)

        # Output
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        layout.addWidget(self.output)

        # Bottone
        self.run_button = QPushButton("Analizza Dataset")
        self.run_button.clicked.connect(self.on_run_clicked)
        layout.addWidget(self.run_button)

        # Bottone Analizza relazione tra due variabili (Test Chi-Quadro)
        self.relation_button = QPushButton("Analizza relazione tra due variabili")
        self.relation_button.clicked.connect(self.on_analyze_relationship_clicked)
        layout.addWidget(self.relation_button)

        # Caricamento iniziale dei target
        self.update_target_combo(self.dataset_combo.currentText())

    def update_target_combo(self, selected_dataset):
        """Aggiorna le opzioni del menu target in base al dataset scelto"""
        loader = DATASETS[selected_dataset]
        data = loader()
        df = pd.DataFrame(data.data, columns=data.feature_names)

        # Aggiunge la colonna target (stringhe se disponibili)
        if hasattr(data, 'target_names') and len(data.target_names) > 0:
            df['target'] = [data.target_names[i] for i in data.target]
        else:
            df['target'] = data.target

        # Pulisce e ripopola il combo delle target
        self.target_combo.clear()
        self.target_combo.addItem("None")
        for col in df.columns:
            self.target_combo.addItem(col)

        self.current_df = df

    def on_run_clicked(self):
        dataset_name = self.dataset_combo.currentText()
        target_selected = self.target_combo.currentText()
        if target_selected == "None":
            target_selected = None

        # Output
        self.output.setText(f"Dataset selezionato: {dataset_name}\n"
                            f"Target selezionata: {target_selected}")
        
        buffer = io.StringIO()
        tee = Tee(sys.stdout, buffer)

        with redirect_stdout(tee):  # stampa sia a terminale che in buffer
            analyzer = DataAnalyzer(self.current_df, target_selected)
            analyzer.preprocess_data()
            analyzer.analyze()
        
        self.output.setText(buffer.getvalue())

    def on_analyze_relationship_clicked(self):
        if self.current_df is None:
            self.output.setText("Errore: nessun dataset caricato.")
            return

        columns = list(self.current_df.columns)

        dialog = VariableSelectorDialog(columns, self)
        if dialog.exec_() == QDialog.Accepted:
            var1, var2 = dialog.selected_variables()

            buffer = io.StringIO()
            tee = Tee(sys.stdout, buffer)

            with redirect_stdout(tee):
                analyzer = DataAnalyzer(self.current_df)
                try:
                    analyzer._analyze_categorical_relationship(var1, var2)
                except Exception as e:
                    print(f"Errore durante l'analisi: {e}")

            self.output.setText(buffer.getvalue())

class VariableSelectorDialog(QDialog):
    def __init__(self, columns, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Seleziona due variabili")
        self.resize(300, 100)

        layout = QVBoxLayout(self)

        # Primo combo
        hbox1 = QHBoxLayout()
        hbox1.addWidget(QLabel("Variabile 1:"))
        self.combo1 = QComboBox()
        self.combo1.addItems(columns)
        hbox1.addWidget(self.combo1)
        layout.addLayout(hbox1)

        # Secondo combo
        hbox2 = QHBoxLayout()
        hbox2.addWidget(QLabel("Variabile 2:"))
        self.combo2 = QComboBox()
        self.combo2.addItems(columns)
        hbox2.addWidget(self.combo2)
        layout.addLayout(hbox2)

        # Pulsanti OK/Cancel
        buttons = QHBoxLayout()
        ok_btn = QPushButton("Conferma")
        cancel_btn = QPushButton("Annulla")
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        buttons.addWidget(ok_btn)
        buttons.addWidget(cancel_btn)
        layout.addLayout(buttons)

    def selected_variables(self):
        return self.combo1.currentText(), self.combo2.currentText()
