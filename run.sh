#!/bin/bash

# Nome dell'ambiente virtuale
VENV_DIR="venv-etivity3"

# Nome del file del programma Python
PROGRAM="src/main.py"

# Nome del file delle dipendenze
REQUIREMENTS="requirements.txt"

# Controlla se il file del programma esiste
if [ ! -f "$PROGRAM" ]; then
    echo "Errore: Il file $PROGRAM non esiste nella directory corrente."
    exit 1
fi

# Crea l'ambiente virtuale solo se non esiste già
if [ ! -d "$VENV_DIR" ]; then
    echo "Creazione dell'ambiente virtuale..."
    python3 -m venv "$VENV_DIR"
fi

# Attiva l'ambiente virtuale
echo "Attivazione dell'ambiente virtuale..."
source "$VENV_DIR/bin/activate"

# Aggiorna pip
echo "Aggiornamento di pip..."
pip install --upgrade pip

# Controlla se il file delle dipendenze esiste
if [ -f "$REQUIREMENTS" ]; then
    # installa dipendenze
    echo "Installazione delle dipendenze da $REQUIREMENTS..."
    pip install -r "$REQUIREMENTS"
else
    echo "Warning: Il file $REQUIREMENTS non esiste, salto l'installazione delle dipendenze."
fi

# Avvia il programma
echo "Avvio del programma..."
python "$PROGRAM"

# Disattiva l'ambiente virtuale solo se è attivo
if type deactivate &>/dev/null; then
  echo "Disattivazione dell'ambiente virtuale..."
  deactivate
fi
