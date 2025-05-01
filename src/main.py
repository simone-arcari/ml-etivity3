import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

import signal
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication
from gui.Etivity3Window import DatasetSelectorWindow

def handle_sigint(signum, frame):
    """Gestisce il segnale SIGINT per terminare l'applicazione"""
    print("Intercettato Ctrl+C, chiusura dell'applicazione...")
    QApplication.quit()

if __name__ == "__main__":
    print(
    "\n     __  __ _          ______ _   _       _ _         ____  "
    "\n    |  \/  | |        |  ____| | (_)     (_) |       |___ \ "
    "\n    | \  / | |  ______| |__  | |_ ___   ___| |_ _   _  __) |"
    "\n    | |\/| | | |______|  __| | __| \ \ / / | __| | | ||__ < "
    "\n    | |  | | |____    | |____| |_| |\ V /| | |_| |_| |___) |"
    "\n    |_|  |_|______|   |______|\__|_| \_/ |_|\__|\__, |____/ "
    "\n                                                 __/ |      "
    "\n                                                |___/       "
    )



    # configura il gestore per il segnale SIGINT
    signal.signal(signal.SIGINT, handle_sigint)

    # Avvia applicazione Qt
    app = QApplication(sys.argv)

    # Per rilevare SIGINT
    timer = QTimer()
    timer.start(100)
    timer.timeout.connect(lambda: None) # tiene vivo il loop di eventi per rilevare SIGINT

    # Lancia finestra grafica
    window = DatasetSelectorWindow()
    window.show()
    sys.exit(app.exec_())
