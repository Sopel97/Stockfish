import chess
import chess.svg
import sys

from PyQt5 import QtCore
from PyQt5.QtCore import QByteArray
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtWidgets import QApplication, QWidget, QListWidget, QListWidgetItem, QFormLayout, QGridLayout, QVBoxLayout

filename = sys.argv[1]

class EntryWidget(QWidget):
    def __init__(self, fen, data):
        super().__init__()

        self.table = QListWidget()
        for k, v in sorted(data.items(), key=lambda x:x[0]):
            self.table.addItem(QListWidgetItem('{} = {}'.format(k, v)))
        self.table.setMaximumSize(200, 100)
        svg_raw = chess.svg.board(chess.Board(fen), size=200).encode('utf-8')
        self.svg = QSvgWidget()
        self.svg.load(QByteArray(svg_raw))
        self.svg.setMaximumSize(200, 200)
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.svg)
        self.layout.addWidget(self.table)
        self.setLayout(self.layout)

class Entry:
    def __init__(self, line):
        fen, meta = [part.strip() for part in line.split(';')]

        self.data = dict()
        self.fen = fen

        kvpairs = meta.replace('=', ' ').replace(':', '').split(' ')
        for i in range(0, len(kvpairs), 2):
            key, value = kvpairs[i], kvpairs[i+1]
            self.data[key] = value

    def get_widget(self):
        return EntryWidget(self.fen, self.data)

class MainWindow(QWidget):
    def __init__(self, entries):
        super().__init__()

        self.entries = entries
        self.w = 8
        self.h = 3
        self.offset = 0
        self.layout = QGridLayout()

        self.setGeometry(0, 0, 200*self.w, 300*self.h)

        self.update_boards()

    def paintEvent(self, event):
        pass

    def clear_layout(self):
        for i in reversed(range(self.layout.count())):
            self.layout.itemAt(i).widget().deleteLater()

    def update_boards(self):
        self.clear_layout()
        for x in range(self.w):
            for y in range(self.h):
                idx = self.offset + y*8+x
                if idx >= 0 and idx < len(self.entries):
                    self.layout.addWidget(self.entries[idx].get_widget(), y, x)
                else:
                    self.layout.addWidget(QListWidget(), y, x)
        self.setLayout(self.layout)

    def keyPressEvent(self, e):
        if e.key() == QtCore.Qt.Key_Left:
            if self.offset > 0:
                self.offset -= self.w * self.h
        elif e.key() == QtCore.Qt.Key_Right:
            if self.offset + self.w * self.h < len(self.entries):
                self.offset += self.w * self.h
        else:
            return

        self.update_boards()

if __name__ == "__main__":
    entries = []

    with open(filename, 'r') as file:
        for line in file:
            entries.append(Entry(line))

    app = QApplication([])
    window = MainWindow(entries)
    window.show()
    app.exec()

