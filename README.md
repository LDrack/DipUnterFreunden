# DIP Unter Freunden

### Präsentation: 25.1.2022

#### Annotieren:
https://github.com/tzutalin/labelImg

#### Augmentieren:
https://github.com/BendiXB/AugmentLabelIMG
Bilder UND Annotationen werden um 90, 180 und 270 Grad gedreht UND gespiegelt -> Verachtfachung des Datensatzes

Der Plan:
- Bilder mit Labelimg annotieren
- Bilder mit AugmentLabelIMG augmentieren
- Mit den augmentierten Daten unser Modell trainieren (auf Basis vom KIVideo-Skript)
- Skript schreiben, das einen Ordner mit Bildern einliest und auf jedes die KI anwendet (auf Basis vom KIVideo-SKript)
  - Einzeichnen der gefundenen Klassen
  - Bildanzeige und Tastensteuerung zum Weiterschalten
  - Ergebnisse mitzählen und am Ende abspeichern zum Auswerten
- 1er griang
