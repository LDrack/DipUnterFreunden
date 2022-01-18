# DIP Unter Freunden

### Präsentation: 25.1.2022

#### Annotieren und Augmentieren
https://roboflow.com

#### KI-Framework
YoloV4

#### Der Plan:
- Bilder mit Roboflow annotieren und augmentieren
- Mit den augmentierten Daten unser Modell trainieren (auf Basis vom KIVideo-Skript)
  - Läuft auf Google Colab in einem Jupyter-Notebook
  - Notebook wurde vom Roboflow-Blog adaptiert: https://blog.roboflow.com/training-yolov4-on-a-custom-dataset/
- Skript schreiben, das einen Ordner mit Bildern einliest und auf jedes die KI anwendet (auf Basis vom KIVideo-SKript)
  - Einzeichnen der gefundenen Klassen
  - Bildanzeige und Tastensteuerung zum Weiterschalten
  - Ergebnisse mitzählen und am Ende abspeichern zum Auswerten
- 1er griang

#### Dokumentation der Arbeit
1. Datensatz erzeugen
  - Roboflow-Account erstellt (Free version)
  - Bilder hochgeladen und 8 Klassen (Normal, NoHat, ...) händisch annotiert
2. Bild-Augmentierung
  - Definition: Erzeugen aus neuen Trainingsdaten auf Basis der vorhandenen durch Verzerren, Drehen etc. Da die Daten schon augmentiert sind, werden auch die Bounding Boxes auch entsprechend gedreht!
  - Augmentierung in Roboflow einfaches Auswählen der verfügbaren Optionen - wir verwenden:
    - Preprocessing:
      - Stretch to 208x208 Bildgröße
      - Auto-Adjust Contrast
      - Filter Null auf 5%, damit auch Fotos ohne Indy im Set vorkommen
    - Augmentation:
      - 90° Rotate (dreht Bilder ganz um 90, 180, 270°)
      - Rotation (rotiert das Bild um Werte zwischen -15 und 15°)
      - Shear (verzerrt Bild um +/- 8° horizontal und vertikal)
  - Ergebnis:
    - Aus 123 ursprünglichen Bilder wurden 299 augmentierte Trainingsbilder erzeugt (Berechnung siehe Roboflow)
    - Mit der Free-Version sind nur 3 Varianten pro Bild erzeugbar - je nachdem wie viel man zahlt kann man bis 50 Varianten pro Bild erzeugen
  - Datensatz kann nun aus Roboflow exportiert werden
3. Modell trainieren
  - Roboflow bietet Blogbeiträge und fertige Jupyter Notebooks zum Trainieren: https://blog.roboflow.com/training-yolov4-on-a-custom-dataset/
  - Wir haben das Notebook minimalst anpassen müssen:
    - Link zu unserem eigenen Datensatz eingefügt (exportiert aus Roboflow)
    - Zu finden im Repo im Ordner `/colab/`
  - (Modell #1) Trainingsergebnis nach 2000 Epochen: Genauigkeit von 84,5 % (mAP)
4. Applikation
  - YoloV4-Modell wird mit OpenCV-Backend geladen (cv2.dnn.readNetFromDarknet)
  - 3 Dateien werden gebraucht:
    - .weights (fertiges Modell)
    - .names (Namen und Reihenfolge der Labels)
    - .cfg (Beschreibung der Layers, Modellparameter)
  - Applikation wurde adaptiert von https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/
    - Testbilder werden aus Ordner geladen und durchiteriert
    - Jedes Bild wird durch die KI gelassen, diese liefert erkannte Bounding Boxes + Confidence
    - Boxen werden eingezeichnet und dem Benutzer angezeigt (cv.imshow, Steuerung per Tastendruck möglich)
    - Ergebnisse werden zur späteren Ansicht gespeichert
5. Resultate
  - Mit dem gegebenen Datensatz liefert das Modell gute Ergebnisse (TODO: Metriken beschreiben - visuelle Prüfung, Precision-Werte...)
  - Skript kann als Startpunkt für andere Projekte verwendet werden

