from imutils.video import VideoStream
from imutils.video import FPS
from imutils.video import FileVideoStream
from scipy.spatial import distance as dist
from imutils import face_utils

import os
import sqlite3
import numpy as np
import argparse
import imutils
import cv2
import dlib
import cv2
import face_recognition
import subprocess
import numpy as np
import argparse
import imutils
import time
import sys




def count_face(capture, nbFace):
    faceCascade = cv2.CascadeClassifier('add-files/haarcascade/haarcascade_frontalface_default.xml')
    while(True):
        ret, frame = capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #On détecte les visages
        faces = faceCascade.detectMultiScale(gray)
        numb_face = len(faces)
        time.sleep(3)
        if numb_face == nbFace:
            return True
        else:
            return False


def os_checker():
    if 'linux' in sys.platform:
        return True
    else:
        return False


"""Installer le paquet v4l2 avce la commande suivante: $ sudo apt-get install v4l-utils"""
def set_cam_index(num):
    nombre = len(webcam())-1
    try:
        num = int(num)
    except Exception as e:
        raise e          
    if 0<=num<=nombre:
        conn = sqlite3.connect('oudjat.db')
        cursor = conn.cursor()
        cursor.execute("""UPDATE admin SET cam_index = ? WHERE id = ? """, (num,'1',))
        conn.commit()
        conn.close()
    else:
        return False


def webcam():
    proc = subprocess.Popen(["ls -l /dev/video*|wc -l"], stdout=subprocess.PIPE, shell=True)
    (nb_cam, err) = proc.communicate()
    a = int(nb_cam)
    proc = subprocess.Popen(["v4l2-ctl --list-devices"], stdout=subprocess.PIPE, shell=True)
    (cmd, err) = proc.communicate()

    cam = []
    for i in cmd.decode('utf-8').split('\n\n')[:-1]:
        cam += [''.join(str(i).split(":")[:-1])]
    return cam

def webcam_list():
    cam1 = webcam()
    cam_list = '\n'+'\n'.join([str(i)+'- '+j for i,j in enumerate(cam1)])+'\n'

    return cam_list



def objet_detect(capture):
    #Initialisation de la classe d'objet à détecter
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "sheep",
        "sofa", "train", "tvmonitor", "knife","phone"]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    # charger notre modèle sérialisé à partir du disque
    net = cv2.dnn.readNetFromCaffe('add-files/MobileNetSSD_deploy.prototxt.txt' ,'add-files/MobileNetSSD_deploy.caffemodel')

    # initialiser le flux vidéo, laisser le capteur de la caméra se réchauffer,
    # et initialiser le compteur de FPS
    fps = FPS().start()

    # Boucle sur les frames de la video
    while True:
        # récupérez l'image du flux vidéo et redimensionnez-le
        # pour avoir une largeur maximale de 600 pixels
        frame = capture.read()
        frame = imutils.resize(frame, width=600)

        # saisir les dimensions du cadre et le convertir en blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
            0.007843, (300, 300), 127.5)

        # passer le blob à travers le réseau de neuronne et obtenir les détections extract
        # prédictions
        net.setInput(blob)
        detections = net.forward()

        # boucle sur les détections
        for i in np.arange(0, detections.shape[2]):
            # extraire la confiance (c'est-à-dire la probabilité) associée à
            # la prédiction
            confidence = detections[0, 0, i, 2]

            # filtrer les détections faibles en veillant à ce que la «confiance» soit
            # plus grand que le minimum de confiance
            if confidence > 0.5:
                # extraire l'index de l'objet de classe de la
                # `detections`, puis calculez les coordonnées (x, y) du
                # cadre de sélection de l'objet
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # dessiner la prédiction sur le cadre
                label = "{}: {:.2f}%".format(CLASSES[idx],
                    confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                    COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

                time.sleep(5)
                if CLASSES[idx] in CLASSES:
                    return False
                
                else:
                    return True
        # afficher la frame de sortie
        cv2.imshow("Frame", frame)

        # mettre à jour le compteur de FPS
        fps.update()

    # arrêter le chronomètre et afficher les informations FPS
    fps.stop()

    # faire un peu de nettoyage
    cv2.destroyAllWindows()
    capture.stop()



def face_recognition(capture):

    #time.sleep(4)

    #On charge une 1ere image légitime qu'on va detecter et reconnaitre dans la capture
    papin_image = face_recognition.load_image_file("image/pa2.jpg")
    papin_face_encoding = face_recognition.face_encodings(papin_image)[0]


    while True:
        # On prend une frame de la vidéo
        ret, frame = capture.read()

        # On convertit l'image de BGR color (utiliser par OpenCV) A  RGB color (utiliser par face_recognition)
        rgb_frame = frame[:, :, ::-1]

        # On localise tous les visages et les visages encodÃ©s dans la frame de la vidÃ©oes
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        #On parcourt chaque visage dans la frame de la vidÃ©o
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            #On vérifie si le visage détecté correspond à celui de l'utilisateur légitime
            matches = face_recognition.compare_faces(known_face_encodings, face_encodings)

            result = False

            # A chaque comparaison on prend le 1er visage reconnu
            if True in matches:
                first_match_index = matches.index(True)
                #name = known_face_names[first_match_index]
                result = True          

    return result



def rapport_aspect_oeil(oeil):
    """Ici nous calculons le rapport d'aspect d'oeil qui nous permettra de détecter les clignements des yeux"""
    # calculer les distances euclidiennes entre les deux ensembles de points de repère des yeux verticaux (x,y)
    A = dist.euclidean(oeil[1], oeil[5])
    B = dist.euclidean(oeil[2], oeil[4])

    # calculer les distances euclidiennes entre l'horizontale, repère visuel (x,y)
    C = dist.euclidean(oeil[0], oeil[3])

    # calculer le rapport d'aspect des yeux
    ear = (A + B) / (2.0 * C)
    return ear


def blink_detection(video_capture):
       
    # definir deux constantes, une pour le rapport d'aspect de l'oeil pour indiquer le clignement
    # puis une seconde constante pour le nombre de frame consecutive avec un rapport d'aspect d'oeil inferieur au seuil
    EAR_SEUIL = 0.30
    EAR_CONSEC_FRAMES = 4

    # Initialisation du compteur des frames et du nombre total de clignement
    COUNTER = 0
    TOTAL = 0

    # Initialisation du detecteur de visage Dlib
    # creation et Chargement du predicteur du point de repere facial
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('add-files/shape_predictor_68_face_landmarks.dat')

    # saisir les index des reperes faciaux pour l'oeil gauche et l'oeil droit, respectivement
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    # demarrer le fil du flux video
    vs = video_capture
    fileStream = False
    # time.sleep(4)

    # boucle sur les images du flux vidéo
    while True:
   
        # recuperation l'image du flux video, redimensionnement
        # et conversion en niveaux de gris les chaines
        frame = vs.read()
        # frame = vs.imutils.resize(frame, width=100)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detecter les visages dans les frames en niveaux de gris
        rects = detector(gray, 0)

        # boucle sur les detections de visage
        for rect in rects:
            # determination des reperes faciaux pour la region du visage, puis
            # conversion du point de repere facial (x, y) en un tableau numPy
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # extraire les coordonnees de l'oeil gauche et droit, puis
            # les coordonnees pour calculer le rapport d'aspect de l'oeil pour les deux yeux
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = rapport_aspect_oeil(leftEye)
            rightEAR = rapport_aspect_oeil(rightEye)

            # moyenne du rapport d'aspect de l'oeil pour les deux yeux
            ear = (leftEAR + rightEAR) / 2.0

            # calculer la coque convexe pour l'oeil gauche et droit, puis visualiser chacun des yeux
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # verifier si le rapport d'aspect de l'oeil est en dessous du seuil du clignement
            # et si c'est le cas, on incremente le compteur de frame clignotant
            if ear < EAR_SEUIL:
                COUNTER += 1

            # sinon, le rapport d'aspect de l'oeil n'est pas inferieur au seuil de clignement
            else:
                # si les yeux etaient fermes pour un nombre suffisant de frame,
                # on incremente le nombre total de clignotements
                if COUNTER >= EAR_CONSEC_FRAMES:
                    TOTAL += 1

                # reinitialiser le compteur des frames de l'oeil
                COUNTER = 0

    # faire un peu de nettoyage
    cv2.destroyAllWindows()
    vs.stop()
    return TOTAL
    


def video_record(capture):
    """Ici nous enrégistrons la vidéo dans un format lisible en live depuis la webcam durant 30 secondes"""

    #Créer le dossier de sauvegarde des videos des attaques s'il n'existe pas sur le disque
    if not os.path.exists('videos_attaques'):
        os.mkdir('videos_attaques')
    # Définir le codec et créer l'objet VideoWritter
    video_format = cv2.VideoWriter_fourcc(*'XVID')
    #video_format = cv2.VideoWriter_fourcc(*'MJPG')
    video_name = time.strftime("%d-%m-%Y_%H:%I:%S")+'.avi'
    video = cv2.VideoWriter('videos_attaques/'+video_name,video_format, 25, (640,480))

    duration = 10
    begin_time = int(time.time())
    elapsed_time = 0 


    while(capture.isOpened() and elapsed_time < duration):
        ret, frame = capture.read()
        if ret == True:
            frame = cv2.flip(frame,0)
            video.write(frame)
        else:
            break
        elapsed_time = int(time.time()) - begin_time

    capture.release()
    video.release()
    cv2.destroyAllWindows()

def video_saver():
    """Ici nous sauvegardons le chemin vers la vidéo, la date et l'heure de son enrégistrement dans la base de données"""
    conn = sqlite3.connect('oudjat.db')
    cursor = conn.cursor()

    global video_name, date, time
    video_name = time.strftime("%d-%m-%Y_%H:%I:%S")+'.avi'
    date = time.strftime("%A %d-%m-%Y")
    time = time.strftime("%H:%I:%S %p")
    cursor.execute("""
        INSERT INTO attaques(videos, dates, heures) VALUES (?, ?, ?)""",('videos_attaques/'+video_name ,date, time))
    
    conn.commit()
    conn.close()
 
  
capture = cv2.VideoCapture(0)
video_record(capture)



