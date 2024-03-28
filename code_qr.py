import cv2
import numpy as np
import random
import qrcode

def distanceEuclidienne(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def distanceAbs(p1, p2):
    return (np.abs(p1[0] - p2[0]) + np.abs(p1[1] - p2[1]))


def distanceInf(p1, p2):
    return max(np.abs(p1[0] - p2[0]), np.abs(p1[1] - p2[1]))

def voronoi_diagramme(nb_germes,width_,height_,method=0):

    # nb germes, fonction de distance renseigné par l'utilisateur
    nbgermes = nb_germes

    if method == 0:
        dist_function = distanceEuclidienne
    elif method == 1:
        dist_function = distanceAbs
    elif method == 2:
        dist_function = distanceInf

    # va contenir les couleurs des pixels
    #ensemble_pixel_couleur = []
    # va contenir les pixels
    ensemble_pixel = []

    width = width_
    height = height_

    # Initialise les pixels:
    # pixel random compris dans l'image
    for i in range(0, nbgermes):
        pixel = [random.uniform(0, width - 1).__round__(), random.uniform(0, height - 1).__round__()]
        ensemble_pixel.append(pixel)

    # stocke le diagramme de voronoi
    #diagramme_voronoi = [[] for _ in range(nb_germes)]
    diagramme_voronoi = []

    # Parcours de l'image
    for x in range(0, width):
        diagramme_ligne = []
        for y in range(0, height):
            dist = []

            # calcule et stocke la distance du pixel courant au ensemble de pixel
            for pix in ensemble_pixel:
                dist.append(dist_function((x, y), pix))

            # stocke l'index du pixel dans l'ensemble le + proche du pixel courant
            min_dist = dist.index(min(dist))

            diagramme_ligne.append(min_dist)
        diagramme_voronoi.append(diagramme_ligne)

    return diagramme_voronoi


def retour_cle(k,A,B):
    if k in A:
        return 0
    else:
        return 1

def cle(N):
    A = []
    # pairs
    for i in range(2, N + 1, 2):
        A.append(i)
    # impairs
    B = []
    for i in range(1, N + 1, 2):
        B.append(i)

    cle_tab = []

    for i in range(0, N):
        code_retour = retour_cle(i, A, B)
        cle_tab.append(code_retour)

    return cle_tab

def generer_qr_code(message, qr_code_name):
    img = qrcode.make(message)
    type(img)
    img.save(qr_code_name+".png")
    return qr_code_name+".png"


def dissimulation(qr_hote,qr_secret, nb_germes):

    I = cv2.imread(qr_hote+".png", cv2.IMREAD_GRAYSCALE)
    I1 = cv2.imread(qr_secret+".png", cv2.IMREAD_GRAYSCALE)

    #print(qr_hote+".png")
    #print(qr_secret + ".png")

    width_I1, height_I1 = I1.shape
    width_I, height_I = I.shape

    width = max(width_I1, width_I)
    height = max(height_I,height_I1)

    I = cv2.resize(I,(width,height))
    I1 = cv2.resize(I1, (width, height))

    #width, height = I.shape
    P = np.zeros((width, height), np.uint8)

    N = nb_germes
    V = voronoi_diagramme(N, width, height)

    cle_tab = cle(N)
    #print(cle_tab)

    for i in range(0, width):
        for j in range(0, height):
            k = V[i][j]
            if I[i][j] == 0 and cle_tab[k] == 0:
                P[i][j] = I1[i][j] + (1 - I1[i][j]) * 2
            elif I[i][j] == 0 and cle_tab[k] == 1:
                P[i][j] = (1-I1[i][j]) + I1[i][j] * 2
            elif I[i][j] == 255 and cle_tab[k] == 0:
                P[i][j] = 255 - I1[i][j] - (1-I1[i][j]) * 2
            elif I[i][j] == 255 and cle_tab[k] == 1:
                P[i][j] = 255 - (1-I1[i][j]) - I1[i][j] * 2

    cv2.imshow('code QR augmenté', P)
    cv2.imwrite('QR_AUGMENTE.png', P)
    return V

def decomposer_en_binaire(pixel):
    if pixel < 4:
        b = pixel // 2
        a = pixel % 2
        #print("pixel<4")
    else:
        pixel_ = 255 - pixel
        b = pixel_ // 2
        a = pixel_ % 2
    return a, b


def extraction(qr_code_augmented, cle_voronoi, nb_germes):

    Q = cv2.imread(qr_code_augmented+".png", cv2.IMREAD_GRAYSCALE)
    width,  height = Q.shape

    N = nb_germes

    I1 = np.zeros((width,height), np.uint8)
    I2 = np.zeros((width,height), np.uint8)

    cle_tab = cle(N)

    for i in range(0,width):
        for j in range(0,height):
            k = cle_voronoi[i][j]
            a,b = decomposer_en_binaire(Q[i][j])
            if cle_tab[k] == 0:
                I1[i][j] = a * 255
                I2[i][j] = b * 255
            else:
                #print(b*255)
                I1[i][j] = b * 255
                I2[i][j] = a * 255

    I2 = 255 - I2
    final_Q = cv2.add(I1,I2)

    cv2.imshow("Extraction I", I1)
    cv2.imshow("Extraction négatif I", I2)
    cv2.imshow("QR Code caché", final_Q)

    cv2.imwrite("Extracted_hidden_qr.png", final_Q)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

choix = -1
while(choix != 4):

    print("------------------------------")
    print("1) Créer un code QR avec message")
    print("2) Cacher un code QR dans un hote")
    print("3) Extraire un code QR d'un hote")
    print("4) Quitter")
    print("------------------------------")

    choix = input("Choix : ")
    choix = int(choix)

    if choix == 1:
        print("------------------------------")
        nom = input("Nom du code QR : ")
        message = input("Message du code QR : ")
        generer_qr_code(message,nom)
        print("------------------------------")

    elif choix == 2:
        print("------------------------------")
        qr_hote = input("Nom du code QR hote : ")
        qr_secret = input("Nom du code QR secret : ")
        nb_germes = int(input("Nombres de germes : "))
        voronoi_key = dissimulation(qr_hote,qr_secret,nb_germes)

    elif choix == 3:
        print("------------------------------")
        qr_augmented = input("Nom du code QR Augmenté : ")
        nb_germes = int(input("Nombres de germes : "))
        extraction(qr_augmented,voronoi_key,nb_germes)

    elif choix == 4:
        break

    cv2.waitKey(0)
    cv2.destroyAllWindows()
#TODO - FIX entrée utilisateur
#     - FIX resize pour avoir qr code de meme taille