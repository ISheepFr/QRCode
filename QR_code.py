import random
import qrcode

import cv2
import numpy as np


def distanceEuclidienne(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def distanceAbs(p1, p2):
    return (np.abs(p1[0] - p2[0]) + np.abs(p1[1] - p2[1]))


def distanceInf(p1, p2):
    return max(np.abs(p1[0] - p2[0]), np.abs(p1[1] - p2[1]))


def voronoi_diagramme(nb_germes,image,method=0):

    # nb germes, fonction de distance renseigné par l'utilisateur
    nbgermes = nb_germes

    if method == 0:
        dist_function = distanceEuclidienne
    elif method == 1:
        dist_function = distanceAbs
    elif method == 2:
        dist_function = distanceInf

    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    width, height,  = img.shape

    # va contenir les couleurs des pixels
    #ensemble_pixel_couleur = []
    # va contenir les pixels
    ensemble_pixel = []


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
        for y in range(0, height):
            dist = []

            # calcule et stocke la distance du pixel courant au ensemble de pixel
            for pix in ensemble_pixel:
                dist.append(dist_function((x, y), pix))

            # stocke l'index du pixel dans l'ensemble le + proche du pixel courant
            min_dist = dist.index(min(dist))
            diagramme_voronoi.append(min_dist)

    return diagramme_voronoi


def retour_cle(k,A,B):
    if k in A:
        return 0
    elif k in B:
        return 1

#def code():

#I = qrcode.make("QR_CODE_pas_caché")
#type(I)
#I.save("original.png")
#
#I1 = qrcode.make("QR_CODE_caché!")
#type(I1)
#I1.save("motif.png")

I = cv2.imread("original.png", cv2.IMREAD_GRAYSCALE)
width, height = I.shape
I1 = cv2.imread("motif.png",cv2.IMREAD_GRAYSCALE)
#I1 = cv2.resize(I1,(width,height))
P = cv2.imread("original.png", cv2.IMREAD_GRAYSCALE)

N = 10
V = voronoi_diagramme(N,"original.png")
#print(V)
A = []
#pairs
for i in range(2, N + 1, 2):
    A.append(i)
#impairs
B = []
for i in range(1, N + 1, 2):
    B.append(i)
l=0
for n in range(0,N):
    l=l+1
    for i in range(0, width):
        for j in range(0,height):
            if I[i][j] == 0:
                if retour_cle(V[n],A,B) == 0:
                    #print("-NOIR---cle 0----",V[n])
                    P[i][j] = I1[i][j] + (1 - I1[i][j]) * 2
                else:
                    #print("--NOIR--cle 1----",V[n])
                    P[i][j] = (1-I1[i][j]) + I1[i][j] * 2
            elif I[i][j] == 255:
                if retour_cle(V[n],A,B) == 0:
                    #print("---BLANC--cle 0----",V[n])
                    P[i][j] = 255 - I1[i][j] - (1-I1[i][j]) * 2
                else:
                    #print("--BLANC--cle 1----",V[n])
                    P[i][j] = 255 - (1-I1[i][j]) - I1[i][j] * 2
            #print("Colonne -  ", j)
        #print("Ligne - ", i)
    print(l)

#cv2.imshow("QR MODIFIER", P)
cv2.imwrite('QR_MODIFIER.png', P)
#cv2.imshow("QR NORMAL", cv2.imread('original.png', cv2.IMREAD_GRAYSCALE))
#cv2.imshow("QR A CACHER", I1)
#cv2.waitKey(0)

#
#
def decomposer_en_binaire(pixel):
    if pixel < 4:
        a = pixel // 2
        b = pixel % 2
    else:
        tmp = 255 - pixel
        a = tmp // 2
        b = tmp % 2
    return a, b

N = 10
V = voronoi_diagramme(N,"QR_MODIFIER.png")
##print(V)
A = []
#pairs
for i in range(2, N + 1, 2):
    A.append(i)
#impairs
B = []
for i in range(1, N + 1, 2):
    B.append(i)
#
Q = cv2.imread("QR_MODIFIER.png",cv2.IMREAD_GRAYSCALE)
width, height = Q.shape

I1 = cv2.imread("QR_MODIFIER.png",cv2.IMREAD_GRAYSCALE)
#I1 = np.zeros((width,height), np.uint8)


for k in range(0,N):
    for i in range(0 ,width):
        for j in range(0, height):
            if retour_cle(V[k],A,B) == 0:
                I1[i][j] = decomposer_en_binaire(Q[i][j])[0]
            else:
                I1[i][j] = decomposer_en_binaire(Q[i][j])[1]

#cv2.imshow("QR ENGLOBANT", Q)
cv2.imshow("QR CACHER", I1)
cv2.waitKey(0)
cv2.destroyAllWindows()