import cv2
import numpy as np
import random
import qrcode
import os
import copy
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

def generer_qr_code(message, qr_code_name, level=None):

    #Utilisation de level pour créer un qr code d'une certaine version si on doit resize

    if level == None:
        qr = qrcode.QRCode(
            box_size=10,
            border=4,
        )
        qr.add_data(message)
        qr.make(fit=True)
    else:
        qr = qrcode.QRCode(
            version=level,
            box_size=10,
            border=4,
        )
        qr.add_data(message)
        qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")
    #img = qrcode.make(message)
    type(img)
    img.save("generated_qr/"+qr_code_name+".png")
    return qr_code_name+".png"

#-------------------------------DISSIMULATION-------------------------------------------------------------------------------------
def dissimulation(qr_hote,qr_secret, nb_germes, nom):

    I = cv2.imread("generated_qr/"+qr_hote, cv2.IMREAD_GRAYSCALE)
    I1 = cv2.imread("generated_qr/"+qr_secret, cv2.IMREAD_GRAYSCALE)

    width_I1, height_I1 = I1.shape
    width_I, height_I = I.shape

    if(width_I1 != width_I):
        print("Les qr codes sont de taille différentes, regénération ...")

        if width_I1 > width_I:
            print("Taille du secret > hote")

            straight_qrcode = cv2.QRCodeDetector().detectAndDecode(I1)[2]
            # Calculer la version du QR code
            if 21 <= len(straight_qrcode) <= 177:
                level_to_scale = (len(straight_qrcode) - 17) // 4
            elif 179 <= len(straight_qrcode) <= 370:
                level_to_scale = (len(straight_qrcode) - 21) // 4

            #print(level_to_scale)

            qrI = cv2.QRCodeDetector().detectAndDecode(I)[0]
            msg_to_rewrite = qrI

            qr_hote_without_png = qr_hote.split('.')[0]
            generer_qr_code(msg_to_rewrite,qr_hote_without_png,level_to_scale)

            print("Nouveau qr de meme taille que le caché sauvegarder !")
            I = cv2.imread("generated_qr/" + qr_hote, cv2.IMREAD_GRAYSCALE)

            #OK PAS INVERSER

        else:
            print("Taille de l'hote > secret")

            straight_qrcode = cv2.QRCodeDetector().detectAndDecode(I)[2]
            # Calculer la version du QR code
            if 21 <= len(straight_qrcode) <= 177:
                level_to_scale = (len(straight_qrcode) - 17) // 4
            elif 179 <= len(straight_qrcode) <= 370:
                level_to_scale = (len(straight_qrcode) - 21) // 4

            #print(level_to_scale)

            qrI1 = cv2.QRCodeDetector().detectAndDecode(I1)[0]
            msg_to_rewrite = qrI1
            print("message rewrite pour le cache: ",msg_to_rewrite)

            qr_secret_without_png = qr_secret.split('.')[0]
            generer_qr_code(msg_to_rewrite, qr_secret_without_png, level_to_scale)

            print("Nouveau qr de meme taille que l'hote sauvegarder !")
            I1 = cv2.imread("generated_qr/" + qr_secret, cv2.IMREAD_GRAYSCALE)


    width , height = I.shape


    P = np.zeros((width, height), np.uint8)

    N = nb_germes
    V = voronoi_diagramme(N, width, height)

    cle_tab = cle(N)

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
    cv2.imwrite('augmented_qr/'+nom+".png", P)
    return V


def dissimulation_N(qr_hote, qrs_secrets, nom):
    I = cv2.imread("generated_qr/" + qr_hote, cv2.IMREAD_GRAYSCALE)

    width, height = I.shape
    P = np.zeros((width, height), np.uint8)

    matrice_images = []
    for image in range(0,len(qrs_secrets)):
        unpack_img = cv2.imread("generated_qr/" + qrs_secrets[image], cv2.IMREAD_GRAYSCALE)
        for ligne in range(0,width):
            for colonne in range(0,height):
                if unpack_img[ligne][colonne] != 0:
                    unpack_img[ligne][colonne] = 1
        matrice_images.append(unpack_img)


    for i in range(0, width):
        for j in range(0, height):
            somme_pix = 0
            for x in range(0, len(matrice_images)):
                somme_pix = somme_pix + (matrice_images[x][i][j] * 2 ** x)

            if I[i][j] == 0:
                P[i][j] = somme_pix
            elif I[i][j] == 255:
                P[i][j] = 255 - somme_pix

    cv2.imshow('code QR augmenté', P)
    cv2.imwrite('augmented_qr/'+nom+".png", P)

    return P

def dissimulation_N_voronoi(qr_hote, qrs_secrets, nb_germes, nom):
    I = cv2.imread("generated_qr/" + qr_hote, cv2.IMREAD_GRAYSCALE)

    width, height = I.shape
    P = np.zeros((width, height), np.uint8)

    N = nb_germes
    V = voronoi_diagramme(N, width, height)

    matrice_images = []
    for image in range(0, len(qrs_secrets)):
        unpack_img = cv2.imread("generated_qr/" + qrs_secrets[image], cv2.IMREAD_GRAYSCALE)
        for ligne in range(0, width):
            for colonne in range(0, height):
                if unpack_img[ligne][colonne] != 0:
                    unpack_img[ligne][colonne] = 1
        matrice_images.append(unpack_img)

    for i in range(0, width):
        for j in range(0, height):

            if V[i][j] == 0:
                if I[i][j] == 0:
                    P[i][j] = matrice_images[3][i][j] * (2 ** 0) + matrice_images[2][i][j] * (2 ** 1) + matrice_images[0][i][j] * (2 ** 2) + matrice_images[1][i][j] * (2 ** 3)
                elif I[i][j] == 255:
                    P[i][j] = 255 - matrice_images[3][i][j] * (2 ** 0) - matrice_images[2][i][j] * (2 ** 1) - matrice_images[0][i][j] * (2 ** 2) - matrice_images[1][i][j] * (2 ** 3)

            elif V[i][j] == 1:
                if I[i][j] == 0:
                    P[i][j] = matrice_images[1][i][j] * (2 ** 0) + matrice_images[3][i][j] * (2 ** 1) + matrice_images[2][i][j] * (2 ** 2) + matrice_images[0][i][j] * (2 ** 3)
                elif I[i][j] == 255:
                    P[i][j] = 255 - matrice_images[1][i][j] * (2 ** 0) - matrice_images[3][i][j] * (2 ** 1) - matrice_images[2][i][j] * (2 ** 2) - matrice_images[0][i][j] * (2 ** 3)

            elif V[i][j] == 2:
                if I[i][j] == 0:
                    P[i][j] = matrice_images[2][i][j] * (2 ** 0) + matrice_images[0][i][j] * (2 ** 1) + matrice_images[3][i][j] * (2 ** 2) + matrice_images[1][i][j] * (2 ** 3)
                elif I[i][j] == 255:
                    P[i][j] = 255 - matrice_images[2][i][j] * (2 ** 0) - matrice_images[0][i][j] * (2 ** 1) - matrice_images[3][i][j] * (2 ** 2) - matrice_images[1][i][j] * (2 ** 3)


    cv2.imshow('code QR augmenté', P)
    cv2.imwrite('augmented_qr/' + nom + ".png", P)

    return V

#-------------------------------BINAIRE-------------------------------------------------------------------------------------
def decomposer_en_binaire(pixel):
    if pixel < 4:
        b = pixel // 2
        a = pixel % 2
    else:
        pixel_ = 255 - pixel
        b = pixel_ // 2
        a = pixel_ % 2
    return a, b

def decomposer_en_binaire_N(pixel_i, pixel_j, img,n, sens=-1):
    if img[pixel_i][pixel_j] < 2 ** (n+1):
        valeur_return = decomposer_en_binaire_reel(img[pixel_i][pixel_j],n,sens)
    else:
        valeur_return = decomposer_en_binaire_reel(255 - img[pixel_i][pixel_j],n,sens)

    return valeur_return

def decomposer_en_binaire_reel(valeur, n, sens=-1):
    if valeur < 0:
        raise ValueError("La fonction ne prend en charge que les valeurs positives.")
    if valeur == 0:
        return [0] * n  # Retourner une liste de n zéros si la valeur est 0

    bits = []
    while valeur > 0:
        bits.insert(0, valeur % 2)
        valeur //= 2

    # Remplir les bits manquants avec des zéros
    while len(bits) < n:
        bits.insert(0, 0)


    #[ 2^3, 2^2, 2^1, 2^0 ]
    bits_temp = copy.copy(bits)

    #print("Sens normal : ", bits)

    if sens == 0:

    # [ 2^3, 2^2, 2^1, 2^0 ]
    # [ 2^3, 2^2, 2^0, 2^1 ]
        bits[2] = bits_temp[3]
        bits[3] = bits_temp[2]
        #print("Sens [ 3 2 0 1] : ", bits)

    elif sens ==  1:
    # [ 2^3, 2^2, 2^1, 2^0 ]
    # [ 2^1, 2^3, 2^2, 2^0 ]
        bits[0] = bits_temp[2]
        bits[1] = bits_temp[0]
        bits[2] = bits_temp[1]
        #print("Sens [ 1 3 2 0] : ", bits_temp)

    elif sens == 2:
    # [ 2^3, 2^2, 2^1, 2^0 ]
    # [ 2^2, 2^0, 2^3, 2^1 ]
        bits[0] = bits_temp[1]
        bits[1] = bits_temp[3]
        bits[2] = bits_temp[0]
        bits[3] = bits_temp[2]
        #print("Sens [ 2 0 3 1] : ", bits_temp)

    return bits

#----------------------------------EXTRACTION------------------------------------------------------------------------------------
def extraction(qr_code_augmented, cle_voronoi, nb_germes):

    Q = cv2.imread("augmented_qr/"+qr_code_augmented, cv2.IMREAD_GRAYSCALE)
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

    output_img = qr_code_augmented.split('.')

    cv2.imwrite("extracted_qr/"+output_img[0]+"_extracted."+output_img[1], final_Q)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def extraction_N(qr_code_augmented,nb_images):
    Q = cv2.imread("augmented_qr/" + qr_code_augmented, cv2.IMREAD_GRAYSCALE)
    width, height = Q.shape

    extracted_images = []
    for img in range(0,nb_images):
        img = np.zeros((width, height), np.uint8)
        extracted_images.append(img)

    for i in range(width):
        for j in range(height):
            valeur = decomposer_en_binaire_N(i,j,Q,nb_images)
            for v in range(0,len(valeur)):
                extracted_images[v][i][j] = valeur[v] * 255

    for index in range(0,len(extracted_images)):
        image = extracted_images[index]

        cv2.imshow("extracted_"+str(index),image)
        img_without_png = qr_code_augmented.split('.')[0]
        cv2.imwrite("extracted_qr/"+img_without_png+"_e"+str(index)+".png",image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def extraction_N_voronoi(qr_code_augmented, cle_voronoi, nb_germes,nb_images):

    Q = cv2.imread("augmented_qr/"+qr_code_augmented, cv2.IMREAD_GRAYSCALE)
    width,  height = Q.shape

    N = nb_germes

    extracted_images = [[np.zeros((width, height), np.uint8) for _ in range(nb_images)] for _ in range(3)]

    for i in range(width):
        for j in range(height):
            k = cle_voronoi[i][j]
            valeur = decomposer_en_binaire_N(i, j, Q, nb_images, k)
            for v in range(len(valeur)):
                extracted_images[k][v][i][j] = valeur[v] * 255

    for index in range(0, nb_images):
        print(index)

        #image = extracted_images[0][index]
        if index == 0:
            region_2 = extracted_images[0][index+2]
            region_1 = extracted_images[1][index]
            region_0 = extracted_images[2][index]

        elif index == 1:
            region_2 = extracted_images[0][index + 2]
            region_1 = extracted_images[1][index + 1]
            region_0 = extracted_images[2][index]

        elif index == 2:
            # [0] car 3 + 2 > 3
            region_2 = extracted_images[0][0]
            region_1 = extracted_images[1][index+1]
            region_0 = extracted_images[2][index]

        elif index == 3:
            #[0] car 3 + 2 > 3
            region_2 = extracted_images[0][1]
            region_1 = extracted_images[1][1]
            region_0 = extracted_images[2][index]


        cv2.imshow("FINALE", region_0 + region_1 + region_2)
        cv2.imshow("r0", region_0)
        cv2.imshow("r1",region_1)
        cv2.imshow("r2", region_2)

        img_without_png = qr_code_augmented.split('.')[0]
        cv2.imwrite("extracted_qr/" + img_without_png + "_e" + str(index) + ".png", region_0+region_1+region_2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def allExtensionDirectory(directory,extension):
    images = []
    i = 1
    for imageName in os.listdir(directory):
        if imageName.endswith("."+extension):
            print(str(i)+") "+imageName)
            i+=1
            images.append(imageName)
    return images

choix = -1

if not os.path.exists('generated_qr'):
    os.makedirs('generated_qr')
if not os.path.exists('augmented_qr'):
    os.makedirs('augmented_qr')
if not os.path.exists('extracted_qr'):
    os.makedirs('extracted_qr')

while(choix != 8):

    print("------------------------------")
    print("1) Créer un code QR avec message")
    print("2) Cacher un code QR dans un hote")
    print("3) Extraire un code QR d'un hote")
    print("4) Cacher plusieurs codes QR dans un hote")
    print("5) Extraire N codes Qr d'un hote")
    print("6) Voronoi - cacher N codes QR dans un hote")
    print("7) Voronoi - extraire N code QR d'un hote")
    print("8) Quitter")
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
        print("    LISTE DES QR CODES DISPONIBLES    ")
        tab_images = allExtensionDirectory('generated_qr','png')
        print("------------------------------")
        print(str(len(tab_images)+1)+") Retour")
        choix_hote = int(input("Choix hote : "))

        while (choix_hote < 1 or choix_hote > len(tab_images)+1):
            choix_hote = int(input("Choix hote : "))

        if choix_hote != len(tab_images)+1:
            choix_secret = int(input("Choix QR à dissimuler : "))
            nb_germes = int(input("Nombres de germes : "))
            nom = input("Nom du qr code final : ")

            voronoi_key = dissimulation(tab_images[choix_hote-1],tab_images[choix_secret-1],nb_germes,nom)
            with open("augmented_qr/"+nom+".txt", 'w') as cle_file:
                for sous_liste in voronoi_key:
                    # Convertir les éléments de la sous-liste en chaînes de caractères
                    elements_str = [str(element) for element in sous_liste]
                    # Joindre les éléments en une seule chaîne séparée par des virgules
                    ligne = ','.join(elements_str)
                    # Écrire la ligne dans le fichier
                    cle_file.write(ligne + '\n')
                cle_file.write(str(nb_germes)+'\n')
            cle_file.close()

    elif choix == 3:
        print("------------------------------")
        print("    LISTE DES QR CODES AUGMENTES    ")
        tab_images = allExtensionDirectory('augmented_qr','png')
        print("------------------------------")
        print(str(len(tab_images) + 1) + ") Retour")
        choix_augmente = int(input("Choix QR Augmenté a extraire : "))
        while (choix_augmente < 1 or choix_augmente > len(tab_images)+1):
            choix_augmente = int(input("Choix QR Augmenté a extraire : "))

        if choix_augmente != len(tab_images) + 1:
            print("    LISTE DES CLES DISPONIBLES  ")
            tab_txt = allExtensionDirectory('augmented_qr','txt')
            choix_cle = int(input("Choix clé : "))

            voronoi_key = []
            with open('augmented_qr/'+tab_txt[choix_cle-1], 'r') as fichier:
                for ligne in fichier:
                    # Diviser la ligne en éléments individuels
                    elements = ligne.strip().split(',')
                    # Convertir les éléments en entiers
                    sous_liste = [int(element) for element in elements]
                    # Ajouter la sous-liste à la liste de listes
                    voronoi_key.append(sous_liste)
            fichier.close()

            nb_germes = voronoi_key.pop()[0]
            print(nb_germes)
            extraction(tab_images[choix_augmente-1],voronoi_key,nb_germes)

    elif choix == 4:
        print("------------------------------")
        print("    LISTE DES QR CODES DISPONIBLES    ")
        tab_images = allExtensionDirectory('generated_qr','png')
        print("------------------------------")
        print(str(len(tab_images)+1)+") Retour")
        choix_hote = int(input("Choix hote : "))

        while (choix_hote < 1 or choix_hote > len(tab_images)+1):
            choix_hote = int(input("Choix hote : "))

        if choix_hote != len(tab_images)+1:
            choix_all_secret = []

            choix_secret =  -1
            while choix_secret != 0:
                print("0) Finir")
                choix_secret = int(input("Choix QR à dissimuler : "))
                if choix_secret != 0:
                    choix_all_secret.append(choix_secret)

            nom = input("Nom du qr code final : ")

            final_tab_secret = []
            for elem in choix_all_secret:
                final_tab_secret.append(tab_images[elem-1])

            #print(choix_all_secret)
            voronoi_key = dissimulation_N(tab_images[choix_hote-1], final_tab_secret, nom)
            #voronoi_key = dissimulation(tab_images[choix_hote-1],tab_images[choix_secret-1],nb_germes,nom)
            #with open("augmented_qr/"+nom+".txt", 'w') as cle_file:
            #    for sous_liste in voronoi_key:
            #        # Convertir les éléments de la sous-liste en chaînes de caractères
            #        elements_str = [str(element) for element in sous_liste]
            #        # Joindre les éléments en une seule chaîne séparée par des virgules
            #        ligne = ','.join(elements_str)
            #        # Écrire la ligne dans le fichier
            #        cle_file.write(ligne + '\n')
            #    cle_file.write(str(nb_germes)+'\n')
            #cle_file.close()

    elif choix == 5:
        print("------------------------------")
        print("    LISTE DES QR CODES AUGMENTES    ")
        tab_images = allExtensionDirectory('augmented_qr', 'png')
        print("------------------------------")
        print(str(len(tab_images) + 1) + ") Retour")
        choix_augmente = int(input("Choix QR Augmenté a extraire : "))
        while (choix_augmente < 1 or choix_augmente > len(tab_images) + 1):
            choix_augmente = int(input("Choix QR Augmenté a extraire : "))

        if choix_augmente != len(tab_images) + 1:
            nb_images = int(input("Nb d'img a extraire : "))
            extraction_N(tab_images[choix_augmente-1],nb_images)

    elif choix == 6:
        print("------------------------------")
        print("    LISTE DES QR CODES DISPONIBLES    ")
        tab_images = allExtensionDirectory('generated_qr', 'png')
        print("------------------------------")
        print(str(len(tab_images) + 1) + ") Retour")
        choix_hote = int(input("Choix hote : "))

        while (choix_hote < 1 or choix_hote > len(tab_images) + 1):
            choix_hote = int(input("Choix hote : "))

        if choix_hote != len(tab_images) + 1:
            choix_all_secret = []

            choix_secret = -1
            while choix_secret != 0:
                print("0) Finir")
                choix_secret = int(input("Choix QR à dissimuler : "))
                if choix_secret != 0:
                    choix_all_secret.append(choix_secret)

            nb_germes = 3#int(input("Nombres de germes : "))
            nom = input("Nom du qr code final : ")

            final_tab_secret = []
            for elem in choix_all_secret:
                final_tab_secret.append(tab_images[elem - 1])

            # print(choix_all_secret)
            voronoi_key = dissimulation_N_voronoi(tab_images[choix_hote - 1], final_tab_secret, nb_germes, nom)
            with open("augmented_qr/"+nom+".txt", 'w') as cle_file:
               for sous_liste in voronoi_key:
                   # Convertir les éléments de la sous-liste en chaînes de caractères
                   elements_str = [str(element) for element in sous_liste]
                   # Joindre les éléments en une seule chaîne séparée par des virgules
                   ligne = ','.join(elements_str)
                   # Écrire la ligne dans le fichier
                   cle_file.write(ligne + '\n')
               cle_file.write(str(nb_germes)+'\n')
            cle_file.close()


    elif choix == 7:
        print("------------------------------")
        print("    LISTE DES QR CODES AUGMENTES    ")
        tab_images = allExtensionDirectory('augmented_qr', 'png')
        print("------------------------------")
        print(str(len(tab_images) + 1) + ") Retour")
        choix_augmente = int(input("Choix QR Augmenté a extraire : "))
        while (choix_augmente < 1 or choix_augmente > len(tab_images) + 1):
            choix_augmente = int(input("Choix QR Augmenté a extraire : "))

        if choix_augmente != len(tab_images) + 1:
            print("    LISTE DES CLES DISPONIBLES  ")
            tab_txt = allExtensionDirectory('augmented_qr', 'txt')
            choix_cle = int(input("Choix clé : "))
            nb_images = int(input("Nb d'img a extraire : "))

            voronoi_key = []
            with open('augmented_qr/' + tab_txt[choix_cle - 1], 'r') as fichier:
                for ligne in fichier:
                    # Diviser la ligne en éléments individuels
                    elements = ligne.strip().split(',')
                    # Convertir les éléments en entiers
                    sous_liste = [int(element) for element in elements]
                    # Ajouter la sous-liste à la liste de listes
                    voronoi_key.append(sous_liste)
            fichier.close()

            nb_germes = voronoi_key.pop()[0]
            print(nb_germes)
            #print(voronoi_key)
            extraction_N_voronoi(tab_images[choix_augmente - 1], voronoi_key, nb_germes,nb_images)

    elif choix == 8:
        break

    cv2.waitKey(0)
    cv2.destroyAllWindows()

#TODO - FIX entrée utilisateur
#     - FIX resize pour avoir qr code de meme taille