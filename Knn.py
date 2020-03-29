"""
Vous allez definir une classe pour chaque algorithme que vous allez développer,
votre classe doit contenit au moins les 3 methodes definies ici bas,
	* train 	: pour entrainer le modèle sur l'ensemble d'entrainement
	* predict 	: pour prédire la classe d'un exemple donné
	* test 		: pour tester sur l'ensemble de test
vous pouvez rajouter d'autres méthodes qui peuvent vous etre utiles, mais moi
je vais avoir besoin de tester les méthodes test, predict et test de votre code.
"""

import numpy as np
from math import sqrt
from statistics import mode
import operator

# le nom de votre classe
# BayesNaif pour le modèle bayesien naif
# Knn pour le modèle des k plus proches voisins

class Knn:  # nom de la class à changer

    train_list = []
    train_labels = []
    Knn_list = []
    train_list_length = 0
    example_length = 0


    def __init__(self, **kwargs):
        """
        c'est un Initializer.
        Vous pouvez passer d'autre paramètres au besoin,
        c'est à vous d'utiliser vos propres notations
        """
        pass

    def train(self, train, train_labels):  # vous pouvez rajouter d'autres attribus au besoin
        """
        c'est la méthode qui va entrainer votre modèle,
        train est une matrice de type Numpy et de taille nxm, avec
        n : le nombre d'exemple d'entrainement dans le dataset
        m : le mobre d'attribus (le nombre de caractéristiques)

        train_labels : est une matrice numpy de taille nx1

        vous pouvez rajouter d'autres arguments, il suffit juste de
        les expliquer en commentaire



        ------------
        Après avoir fait l'entrainement, faites maintenant le test sur
        les données d'entrainement
        IMPORTANT :
        Vous devez afficher ici avec la commande print() de python,
        - la matrice de confision (confusion matrix)
        - l'accuracy
        - la précision (precision)
        - le rappel (recall)

        Bien entendu ces tests doivent etre faits sur les données d'entrainement
        nous allons faire d'autres tests sur les données de test dans la méthode test()
        """

        self.train_list = train
        self.train_labels = train_labels
        self.train_list_length = len(self.train_list)
        self.example_length = len(self.train_list[0])

    def predict(self, exemple, label, K):
        """
        Prédire la classe d'un exemple donné en entrée
        exemple est de taille 1xm
        K est la taille de la liste des plus proches voisins
        dataset indique le dataset que l'on utilise
            0 : pour bezdekIris.data
            1 : pour house-votes-84.data
            2: pour les monks datasets

        si la valeur retournée est la meme que la veleur dans label
        alors l'exemple est bien classifié, si non c'est une missclassification
        """

        print("longuer de K = " + str(K))
        for i in range(K):

            euclidean_distance = 0.0
            addition_attribut = 0.0

            for j in range(self.example_length):

                addition_attribut += (float(exemple[j]) - float(self.train_list[i][j])) ** 2

            euclidean_distance = sqrt(addition_attribut)

            self.Knn_list.append((self.train_labels[i], euclidean_distance))

        self.Knn_list.sort(key=lambda x: x[1])
        print(self.Knn_list[K-1])

        for i in range(K, self.train_list_length):

            euclidean_distance = 0.0
            addition_attribut = 0.0

            for j in range(self.example_length):

                addition_attribut += (float(exemple[j]) - float(self.train_list[i][j])) ** 2

            euclidean_distance = sqrt(addition_attribut)

            # Vérifions si la valeur du nouveau noeud est plus petite que celui du Ke voisin le plus éloigné
            # Si oui on le remplace par ce nouveau noeud
            if euclidean_distance < self.Knn_list[K-1][1]:
                self.Knn_list.pop()
                self.Knn_list.append((self.train_labels[i], euclidean_distance))
                self.Knn_list.sort(key=lambda x: x[1])

        Knn_label_list = []
        for i in range(K):
            tuple_list = self.Knn_list[i]

            Knn_label_list.append(tuple_list[0])

        if Knn.valeure_la_plus_frequente(self, Knn_label_list) == label:
            return True
        else:
            return False

    def test(self, test, test_labels, ):
        """
        c'est la méthode qui va tester votre modèle sur les données de test
        l'argument test est une matrice de type Numpy et de taille nxm, avec
        n : le nombre d'exemple de test dans le dataset
        m : le mobre d'attribus (le nombre de caractéristiques)

        test_labels : est une matrice numpy de taille nx1

        vous pouvez rajouter d'autres arguments, il suffit juste de
        les expliquer en commentaire

        Faites le test sur les données de test, et afficher :
        - la matrice de confision (confusion matrix)
        - l'accuracy
        - la précision (precision)
        - le rappel (recall)

        Bien entendu ces tests doivent etre faits sur les données de test seulement

        """

    def valeure_la_plus_frequente(self, list_label_voisin):
        """"
        Fonction qui calcule la valeur qui revient le plus souvent dans une liste.
        l'argument est la liste des labels des plus proches voisins

        list_label_voisin : liste des K plus proches voisins

        """

        counter = 0
        position = list_label_voisin[0]

        for i in list_label_voisin:
            frequence_courante = list_label_voisin.count(i)
            if frequence_courante > counter:
                counter = frequence_courante
                position = i

        return position

# Vous pouvez rajouter d'autres méthodes et fonctions,
# il suffit juste de les commenter.