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


# le nom de votre classe
# BayesNaif pour le modèle bayesien naif
# Knn pour le modèle des k plus proches voisins

class Knn:  # nom de la class à changer



    def __init__(self, **kwargs):
        """
        c'est un Initializer.
        Vous pouvez passer d'autre paramètres au besoin,
        c'est à vous d'utiliser vos propres notations
        """
        self.train_list = []
        self.train_labels = []
        self.train_list_length = 0
        self.example_length = 0
        self.test_list_length = 0

    def train(self, train, train_labels, dataset):  # vous pouvez rajouter d'autres attribus au besoin
        """
        c'est la méthode qui va entrainer votre modèle,
        train est une matrice de type Numpy et de taille nxm, avec
        n : le nombre d'exemple d'entrainement dans le dataset
        m : le mobre d'attribus (le nombre de caractéristiques)

        train_labels : est une matrice numpy de taille nx1
        dataset : indique le dataset que l'on utilise
            0 : pour bezdekIris.data
            1 : pour house-votes-84.data
            2: pour les monks datasets

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



        # Test pour le dataset bezdekIris
        if dataset == 0:

            list_setosa = [0, 0, 0]
            list_versicolor = [0, 0, 0]
            list_virginica = [0, 0, 0]

            for i in range(self.train_list_length):

                reponse = train_labels[i]
                prediction = self.predict(train[i], train_labels[i], 5)

                # La prédiction de l'algo Knn est vrai
                if prediction[1]:

                    if prediction[0] == 0:
                        list_setosa[0] += 1

                    if prediction[0] == 1:
                        list_versicolor[1] += 1

                    if prediction[0] == 2:
                        list_virginica[2] += 1

                # La prediction de l'algo Knn est fausse
                if not prediction[1]:

                    if reponse == 0:

                        if prediction[0] == 1:
                            list_setosa[1] += 1

                        if prediction[0] == 2:
                            list_setosa[2] += 1

                    if reponse == 1:

                        if prediction[0] == 0:
                            list_versicolor[0] += 1

                        if prediction[0] == 2:
                            list_versicolor[2] += 1

                    if reponse == 2:

                        if prediction[0] == 0:
                            list_virginica[0] += 1

                        if prediction[0] == 1:
                            list_virginica[1] += 1

            # Calcule de l'éxactitude
            somme_bonnes_reponses = list_setosa[0] + list_versicolor[1] + list_virginica[2]
            exacitude = somme_bonnes_reponses / self.train_list_length

            # Calcule de la précision pour la classe setosa
            somme_bonnes_reponse_setosa = list_setosa[0] +list_setosa[1] +list_setosa[2]
            precision_setosa = list_setosa[0] / somme_bonnes_reponse_setosa

            # Calcule du rappel pour la classe setosa
            somme_reponse_setosa_total = list_setosa[0] + list_versicolor[0] +list_virginica[0]
            rappel_setosa = list_setosa[0] / somme_reponse_setosa_total

            # Calcule de la précision pour la classe versicolor
            somme_bonnes_reponse_versicolor = list_versicolor[0] + list_versicolor[1] + list_versicolor[2]
            precision_versicolor = list_versicolor[1] / somme_bonnes_reponse_versicolor

            # Calcule du rappel pour la classe versicolor
            somme_reponse_versicolor_total = list_setosa[1] + list_versicolor[1] + list_virginica[1]
            rappel_versicolor = list_versicolor[1] / somme_reponse_versicolor_total

            # Calcule de la précision pour la classe virginica
            somme_bonnes_reponse_virginica = list_virginica[0] + list_virginica[1] + list_virginica[2]
            precision_virginica = list_virginica[2] / somme_bonnes_reponse_virginica

            # Calcule du rappel pour la classe virginica
            somme_reponse_virginica_total = list_setosa[2] + list_versicolor[2] + list_virginica[2]
            rappel_virginica = list_virginica[2] / somme_reponse_virginica_total

            print("DATASET : bezdekIris")
            print("METHODE : train\n")
            print("Matrice de confusion")
            print(list_setosa)
            print(list_versicolor)
            print(list_virginica)
            print("\nL'éxactitude")
            print(exacitude)
            print("\nLa précision du tri de la classe setosa")
            print(precision_setosa)
            print("\nLe rappel du tri de la classe setosa")
            print(rappel_setosa)
            print("\nLa précision du tri de la classe versicolor")
            print(precision_versicolor)
            print("\nLe rappel du tri de la classe versicolor")
            print(rappel_versicolor)
            print("\nLa précision du tri de la classe virginica")
            print(precision_virginica)
            print("\nLe rappel du tri de la classe virginica")
            print(rappel_virginica)

        # Test pour le dataset house-votes-84
        if dataset == 1:

            list_republicain = [0, 0]
            list_democrat = [0, 0]

            for i in range(self.train_list_length):

                reponse = train_labels[i]
                prediction = self.predict(train[i], train_labels[i], 5)

                # La prediction de l'algo Knn est vrai
                if prediction[1]:

                    if prediction[0] == 0:
                        list_republicain[0] += 1

                    if prediction[0] == 1:
                        list_democrat[1] += 1

                # La prediction de l'algo Knn est vrai
                if not prediction[1]:

                    if reponse == 0:

                        if prediction[0] == 1:
                            list_republicain[1] += 1

                    if reponse == 1:

                        if prediction[0] == 0:
                            list_democrat[0] += 1

            # Calcule de l'éxactitude
            somme_bonnes_reponses = list_republicain[0] + list_democrat[1]
            exacitude = somme_bonnes_reponses / self.train_list_length

            # Calcule de la précision
            somme_nb_republican = list_republicain[0] + list_republicain[1]
            precision = list_republicain[0] / somme_nb_republican

            # Calcule du rappel
            somme_reponse_republican = list_republicain[0] + list_democrat[0]
            rappel = list_republicain[0] / somme_reponse_republican

            print("\nDATASET : house-votes-84")
            print("METHODE : train\n")
            print("Matrice de confusion")
            print(list_republicain)
            print(list_democrat)
            print("\nL'éxactitude")
            print(exacitude)
            print("\nLa précision")
            print(precision)
            print("\nLe rappel")
            print(rappel)

        # Test pour les datasets Monks
        if dataset == 2:

            list_classe_0 = [0, 0]
            list_classe_1 = [0, 0]

            for i in range(self.train_list_length):

                reponse = train_labels[i]
                prediction = self.predict(train[i], train_labels[i], 5)

                # La prediction de l'algo Knn est vrai
                if prediction[1]:

                    if int(prediction[0]) == 0:
                        list_classe_0[0] += 1

                    if int(prediction[0]) == 1:
                        list_classe_1[1] += 1

                # La prediction de l'algo Knn est vrai
                if not prediction[1]:

                    if int(reponse) == 0:

                        if int(prediction[0]) == 1:
                            list_classe_0[1] += 1

                    if int(reponse) == 1:

                        if int(prediction[0]) == 0:
                            list_classe_1[0] += 1

            # Calcule de l'éxactitude
            somme_bonnes_reponses = list_classe_0[0] + list_classe_1[1]
            exacitude = somme_bonnes_reponses / self.train_list_length

            # Calcule de la précision
            somme_nb_classe_0 = list_classe_0[0] + list_classe_0[1]
            precision = list_classe_0[0] / somme_nb_classe_0

            # Calcule du rappel
            somme_reponse_classe_0 = list_classe_0[0] + list_classe_1[0]
            rappel = list_classe_0[0] / somme_reponse_classe_0

            print("\nDATASET : monks")
            print("METHODE : train\n")
            print("Matrice de confusion")
            print(list_classe_0)
            print(list_classe_1)
            print("\nL'éxactitude")
            print(exacitude)
            print("\nLa précision")
            print(precision)
            print("\nLe rappel")
            print(rappel)

    def predict(self, exemple, label, K):
        """
        Prédire la classe d'un exemple donné en entrée
        exemple est de taille 1xm
        K est la taille de la liste des plus proches voisins

        si la valeur retournée est la meme que la veleur dans label
        alors l'exemple est bien classifié, si non c'est une missclassification
        """

        Knn_list = []
        for i in range(K):

            euclidean_distance = 0.0
            addition_attribut = 0.0

            for j in range(self.example_length):

                addition_attribut += (float(exemple[j]) - float(self.train_list[i][j])) ** 2

            euclidean_distance = sqrt(addition_attribut)

            Knn_list.append((self.train_labels[i], euclidean_distance))

        Knn_list.sort(key=lambda x: x[1])

        for i in range(K, self.train_list_length):

            euclidean_distance = 0.0
            addition_attribut = 0.0

            for j in range(self.example_length):

                addition_attribut += (float(exemple[j]) - float(self.train_list[i][j])) ** 2

            euclidean_distance = sqrt(addition_attribut)

            # Vérifions si la valeur du nouveau noeud est plus petite que celui du Ke voisin le plus éloigné
            # Si oui on le remplace par ce nouveau noeud
            if euclidean_distance < Knn_list[K-1][1]:
                Knn_list.pop()
                Knn_list.append((self.train_labels[i], euclidean_distance))
                Knn_list.sort(key=lambda x: x[1])

        Knn_label_list = []
        for i in range(K):
            tuple_list = Knn_list[i]

            Knn_label_list.append(tuple_list[0])

        reponse = self.valeure_la_plus_frequente(Knn_label_list)
        if reponse == label:
            return reponse, True
        else:
            return reponse, False


    def test(self, test, test_labels, dataset):
        """
        c'est la méthode qui va tester votre modèle sur les données de test
        l'argument test est une matrice de type Numpy et de taille nxm, avec
        n : le nombre d'exemple de test dans le dataset
        m : le mobre d'attribus (le nombre de caractéristiques)

        test_labels : est une matrice numpy de taille nx1
        dataset : indique le dataset que l'on utilise
            0 : pour bezdekIris.data
            1 : pour house-votes-84.data
            2: pour les monks datasets

        vous pouvez rajouter d'autres arguments, il suffit juste de
        les expliquer en commentaire

        Faites le test sur les données de test, et afficher :
        - la matrice de confision (confusion matrix)
        - l'accuracy
        - la précision (precision)
        - le rappel (recall)

        Bien entendu ces tests doivent etre faits sur les données de test seulement

        """

        self.test_list_length = len(test)

        # Test pour le dataset bezdekIris
        if dataset == 0:

            list_setosa = [0, 0, 0]
            list_versicolor = [0, 0, 0]
            list_virginica = [0, 0, 0]

            for i in range(self.test_list_length):

                reponse = test_labels[i]
                prediction = self.predict(test[i], test_labels[i], 5)

                # La prédiction de l'algo Knn est vrai
                if prediction[1]:

                    if prediction[0] == 0:
                        list_setosa[0] += 1

                    if prediction[0] == 1:
                        list_versicolor[1] += 1

                    if prediction[0] == 2:
                        list_virginica[2] += 1

                # La prediction de l'algo Knn est fausse
                if not prediction[1]:

                    if reponse == 0:

                        if prediction[0] == 1:
                            list_setosa[1] += 1

                        if prediction[0] == 2:
                            list_setosa[2] += 1

                    if reponse == 1:

                        if prediction[0] == 0:
                            list_versicolor[0] += 1

                        if prediction[0] == 2:
                            list_versicolor [2] += 1

                    if reponse == 2:

                        if prediction[0] == 0:
                            list_virginica[0] += 1

                        if prediction[0] == 1:
                            list_virginica[1] += 1

            # Calcule de l'éxactitude
            somme_bonnes_reponses = list_setosa[0] + list_versicolor[1] + list_virginica[2]
            exacitude = somme_bonnes_reponses / self.test_list_length

            # Calcule de la précision pour la classe setosa
            somme_bonnes_reponse_setosa = list_setosa[0] + list_setosa[1] + list_setosa[2]
            precision_setosa = list_setosa[0] / somme_bonnes_reponse_setosa

            # Calcule du rappel pour la classe setosa
            somme_reponse_setosa_total = list_setosa[0] + list_versicolor[0] + list_virginica[0]
            rappel_setosa = list_setosa[0] / somme_reponse_setosa_total

            # Calcule de la précision pour la classe versicolor
            somme_bonnes_reponse_versicolor = list_versicolor[0] + list_versicolor[1] + list_versicolor[2]
            precision_versicolor = list_versicolor[1] / somme_bonnes_reponse_versicolor

            # Calcule du rappel pour la classe versicolor
            somme_reponse_versicolor_total = list_setosa[1] + list_versicolor[1] + list_virginica[1]
            rappel_versicolor = list_versicolor[1] / somme_reponse_versicolor_total

            # Calcule de la précision pour la classe virginica
            somme_bonnes_reponse_virginica = list_virginica[0] + list_virginica[1] + list_virginica[2]
            precision_virginica = list_virginica[2] / somme_bonnes_reponse_virginica

            # Calcule du rappel pour la classe virginica
            somme_reponse_virginica_total = list_setosa[2] + list_versicolor[2] + list_virginica[2]
            rappel_virginica = list_virginica[2] / somme_reponse_virginica_total

            print("DATASET : bezdekIris")
            print("METHODE : test\n")
            print("Matrice de confusion")
            print(list_setosa)
            print(list_versicolor)
            print(list_virginica)
            print("\nL'éxactitude")
            print(exacitude)
            print("\nLa précision du tri de la classe setosa")
            print(precision_setosa)
            print("\nLe rappel du tri de la classe setosa")
            print(rappel_setosa)
            print("\nLa précision du tri de la classe versicolor")
            print(precision_versicolor)
            print("\nLe rappel du tri de la classe versicolor")
            print(rappel_versicolor)
            print("\nLa précision du tri de la classe virginica")
            print(precision_virginica)
            print("\nLe rappel du tri de la classe virginica")
            print(rappel_virginica)


        # Test pour le dataset house-votes-84
        if dataset == 1:

            list_republicain = [0, 0]
            list_democrat = [0, 0]

            for i in range(self.test_list_length):

                reponse = test_labels[i]
                prediction = self.predict(test[i], test_labels[i], 5)

                # La prediction de l'algo Knn est vrai
                if prediction[1]:

                    if prediction[0] == 0:
                        list_republicain[0] += 1

                    if prediction[0] == 1:
                        list_democrat[1] += 1

                # La prediction de l'algo Knn est vrai
                if not prediction[1]:

                    if reponse == 0:

                        if prediction[0] == 1:
                            list_republicain[1] += 1

                    if reponse == 1:

                        if prediction[0] == 0:
                            list_democrat[0] += 1

            # Calcule de l'éxactitude
            somme_bonnes_reponses = list_republicain[0] + list_democrat[1]
            exacitude = somme_bonnes_reponses / self.test_list_length

            # Calcule de la précision
            somme_nb_republican = list_republicain[0] + list_republicain[1]
            precision = list_republicain[0] / somme_nb_republican

            # Calcule du rappel
            somme_reponse_republican = list_republicain[0] + list_democrat[0]
            rappel = list_republicain[0] / somme_reponse_republican

            print("\nDATASET : house-votes-84")
            print("METHODE : test\n")
            print("Matrice de confusion")
            print(list_republicain)
            print(list_democrat)
            print("\nL'éxactitude")
            print(exacitude)
            print("\nLa précision")
            print(precision)
            print("\nLe rappel")
            print(rappel)


        # Test pour les datasets Monks
        if dataset == 2:

            list_classe_0 = [0, 0]
            list_classe_1 = [0, 0]

            for i in range(self.test_list_length):

                reponse = test_labels[i]
                prediction = self.predict(test[i], test_labels[i], 5)

                # La prediction de l'algo Knn est vrai
                if prediction[1]:

                    if int(prediction[0]) == 0:
                        list_classe_0[0] += 1

                    if int(prediction[0]) == 1:
                        list_classe_1[1] += 1

                # La prediction de l'algo Knn est vrai
                if not prediction[1]:

                    if int(reponse) == 0:

                        if int(prediction[0]) == 1:
                            list_classe_0[1] += 1

                    if int(reponse) == 1:

                        if int(prediction[0]) == 0:
                            list_classe_1[0] += 1

            # Calcule de l'éxactitude
            somme_bonnes_reponses = list_classe_0[0] + list_classe_1[1]
            exacitude = somme_bonnes_reponses / self.test_list_length

            # Calcule de la précision
            somme_nb_classe_0 = list_classe_0[0] + list_classe_0[1]
            precision = list_classe_0[0] / somme_nb_classe_0

            # Calcule du rappel
            somme_reponse_classe_0 = list_classe_0[0] + list_classe_1[0]
            rappel = list_classe_0[0] / somme_reponse_classe_0

            print("\nDATASET : Monks")
            print("METHODE : train\n")
            print("Matrice de confusion")
            print(list_classe_0)
            print(list_classe_1)
            print("\nL'éxactitude")
            print(exacitude)
            print("\nLa précision")
            print(precision)
            print("\nLe rappel")
            print(rappel)

    def valeure_la_plus_frequente(self, list_label_voisin):
        """"
        Fonction qui calcule la valeur qui revient le plus souvent dans une liste.
        l'argument est la liste des labels des plus proches voisins

        list_label_voisin : liste des K plus proches voisins

        """

        counter = 0
        position = list_label_voisin[0]

        for i in range(3):
            frequence_courante = list_label_voisin.count(i)
            if frequence_courante > counter:
                counter = frequence_courante
                position = i

        return position

# Vous pouvez rajouter d'autres méthodes et fonctions,
# il suffit juste de les commenter.