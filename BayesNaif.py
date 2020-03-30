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


# le nom de votre classe
# BayesNaif pour le modèle bayesien naif
# Knn pour le modèle des k plus proches voisins

class BayesNaif:  # nom de la class à changer

    def __init__(self, **kwargs):
        """
        c'est un Initializer.
        Vous pouvez passer d'autre paramètres au besoin,
        c'est à vous d'utiliser vos propres notations
        """
        self.compiled_attributes = None
        self.compiled_class_totals = {}

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

        entries_quantity = len(train)
        entry_length = len(train[0])

        compiled_attributes = []
        for attribute_index in range(entry_length):
            compiled_attributes.append({})

        # vrai work
        for entry_index in range(entries_quantity):
            entry = train[entry_index]
            entry_label = train_labels[entry_index]

            if self.compiled_class_totals.get(entry_label) is None:
                self.compiled_class_totals[entry_label] = 0
            self.compiled_class_totals[entry_label] += 1

            for attribute_index in range(entry_length):
                attribute = entry[attribute_index]

                if compiled_attributes[attribute_index].get(entry_label) is None:
                    compiled_attributes[attribute_index][entry_label] = {}
                if compiled_attributes[attribute_index][entry_label].get(attribute) is None:
                    compiled_attributes[attribute_index][entry_label][attribute] = 0

                compiled_attributes[attribute_index][entry_label][attribute] += 1

        self.compiled_attributes = compiled_attributes


    def predict(self, exemple, label):
        """
        Prédire la classe d'un exemple donné en entrée
        exemple est de taille 1xm

        si la valeur retournée est la meme que la veleur dans label
        alors l'exemple est bien classifié, si non c'est une missclassification
        """

        compiled_attributes = self.compiled_attributes
        global_total = 0
        for value in self.compiled_class_totals.values():
            global_total += value

        class_total = self.compiled_class_totals.get(label)
        probability_of_class = class_total / global_total
        for attribute_index in range(len(exemple)):
            exemple_attribute = exemple[attribute_index]
            total_of_attribute_for_class = compiled_attributes[attribute_index][label][exemple_attribute]

            probability_of_attribute_given_class = total_of_attribute_for_class / class_total
            probability_of_class *= probability_of_attribute_given_class

        counter_label = 1 if label == 0 else 0
        counter_class_total = self.compiled_class_totals.get(counter_label)
        probability_of_counter_class = counter_class_total / global_total
        for attribute_index in range(len(exemple)):
            exemple_attribute = exemple[attribute_index]
            total_of_attribute_for_class = compiled_attributes[attribute_index][counter_label][exemple_attribute]

            probability_of_attribute_given_class = total_of_attribute_for_class / counter_class_total
            probability_of_counter_class *= probability_of_attribute_given_class

        reduction = probability_of_class / probability_of_counter_class

        expected_answer = label if reduction > 1 else counter_label
        return expected_answer is label


    def test(self, test, test_labels):
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

# Vous pouvez rajouter d'autres méthodes et fonctions,
# il suffit juste de les commenter.