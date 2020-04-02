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
import math


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

        self.class0_avg = []
        self.class1_avg = []
        self.class2_avg = []
        self.class0_variance = []
        self.class1_variance = []
        self.class2_variance = []

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

        # Test pour le dataset bezdekIris
        if dataset == 0:
            entries_quantity = len(train)
            sum_class0, sum_class1, sum_class2 = 0, 0, 0
            class0_avg, class1_avg, class2_avg = [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]
            for entry_index in range(entries_quantity):
                entry = train[entry_index]
                entry_label = train_labels[entry_index]

                if entry_label == 0:
                    sum_class0 += 1
                    class0_avg[0] += float(entry[0])
                    class0_avg[1] += float(entry[1])
                    class0_avg[2] += float(entry[2])
                    class0_avg[3] += float(entry[3])
                elif entry_label == 1:
                    sum_class1 += 1
                    class1_avg[0] += float(entry[0])
                    class1_avg[1] += float(entry[1])
                    class1_avg[2] += float(entry[2])
                    class1_avg[3] += float(entry[3])
                elif entry_label == 2:
                    sum_class2 += 1
                    class2_avg[0] += float(entry[0])
                    class2_avg[1] += float(entry[1])
                    class2_avg[2] += float(entry[2])
                    class2_avg[3] += float(entry[3])

            for i in range(len(class0_avg)):
                class0_avg[i] = class0_avg[i] / sum_class0
            for i in range(len(class1_avg)):
                class1_avg[i] = class1_avg[i] / sum_class1
            for i in range(len(class2_avg)):
                class2_avg[i] = class2_avg[i] / sum_class2

            class0_variance, class1_variance, class2_variance = [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]
            for entry_index in range(entries_quantity):
                entry = train[entry_index]
                entry_label = train_labels[entry_index]

                if entry_label == 0:
                    class0_variance[0] += (float(entry[0]) - class0_avg[0]) ** 2
                    class0_variance[1] += (float(entry[1]) - class0_avg[1]) ** 2
                    class0_variance[2] += (float(entry[2]) - class0_avg[2]) ** 2
                    class0_variance[3] += (float(entry[3]) - class0_avg[3]) ** 2
                elif entry_label == 1:
                    class1_variance[0] += (float(entry[0]) - class1_avg[0]) ** 2
                    class1_variance[1] += (float(entry[1]) - class1_avg[1]) ** 2
                    class1_variance[2] += (float(entry[2]) - class1_avg[2]) ** 2
                    class1_variance[3] += (float(entry[3]) - class1_avg[3]) ** 2
                elif entry_label == 2:
                    class2_variance[0] += (float(entry[0]) - class2_avg[0]) ** 2
                    class2_variance[1] += (float(entry[1]) - class2_avg[1]) ** 2
                    class2_variance[2] += (float(entry[2]) - class2_avg[2]) ** 2
                    class2_variance[3] += (float(entry[3]) - class2_avg[3]) ** 2

            for i in range(len(class0_variance)):
                class0_variance[i] = class0_variance[i] / (sum_class0-1)
            for i in range(len(class1_variance)):
                class1_variance[i] = class1_variance[i] / (sum_class1-1)
            for i in range(len(class2_variance)):
                class2_variance[i] = class2_variance[i] / (sum_class2-1)

            self.class0_avg = class0_avg
            self.class1_avg = class1_avg
            self.class2_avg = class2_avg
            self.class0_variance = class0_variance
            self.class1_variance = class1_variance
            self.class2_variance = class2_variance

            predictions = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
            for i in range(entries_quantity):
                label = train_labels[i]
                prediction = self.predict(train[i], label, 0)
                predictions[label][prediction] += 1

            # Calcule de l'éxactitude
            all_good_answers = predictions[0][0] + predictions[1][1] + predictions[2][2]
            exactitude = all_good_answers / entries_quantity

            # Calcule de la précision
            precision = 1
            class0_positives, class1_positives, class2_positives = predictions[0][0], predictions[1][1], predictions[2][2]
            class0_false_positives = predictions[1][0] + predictions[2][0]
            class1_false_positives = predictions[0][1] + predictions[2][1]
            class2_false_positives = predictions[0][2] + predictions[1][2]
            precision *= (class0_positives / (class0_positives+class0_false_positives))
            precision *= (class1_positives / (class1_positives+class1_false_positives))
            precision *= (class2_positives / (class2_positives+class2_false_positives))

            # Calcule du rappel
            class0_false_negatives = predictions[0][1] + predictions[0][2]
            class1_false_negatives = predictions[1][0] + predictions[1][2]
            class2_false_negatives = predictions[2][0] + predictions[2][1]
            rappel = 1
            rappel *= (class0_positives / (class0_positives+class0_false_negatives))
            rappel *= (class1_positives / (class0_positives+class1_false_negatives))
            rappel *= (class2_positives / (class0_positives+class2_false_negatives))

            print("\nDATASET : bezdekIris")
            print("METHODE : train\n")
            print("Matrice de confusion")
            print("     prediction ->  setosa | versicolor | virginica")
            print("reality setosa    :   ", predictions[0][0], "       ", predictions[0][1], "       ", predictions[0][2])
            print("reality versicolor:   ", predictions[1][0], "       ", predictions[1][1], "       ", predictions[1][2])
            print("reality virginica :   ", predictions[2][0], "       ", predictions[2][1], "       ", predictions[2][2])
            print("\nL'éxactitude")
            print(exactitude)
            print("\nLa précision")
            print(precision)
            print("\nLe rappel")
            print(rappel)

        # Test pour le dataset house-votes-84
        if dataset == 1:
            entries_quantity = len(train)
            entry_length = len(train[0])

            compiled_attributes = []
            for attribute_index in range(entry_length):
                compiled_attributes.append({})

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

            # matrice de confusion
            democrat_and_correct = 0
            democrat_and_incorrect = 0
            republicain_and_correct = 0
            republicain_and_incorrect = 0
            for entry_index in range(entries_quantity):
                reality = train_labels[entry_index]
                prediction = self.predict(train[entry_index], reality, 1)

                #democrat
                if reality == 0:
                    if prediction:
                        democrat_and_correct += 1
                    else:
                        democrat_and_incorrect += 1
                #republicain
                if reality == 1:
                    if prediction:
                        republicain_and_correct += 1
                    else:
                        republicain_and_incorrect += 1

            # Calcule de l'éxactitude
            all_good_answers = republicain_and_correct + democrat_and_correct
            exactitude = all_good_answers / entries_quantity

            # Calcule de la précision
            somme_nb_republican = republicain_and_correct + democrat_and_incorrect
            precision = republicain_and_correct / somme_nb_republican

            # Calcule du rappel
            somme_reponse_republican = republicain_and_correct + republicain_and_incorrect
            rappel = republicain_and_correct / somme_reponse_republican

            print("\nDATASET : house-votes-84")
            print("METHODE : train\n")
            print("Matrice de confusion")
            print("          prediction ->  republicain | democrat")
            print("reality republicain:      ", republicain_and_correct, "         ", republicain_and_incorrect)
            print("reality democrat   :      ", democrat_and_incorrect, "          ", democrat_and_correct)
            print("\nL'éxactitude")
            print(exactitude)
            print("\nLa précision")
            print(precision)
            print("\nLe rappel")
            print(rappel)

        # Test pour les datasets Monks
        if dataset == 2:
            pass

    def predict(self, exemple, label, dataset):
        """
        Prédire la classe d'un exemple donné en entrée
        exemple est de taille 1xm

        si la valeur retournée est la meme que la veleur dans label
        alors l'exemple est bien classifié, si non c'est une missclassification
        """
        # Test pour le dataset bezdekIris
        if dataset == 0:
            class0_prob = 1
            class1_prob = 1
            class2_prob = 1
            for i in range(len(exemple)):
                x = exemple[i]
                class0_prob *= self.prob_for_class(self.class0_avg[i], self.class0_variance[i], x)
                class1_prob *= self.prob_for_class(self.class1_avg[i], self.class1_variance[i], x)
                class2_prob *= self.prob_for_class(self.class2_avg[i], self.class2_variance[i], x)

            as_list = [class0_prob, class1_prob, class2_prob]
            best_match_class = as_list.index(max(as_list))
            return best_match_class

        # Test pour le dataset house-votes-84
        if dataset == 1:
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

        # Test pour les datasets Monks
        if dataset == 2:
            pass

    def prob_for_class(self, avg, variance, x):
        main_arg = 1/(math.sqrt(2*math.pi*variance))
        exponent = -1 * (((float(x)-avg)**2)/(2*variance))

        prob = main_arg * (math.e**exponent)

        return prob

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
        entries_quantity = len(test)

        if dataset == 0:
            #matrice de confusion
            predictions = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
            for i in range(entries_quantity):
                label = test_labels[i]
                prediction = self.predict(test[i], label, 0)
                predictions[label][prediction] += 1

            # Calcule de l'éxactitude
            all_good_answers = predictions[0][0] + predictions[1][1] + predictions[2][2]
            exactitude = all_good_answers / entries_quantity

            # Calcule de la précision
            precision = 1
            class0_positives, class1_positives, class2_positives = predictions[0][0], predictions[1][1], predictions[2][
                2]
            class0_false_positives = predictions[1][0] + predictions[2][0]
            class1_false_positives = predictions[0][1] + predictions[2][1]
            class2_false_positives = predictions[0][2] + predictions[1][2]
            precision *= (class0_positives / (class0_positives + class0_false_positives))
            precision *= (class1_positives / (class1_positives + class1_false_positives))
            precision *= (class2_positives / (class2_positives + class2_false_positives))

            # Calcule du rappel
            class0_false_negatives = predictions[0][1] + predictions[0][2]
            class1_false_negatives = predictions[1][0] + predictions[1][2]
            class2_false_negatives = predictions[2][0] + predictions[2][1]
            rappel = 1
            rappel *= (class0_positives / (class0_positives + class0_false_negatives))
            rappel *= (class1_positives / (class0_positives + class1_false_negatives))
            rappel *= (class2_positives / (class0_positives + class2_false_negatives))

            print("\nDATASET : bezdekIris")
            print("METHODE : test\n")
            print("Matrice de confusion")
            print("     prediction ->  setosa | versicolor | virginica")
            print("reality setosa    :   ", predictions[0][0], "       ", predictions[0][1], "       ",
                  predictions[0][2])
            print("reality versicolor:   ", predictions[1][0], "       ", predictions[1][1], "       ",
                  predictions[1][2])
            print("reality virginica :   ", predictions[2][0], "       ", predictions[2][1], "       ",
                  predictions[2][2])
            print("\nL'éxactitude")
            print(exactitude)
            print("\nLa précision")
            print(precision)
            print("\nLe rappel")
            print(rappel)

        if dataset == 1:
            # matrice de confusion
            democrat_and_correct = 0
            democrat_and_incorrect = 0
            republicain_and_correct = 0
            republicain_and_incorrect = 0
            for entry_index in range(entries_quantity):
                reality = test_labels[entry_index]
                prediction = self.predict(test[entry_index], reality, 1)

                #democrat
                if reality == 0:
                    if prediction:
                        democrat_and_correct += 1
                    else:
                        democrat_and_incorrect += 1
                #republicain
                if reality == 1:
                    if prediction:
                        republicain_and_correct += 1
                    else:
                        republicain_and_incorrect += 1

            # Calcule de l'éxactitude
            all_good_answers = republicain_and_correct + democrat_and_correct
            exactitude = all_good_answers / entries_quantity

            # Calcule de la précision
            somme_nb_republican = republicain_and_correct + democrat_and_incorrect
            precision = republicain_and_correct / somme_nb_republican

            # Calcule du rappel
            somme_reponse_republican = republicain_and_correct + republicain_and_incorrect
            rappel = republicain_and_correct / somme_reponse_republican

            print("\nDATASET : house-votes-84")
            print("METHODE : test\n")
            print("Matrice de confusion")
            print("          prediction ->  republicain | democrat")
            print("reality republicain:      ", republicain_and_correct, "         ", republicain_and_incorrect)
            print("reality democrat   :      ", democrat_and_incorrect, "          ", democrat_and_correct)
            print("\nL'éxactitude")
            print(exactitude)
            print("\nLa précision")
            print(precision)
            print("\nLe rappel")
            print(rappel)

        if dataset == 2:
            pass

# Vous pouvez rajouter d'autres méthodes et fonctions,
# il suffit juste de les commenter.