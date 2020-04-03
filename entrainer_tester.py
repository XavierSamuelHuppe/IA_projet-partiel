import numpy as np
import sys
import load_datasets
import BayesNaif # importer la classe du classifieur bayesien
import Knn # importer la classe du Knn
#importer d'autres fichiers et classes si vous en avez développés


"""
C'est le fichier main duquel nous allons tout lancer
Vous allez dire en commentaire c'est quoi les paramètres que vous avez utilisés
En gros, vous allez :
1- Initialiser votre classifieur avec ses paramètres
2- Charger les datasets
3- Entrainer votre classifieur
4- Le tester

"""

# Initializer vos paramètres
train_pourcentage_iris = 0.7
train_pourcentage_congressional = 0.7

num_datset_iris = 0
num_dataset_congressional = 1
num_dataset_monks = 2

# Initializer/instanciez vos classifieurs avec leurs paramètres
classifieur_Knn_iris = Knn.Knn()
classifieur_Knn_congressional = Knn.Knn()
classifieur_Knn_monks_1 = Knn.Knn()
classifieur_Knn_monks_2 = Knn.Knn()
classifieur_Knn_monks_3 = Knn.Knn()

classifieur_bayes_naif_iris = BayesNaif.BayesNaif()
classifieur_bayes_naif_congressional = BayesNaif.BayesNaif()
classifieur_bayes_naif_monks_1 = BayesNaif.BayesNaif()
classifieur_bayes_naif_monks_2 = BayesNaif.BayesNaif()
classifieur_bayes_naif_monks_3 = BayesNaif.BayesNaif()

dataset_iris = load_datasets.load_iris_dataset(train_pourcentage_iris)
dataset_congressional = load_datasets.load_congressional_dataset(train_pourcentage_congressional)
dataset_monks_1 = load_datasets.load_monks_dataset(1)
dataset_monks_2 = load_datasets.load_monks_dataset(2)
dataset_monks_3 = load_datasets.load_monks_dataset(3)


#Entrainez votre classifieur
print("\n")
print("Entrainement du classifieur KNN avec le dataset Iris et un pourcentage d'entrainement de " + str(train_pourcentage_iris))
classifieur_Knn_iris.train(dataset_iris[0], dataset_iris[1], num_datset_iris)
print("---------------------------------------------------------\n")
print("Entrainement du classifieur KNN avec le dataset Congressional et un pourcentage d'entrainement de " + str(train_pourcentage_congressional))
classifieur_Knn_congressional.train(dataset_congressional[0], dataset_congressional[1], num_dataset_congressional)
print("---------------------------------------------------------\n")
print("Entrainement du classifieur KNN avec le dataset Monks_1")
classifieur_Knn_monks_1.train(dataset_monks_1[0], dataset_monks_1[1], num_dataset_monks)
print("---------------------------------------------------------\n")
print("Entrainement du classifieur KNN avec le dataset Monks_2")
classifieur_Knn_monks_2.train(dataset_monks_2[0], dataset_monks_2[1], num_dataset_monks)
print("---------------------------------------------------------\n")
print("Entrainement du classifieur KNN avec le dataset Monks_3")
classifieur_Knn_monks_3.train(dataset_monks_3[0], dataset_monks_3[1], num_dataset_monks)

print("---------------------------------------------------------\n")
print("Entrainement du classifieur Bayes Naif avec le dataset Iris et un pourcentage d'entrainement de " + str(train_pourcentage_iris))
classifieur_bayes_naif_iris.train(dataset_iris[0], dataset_iris[1], 0)
print("---------------------------------------------------------\n")
print("Entrainement du classifieur Bayes Naif avec le dataset Congressional et un pourcentage d'entrainement de " + str(train_pourcentage_congressional))
classifieur_bayes_naif_congressional.train(dataset_congressional[0], dataset_congressional[1], 1)
print("---------------------------------------------------------\n")
print("Entrainement du classifieur Bayes Naif avec le dataset Monks_1")
classifieur_bayes_naif_monks_1.train(dataset_monks_1[0], dataset_monks_1[1], 2)
print("---------------------------------------------------------\n")
print("Entrainement du classifieur Bayes Naif avec le dataset Monks_2")
classifieur_bayes_naif_monks_2.train(dataset_monks_2[0], dataset_monks_2[1], 2)
print("---------------------------------------------------------\n")
print("Entrainement du classifieur Bayes Naif avec le dataset Monks_3")
classifieur_bayes_naif_monks_3.train(dataset_monks_3[0], dataset_monks_3[1], 2)
print("---------------------------------------------------------\n")

#Tester votre classifieur

print("Test du classifieur KNN avec le dataset Iris et un pourcentage de test de" + str(1 - train_pourcentage_iris))
classifieur_Knn_iris.test(dataset_iris[2], dataset_iris[3], num_datset_iris)
print("---------------------------------------------------------\n")
print("Test du classifieur KNN avec le dataset Congressional et un pourcentage de test de " + str(1 - train_pourcentage_congressional))
classifieur_Knn_congressional.test(dataset_congressional[2], dataset_congressional[3], num_dataset_congressional)
print("---------------------------------------------------------\n")
print("Test du classifieur KNN avec le dataset Monks_1")
classifieur_Knn_monks_1.test(dataset_monks_1[2], dataset_monks_1[3], num_dataset_monks)
print("---------------------------------------------------------\n")
print("Test du classifieur KNN avec le dataset Monks_2")
classifieur_Knn_monks_2.test(dataset_monks_2[2], dataset_monks_2[3], num_dataset_monks)
print("---------------------------------------------------------\n")
print("Test du classifieur KNN avec le dataset Monks_3")
classifieur_Knn_monks_3.test(dataset_monks_3[2], dataset_monks_3[3], num_dataset_monks)

print("---------------------------------------------------------\n")
print("Test du classifieur Bayes Naif avec le dataset Iris et un pourcentage de test de " + str(1 - train_pourcentage_iris))
classifieur_bayes_naif_iris.test(dataset_iris[2], dataset_iris[3], 0)
print("---------------------------------------------------------\n")
print("Test du classifieur Bayes Naif avec le dataset Congressional et un pourcentage de test de " + str(1 - train_pourcentage_congressional))
classifieur_bayes_naif_congressional.test(dataset_congressional[2], dataset_congressional[3], 1)
print("---------------------------------------------------------\n")
print("Test du classifieur Bayes Naif avec le dataset Monks_1")
classifieur_bayes_naif_monks_1.test(dataset_monks_1[2], dataset_monks_1[3], 2)
print("---------------------------------------------------------\n")
print("Test du classifieur Bayes Naif avec le dataset Monks_2")
classifieur_bayes_naif_monks_2.test(dataset_monks_2[2], dataset_monks_2[3], 2)
print("---------------------------------------------------------\n")
print("Test du classifieur Bayes Naif avec le dataset Monks_3")
classifieur_bayes_naif_monks_3.test(dataset_monks_3[2], dataset_monks_3[3], 2)

