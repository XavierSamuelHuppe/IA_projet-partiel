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



# Initializer/instanciez vos classifieurs avec leurs paramètres
classifieur_Knn_iris = Knn.Knn()
classifieur_Knn_congressional = Knn.Knn()
classifieur_Knn_monks_1 = Knn.Knn()
classifieur_Knn_monks_2 = Knn.Knn()
classifieur_Knn_monks_3 = Knn.Knn()

classifieur_bayes_naif_iris = BayesNaif.BayesNaif()
classifieur_bayes_naif_congressional = BayesNaif.BayesNaif()


# Charger/lire les datasets

dataset_iris = load_datasets.load_iris_dataset(0.7)
dataset_congressional = load_datasets.load_congressional_dataset(0.7)
dataset_monks_1 = load_datasets.load_monks_dataset(1)
dataset_monks_2 = load_datasets.load_monks_dataset(2)
dataset_monks_3 = load_datasets.load_monks_dataset(3)


# Entrainez votre classifieur

classifieur_Knn_iris.train(dataset_iris[0], dataset_iris[1], 0)
classifieur_Knn_congressional.train(dataset_congressional[0], dataset_congressional[1], 1)
classifieur_Knn_monks_1.train(dataset_monks_1[0], dataset_monks_1[1], 2)
classifieur_Knn_monks_2.train(dataset_monks_2[0], dataset_monks_2[1], 2)
classifieur_Knn_monks_3.train(dataset_monks_3[0], dataset_monks_3[1], 2)

classifieur_bayes_naif_iris.train(dataset_iris[0], dataset_iris[1], 0)
classifieur_bayes_naif_congressional.train(dataset_congressional[0], dataset_congressional[1], 1)


# Tester votre classifieur

classifieur_Knn_iris.test(dataset_iris[2], dataset_iris[3], 0)
classifieur_Knn_congressional.test(dataset_congressional[2], dataset_congressional[3], 1)
classifieur_Knn_monks_1.test(dataset_monks_1[2], dataset_monks_1[3], 2)
classifieur_Knn_monks_2.test(dataset_monks_3[2], dataset_monks_3[3], 2)
classifieur_Knn_monks_3.test(dataset_monks_3[2], dataset_monks_3[3], 2)

classifieur_bayes_naif_iris.test(dataset_iris[2], dataset_iris[3], 0)
classifieur_bayes_naif_congressional.test(dataset_congressional[2], dataset_congressional[3], 1)





