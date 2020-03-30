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
a = load_datasets.load_iris_dataset(0.5,)
b = load_datasets.load_congressional_dataset(0.5)
c = load_datasets.load_monks_dataset(2)

d = Knn.Knn()
d.train(a[0], a[1], 0)
d.test(a[2], a[3], 0)


e = Knn.Knn()
e.train(b[0], b[1], 1)
e.test(b[2], b[3], 1)

f = Knn.Knn()
f.train(c[0], c[1], 2)
f.test(c[2], c[3], 2)

# Initializer/instanciez vos classifieurs avec leurs paramètres





# Charger/lire les datasets




# Entrainez votre classifieur





# Tester votre classifieur






