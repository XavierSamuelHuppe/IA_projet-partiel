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
a = load_datasets.load_iris_dataset(0.5)
b = load_datasets.load_congressional_dataset(0.5)
c = load_datasets.load_monks_dataset(2)
d = Knn.Knn()
d.train(c[0], c[1])

#print(a[0][2])
print(d.predict(c[0][2], 1, 5))


#print(c)

# Initializer/instanciez vos classifieurs avec leurs paramètres





# Charger/lire les datasets




# Entrainez votre classifieur





# Tester votre classifieur






