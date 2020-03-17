import numpy as np
import random

def load_iris_dataset(train_ratio):
    """Cette fonction a pour but de lire le dataset Iris

    Args:
        train_ratio: le ratio des exemples (ou instances) qui vont etre attribués à l'entrainement,
        le rest des exemples va etre utilisé pour les tests.
        Par exemple : si le ratio est 50%, il y aura 50% des exemple (75 exemples) qui vont etre utilisé
        pour l'entrainement, et 50% (75 exemples) pour le test.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy, train, train_labels, test, et test_labels
		
        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque 
        ligne dans cette matrice représente un exemple (ou instance) d'entrainement.
		
        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est le label (ou l'etiquette) pour l'exemple train[i]
        
        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque 
        ligne dans cette matrice représente un exemple (ou instance) de test.
		
        - test_labels : contient les labels (ou les étiquettes) pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est le label (ou l'etiquette) pour l'exemple test[i]
    """
    
    random.seed(1) # Pour avoir les meme nombres aléatoires à chaque initialisation.
    
    # Vous pouvez utiliser des valeurs numériques pour les différents types de classes, tel que :
    conversion_labels = {'Iris-setosa': 0, 'Iris-versicolor' : 1, 'Iris-virginica' : 2}
    
    # Le fichier du dataset est dans le dossier datasets en attaché 
    f = open('datasets/bezdekIris.data', 'r')

    # TODO : le code ici pour lire le dataset
    trainingFlowers = int(train_ratio*150)
    testingFlowers = 150 - trainingFlowers

    train_list = []
    train_labels_list = []
    test_list = []
    test_labels_list = []
    lineList = []

    for ds in f:
        if ds is not '\n':
            lineAttributes = ds.rsplit(',')
            nameWithoutTrailingNewLine = lineAttributes[4].rstrip()
            lineAttributes[4] = conversion_labels[nameWithoutTrailingNewLine]
            lineList.append(lineAttributes)

    random.shuffle(lineList)

    for i in range(0, trainingFlowers - 1):
        line = lineList[i]
        train_list.append([line[0], line[1], line[2], line[3]])
        train_labels_list.append(line[4])

    train = np.array(train_list)
    train_labels = np.array(train_labels_list)

    for i in range(trainingFlowers, (testingFlowers + testingFlowers - 1)):
        line = lineList[i]
        test_list.append([line[0], line[1], line[2], line[3]])
        test_labels_list.append(line[4])

    test = np.array(test_list)
    test_labels = np.array(train_labels_list)

    # REMARQUE très importante : 
	# remarquez bien comment les exemples sont ordonnés dans 
    # le fichier du dataset, ils sont ordonnés par type de fleur, cela veut dire que 
    # si vous lisez les exemples dans cet ordre et que si par exemple votre ration est de 60%,
    # vous n'allez avoir aucun exemple du type Iris-virginica pour l'entrainement, pensez
    # donc à utiliser la fonction random.shuffle pour melanger les exemples du dataset avant de séparer
    # en train et test.
       
    
    # Tres important : la fonction doit retourner 4 matrices (ou vecteurs) de type Numpy. 
    return (train, train_labels, test, test_labels)
	
	
	
def load_congressional_dataset(train_ratio):
    """Cette fonction a pour but de lire le dataset Congressional Voting Records

    Args:
        train_ratio: le ratio des exemples (ou instances) qui vont servir pour l'entrainement,
        le rest des exemples va etre utilisé pour les test.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy, train, train_labels, test, et test_labels
        
        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque 
        ligne dans cette matrice représente un exemple (ou instance) d'entrainement.
		
        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est le label (ou l'etiquette) pour l'exemple train[i]
        
        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque 
        ligne dans cette matrice représente un exemple (ou instance) de test.
		
        - test_labels : contient les labels (ou les étiquettes) pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est le label (ou l'etiquette) pour l'exemple test[i]
    """
    
    random.seed(1) # Pour avoir les meme nombres aléatoires à chaque initialisation.
    
    # Vous pouvez utiliser un dictionnaire pour convertir les attributs en numériques 
    # Notez bien qu'on a traduit le symbole "?" pour une valeur numérique
    # Vous pouvez biensur utiliser d'autres valeurs pour ces attributs
    conversion_labels = {'republican' : 0, 'democrat' : 1, 
                         'n' : 0, 'y' : 1, '?' : 2} 
    
    # Le fichier du dataset est dans le dossier datasets en attaché 
    f = open('datasets/house-votes-84.data', 'r')
	
    # TODO : le code ici pour lire le dataset

    training_congres = int(train_ratio * 435)
    testing_congres = 435 - training_congres

    train_list = []
    train_labels_list = []
    test_list = []
    test_labels_list = []
    line_list = []

    for ds in f:
        if ds is not '\n':
            lineAttributes = ds.rsplit(',')
            lineAttributes[16] = lineAttributes[16].rstrip()

            for i in range(0, 17):
                lineAttributes[i] = conversion_labels[lineAttributes[i]]

            line_list.append(lineAttributes)

    random.shuffle(line_list)

    for i in range(0, training_congres - 1):
        line = line_list[i]
        train_list.append([line[1], line[2], line[3], line[4], line[5], line[6], line[7], line[8], line[9], line[10], line[11], line[12], line[13], line[14], line[15], line[16]])
        train_labels_list.append(line[0])

    train = np.array(train_list)
    train_labels = np.array(train_labels_list)

    for i in range(training_congres, (testing_congres + testing_congres - 1)):
        line = line_list[i]
        test_list.append([line[1], line[2], line[3], line[4], line[5], line[6], line[7], line[8], line[9], line[10], line[11], line[12], line[13], line[14], line[15], line[16]])
        test_labels_list.append(line[0])

    test = np.array(test_list)
    test_labels = np.array(test_labels_list)

    # La fonction doit retourner 4 structures de données de type Numpy.
    return (train, train_labels, test, test_labels)
	

def load_monks_dataset(numero_dataset):
    """Cette fonction a pour but de lire le dataset Monks
    
    Notez bien que ce dataset est différent des autres d'un point de vue
    exemples entrainement et exemples de tests.
    Pour ce dataset, nous avons 3 différents sous problèmes, et pour chacun
    nous disposons d'un fichier contenant les exemples d'entrainement et 
    d'un fichier contenant les fichiers de tests. Donc nous avons besoin 
    seulement du numéro du sous problème pour charger le dataset.

    Args:
        numero_dataset: lequel des sous problèmes nous voulons charger (1, 2 ou 3 ?)
		par exemple, si numero_dataset=2, vous devez lire :
			le fichier monks-2.train contenant les exemples pour l'entrainement
			et le fichier monks-2.test contenant les exemples pour le test
        les fichiers sont tous dans le dossier datasets
    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy, train, train_labels, test, et test_labels
        
        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque 
        ligne dans cette matrice représente un exemple (ou instance) d'entrainement.
        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est le label (ou l'etiquette) pour l'exemple train[i]
        
        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque 
        ligne dans cette matrice représente un exemple (ou instance) de test.
        - test_labels : contient les labels (ou les étiquettes) pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est le label (ou l'etiquette) pour l'exemple test[i]
    """
	
	
	# TODO : votre code ici, vous devez lire les fichiers .train et .test selon l'argument numero_dataset
    str_numero_dataset = str(numero_dataset)
    str_monk_train = 'datasets/monks-' + str_numero_dataset + '.train'
    str_monk_test = 'datasets/monks-' + str_numero_dataset + '.test'
    f1 = open(str_monk_train, 'r')
    f2 = open(str_monk_test, 'r')

    train_list = []
    train_labels_list = []
    test_list = []
    test_labels_list = []
    line_list_train = []
    line_list_test = []

    for ds in f1:

        if ds is not '\n':
            lineAttributes = ds.rsplit(' ')
            lineAttributes[0] = lineAttributes[1]
            lineAttributes[1] = lineAttributes[2]
            lineAttributes[2] = lineAttributes[3]
            lineAttributes[3] = lineAttributes[4]
            lineAttributes[4] = lineAttributes[5]
            lineAttributes[5] = lineAttributes[6]
            lineAttributes[6] = lineAttributes[7]
            lineAttributes[8] = lineAttributes[8].rstrip()
            lineAttributes[7] = lineAttributes[8]
            lineAttributes.pop(8)
            line_list_train.append(lineAttributes)

    random.shuffle(line_list_train)

    for i in line_list_train:

        line = i
        train_list.append(
            [line[1], line[2], line[3], line[4], line[5], line[6], line[7]])
        train_labels_list.append(line[0])



    train = np.array(train_list)
    train_labels = np.array(train_labels_list)

    for ds in f2:

        if ds is not '\n':
            lineAttributes = ds.rsplit(' ')
            lineAttributes[0] = lineAttributes[1]
            lineAttributes[1] = lineAttributes[2]
            lineAttributes[2] = lineAttributes[3]
            lineAttributes[3] = lineAttributes[4]
            lineAttributes[4] = lineAttributes[5]
            lineAttributes[5] = lineAttributes[6]
            lineAttributes[6] = lineAttributes[7]
            lineAttributes[8] = lineAttributes[8].rstrip()
            lineAttributes[7] = lineAttributes[8]
            lineAttributes.pop(8)
            line_list_train.append(lineAttributes)

    random.shuffle(line_list_test)


    for i in line_list_test:
        line = i
        test_list.append(
            [line[1], line[2], line[3], line[4], line[5], line[6], line[7]])
        test_labels_list.append(line[0])

    test = np.array(test_list)
    test_labels = np.array(test_labels_list)

    # La fonction doit retourner 4 matrices (ou vecteurs) de type Numpy.
    return (train, train_labels, test, test_labels)