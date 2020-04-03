1- Description des classes

	Classe entrainer_tester : La classe main qui va nous permettre de, en ordre, instancier
			les classifieurs avec leurs paramètres, charger et lire les datasets,
			entrainer les classifieurs et pour finir de tester nos classifieurs.

	Classe load_datasets : Cette classe nous permet de lire chacun des datasets ainsi que de
			les séparers entre les données utiliser pour entrainer nos classifieurs
			et les données pour tester nos claissifieurs.

	Classe Knn : La classe KNN est un classifieur qui utilise la technique des K plus proches
			voisins pour s'entrainer et tester les données. La méthode Train entraine
			le classifieur. La méthode Test teste le classifieur sur les données tests.
			Et la méthode predict nous permet de vérifier si un exemple donné appartient bien à
			a une classe spécifique selon l'entrainement que le classifieur a eu. Cette
			méthode predict est utilisé par les autres méthodes train et test de la classe.

	Classe BayesNaif : La classe BayesNaif est un classifieur qui utilise la technique Bayes naif
			pour s'entrainer et tester les données. La méthode Train entraine
			le classifieur. La méthode Test teste le classifieur sur les données tests.
			Et la méthode predict nous permet de vérifier si un exemple donné appartient bien à
			a une classe spécifique selon l'entrainement que le classifieur a eu. Cette
			méthode predict est utilisé par les autres méthodes train et test de la classe.

2- Répartition des tâches de travail
	
	Olivier-Marc Gravier : Classe Knn, classe load_datasets, classe entrainer_tester,
			Interprétation des données

	
	Xavier Samuel Morisset-Huppé : Classe Bayes Naif, classe load_datasets, Interprétation
			des données

3- Difficultés rencontrées

	1- Difficulté/Impossibilité de se rencontrer en personne suite aux événements 
	liés au covid-19.

	2- Petite difficulté au départ à comprendre le format du code car, aux finales,
	il fallait faire trois algoritmhes tous un peu différents. Il aurait été mieux
	d'avoir le code divisé en trois cylindres (pour chaques dataset) plutôt que 
	d'avoir à faire 3 algorithmes dans chaque méthodes.