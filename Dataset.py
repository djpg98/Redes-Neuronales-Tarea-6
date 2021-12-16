import csv
import random
import numpy as np

""" Conjunto de funciones básicas que debe tener un dataset """
class DatasetMixin:

    """ Añade una componente de sesgo a cada dato en el dataset. Esto se logra
        agregando una componente adicional que siempre vale 1 a todos los datos 
        en el dataset.
    """
    def add_bias_term(self):

        for i in range(len(self.features)):

            self.features[i] = np.concatenate((np.ones(1), self.features[i]))

        self.features = np.array(self.features)
        self.values = np.array(self.values)

    """ Devuelve la cantidad de elementos en el dataset """
    def size(self):

        return len(self.features)

    """ Devuelve la cantidad de elementos pertenecientes
        al conjunto de entrenamiento del dataset """
    def training_data_size(self):

        return len(self.training_data)

    """ Devuelve la cantidad de elementos pertenecientes
        al conjunto de validación del dataset """
    def test_data_size(self):

        return len(self.test_data)

    """ Devuelve el tamaño del input vector (Incluendo el término de bias si
        este ha sido agregado
    """
    def feature_vector_length(self):

        return len(self.features[0])

    """ Iterador para todos los elementos del dataset """
    def __iter__(self):

        for pair in zip(self.features, self.values):

            yield pair

    """ Iterador para el conjunto de datos de entrenamiento
        del dataset
    """
    def training_data_iter(self):

        for index in self.training_data:

            yield (self.features[index], self.values[index])

    """ Iterador para el conjunto de datos de validación
        del dataset
    """
    def test_data_iter(self):

        for index in self.test_data:

            yield (self.features[index], self.values[index])

    def training_data_arrays(self, zipped=False):

        feature_list = []
        value_list = []

        for feature, value in self.training_data_iter():

            feature_list.append(feature)
            value_list.append(value)

        if zipped:
            return np.array(zip(feature_list, value_list))
        else:
            return np.array(feature_list), np.array(value_list)

    def test_data_arrays(self, zipped=False):

        feature_list = []
        value_list = []

        for feature, value in self.test_data_iter():

            feature_list.append(feature)
            value_list.append(value)

        if zipped:
            return np.array(zip(feature_list, value_list))
        else:
            return np.array(feature_list), np.array(value_list)

    """ Altera aleatoriamente el orden en que se iteran los elementos del
        conjunto de datos de entrenamiento
    """
    def shuffle_training_data(self):
        random.shuffle(self.training_data)

class LegoDataset(DatasetMixin):

    """ Constructor de la clase:
        Parámetros:
            - datafile: Archivo csv de donde se extrae la información

    """
    def __init__(self, datafile):

        self.features = []
        self.values = []

        with open(datafile, 'r') as csv_file:

            data_reader = csv.reader(csv_file, delimiter=",")

            for row in data_reader:

                features, value = row[:-1], row[-1:][0]

                self.features.append(np.array(list(map(float, features))))
                self.values.append(int(value))

            csv_file.close()

        self.features = np.array(self.features)
        self.values = np.array(self.values)
        index_list = [i for i in range(len(self.features))]
        self.training_data = random.sample(index_list, int(0.80 * len(self.features)))
        self.test_data = [index for index in index_list if index not in self.training_data]
        self.visualization_data = random.sample(index_list, int(0.10 * len(self.features)))

    def visualization_data_iter(self):

        for index in self.visualization_data:

            yield (self.features[index], self.values[index])

    def visualization_data_arrays(self, zipped=False):

        feature_list = []
        value_list = []

        for feature, value in self.visualization_data_iter():

            feature_list.append(feature)
            value_list.append(value)

        if zipped:
            return np.array(zip(feature_list, value_list))
        else:
            return np.array(feature_list), np.array(value_list)