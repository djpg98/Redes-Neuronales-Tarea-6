from Dataset import LegoDataset
from metrics import accuracy, precision
from sklearn import svm
from mlxtend.plotting import plot_decision_regions
from collections import Counter
import matplotlib.pyplot as plt
import sys
import math

#Primer argumento: Valor de C
#Segundo argumento: Kernel: ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, 
#Tercer argumento: Dependiendo del kernel, puede ser el grado ('poly') o dispersión ('rbf')

dataset = LegoDataset('lego.csv')

#Inicializar clasificador dependiendo del kernel
if sys.argv[2] == 'poly':
    classifier = svm.SVC(kernel='poly', C=float(sys.argv[1]), degree=float(sys.argv[3]), coef0=1)
elif sys.argv[2] == 'rbf':
    #Convertir el parámetro de dispersión en gamma
    gamma = 1 / (2 * math.pow(float(sys.argv[3]), 2))
    classifier = svm.SVC(kernel='rbf', C=float(sys.argv[1]), gamma=gamma)

#Arrays de entrenamiento
training_features, training_values = dataset.training_data_arrays()
#Entrenamiento
classifier.fit(training_features, training_values)
print('Classifier is ready')
#plot_svc(classifier, dataset.features, dataset.values)

#Arrays de visualización y generar el plot
visualization_features, visualization_values = dataset.visualization_data_arrays()
plot_decision_regions(X=visualization_features, 
                      y=visualization_values,
                      clf=classifier, 
                      legend=2)

# Update plot object with X/Y axis labels and Figure Title
plt.xlabel('X', size=14)
plt.ylabel('Y', size=14)
plt.title('Región de decisión', size=16)
#plt.show()
plt.savefig(f'Results_{sys.argv[2]}/svm_{sys.argv[2]}_{sys.argv[3]}_C{sys.argv[1]}.png')
print(f'Plot saved as svm_{sys.argv[2]}_{sys.argv[3]}_C{sys.argv[1]}.png')

#Cálculo de cantidad de errores y precisión
total_train = Counter(training_values)
dif_train = classifier.predict(training_features) - training_values

false_pos = Counter(dif_train)
errors_training = sum(map(abs, dif_train))
prec_train = (precision(total_train[0] - false_pos[1], false_pos[-1]), precision(total_train[1] - false_pos[-1], false_pos[1]))

#Test arrays
test_features, test_values = dataset.test_data_arrays()

total_test = Counter(test_values)
#Test
dif_test = classifier.predict(test_features) - test_values

false_pos = Counter(dif_test)
errors_test = sum(map(abs, dif_test))
prec_test = (precision(total_test[0] - false_pos[1], false_pos[-1]), precision(total_test[1] - false_pos[-1], false_pos[1]))

#Escribir reporte
with open(f'Results_{sys.argv[2]}/svm_{sys.argv[2]}_{sys.argv[3]}_C{sys.argv[1]}.txt', 'w') as report:

    report.write('REPORTE SVM\n\n')
    report.write(f'Kernel: {sys.argv[2]}\n')
    if sys.argv[2] == 'poly':
        report.write(f'Grado: {sys.argv[3]}\n')
    else:
        report.write(f'Dispersión: {sys.argv[3]}\n')
    report.write(f'C = {sys.argv[1]}\n\n')

    report.write(f'Número de vectores de soporte: {classifier.support_.size}\n')
    report.write(f'Accuracy (Training): {accuracy(dataset.training_data_size(), errors_training)}\n')
    report.write(f'Precision (Train):\n')
    report.write(f'\t- Class 0: {prec_train[0]}\n')
    report.write(f'\t- Class 1: {prec_train[1]}\n')
    report.write(f'Accuracy (Test): {accuracy(dataset.test_data_size(), errors_test)}\n')
    report.write(f'Precision (Test):\n')
    report.write(f'\t- Class 0: {prec_test[0]}\n')
    report.write(f'\t- Class 1: {prec_test[1]}\n')


print(f'Report saved as svm_{sys.argv[2]}_{sys.argv[3]}_C{sys.argv[1]}.txt')
