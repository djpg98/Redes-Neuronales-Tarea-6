from Dataset import LegoDataset
from metrics import accuracy
from sklearn import svm
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
import numpy as np
import sys
import math

#Primer argumento: Valor de C
#Segundo argumento: Kernel: ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, 
#Tercer argumento: Dependiendo del kernel, puede ser el grado ()

dataset = LegoDataset('lego.csv')

if sys.argv[2] == 'poly':
    classifier = svm.SVC(kernel='poly', C=float(sys.argv[1]), degree=float(sys.argv[3]))
elif sys.argv[2] == 'rbf':
    gamma = 1 / (2 * math.pow(float(sys.argv[3]), 2))
    classifier = svm.SVC(kernel='rbf', C=float(sys.argv[1]), gamma=gamma)

classifier.fit(dataset.features, dataset.values)
print('Classifier is ready')
#plot_svc(classifier, dataset.features, dataset.values)
plot_decision_regions(X=np.array([point[0] for point in dataset.validation_data_iter()]), 
                      y=np.array([point[1] for point in dataset.validation_data_iter()]),
                      clf=classifier, 
                      legend=2)

# Update plot object with X/Y axis labels and Figure Title
plt.xlabel('X', size=14)
plt.ylabel('Y', size=14)
plt.title('Región de decisión', size=16)
#plt.show()
plt.savefig(f'svm_{sys.argv[2]}_{sys.argv[3]}_C{sys.argv[1]}.png')
print(f'Plot saved as svm_{sys.argv[2]}_{sys.argv[3]}_C{sys.argv[1]}.png')

errors = sum(map(abs, classifier.predict(dataset.features) - dataset.values))

with open(f'svm_{sys.argv[2]}_{sys.argv[3]}_C{sys.argv[1]}.txt', 'w') as report:

    report.write('REPORTE SVM\n\n')
    report.write(f'Kernel: {sys.argv[2]}\n')
    if sys.argv[2] == 'poly':
        report.write(f'Grado: {sys.argv[3]}\n')
    else:
        report.write(f'Dispersión: {sys.argv[3]}\n')
    report.write(f'C = {sys.argv[1]}\n\n')

    report.write(f'Número de vectores de soporte: {classifier.support_.size}\n')
    report.write(f'Accuracy: {accuracy(dataset.size(), errors)}\n')

print(f'Report saved as svm_{sys.argv[2]}_{sys.argv[3]}_C{sys.argv[1]}.txt')
