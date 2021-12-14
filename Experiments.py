from Dataset import LegoDataset
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
print(dataset.size())
#plot_svc(classifier, dataset.features, dataset.values)
plot_decision_regions(X=np.array([point[0] for point in dataset.validation_data_iter()]), 
                      y=np.array([point[1] for point in dataset.validation_data_iter()]),
                      clf=classifier, 
                      legend=2)

# Update plot object with X/Y axis labels and Figure Title
plt.xlabel('X', size=14)
plt.ylabel('Y', size=14)
plt.title('Región de decisión', size=16)
plt.show()