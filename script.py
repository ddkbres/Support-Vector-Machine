import codecademylib3_seaborn
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from svm_visualization import draw_boundary
from players import aaron_judge, jose_altuve, david_ortiz

fig, ax = plt.subplots()

#print(aaron_judge.columns)
#print(aaron_judge.description.unique())

def find_strike_zone(data_set):

  data_set['type'] = data_set['type'] .map({'S':1, 'B':2, 'X':3})

  #print(aaron_judge['plate_x'])

  data_set = data_set.dropna(subset = ['type','plate_x','plate_z'])

  #print(aaron_judge.type.unique())

  plt.scatter(x='plate_x', y='plate_z', c='type', alpha=0.25, cmap = plt.cm.coolwarm, data=data_set)

  training_set, validation_set = train_test_split(data_set, random_state = 1)

  largest = {'value':0, 'gamma':1, 'C':1}
  for gamma in range(1,5):
    for C in range(1,5):
      classifier = SVC(kernel = 'rbf', gamma = gamma, C = C)
      classifier.fit(training_set[['plate_x','plate_z']],training_set['type'])

      score = classifier.score(validation_set[['plate_x','plate_z']],validation_set['type'])

      if(score >  largest['value']):
        largest['value'] = score
        largest['gamma'] = gamma
        largest['C'] = C

  print(score)
  draw_boundary(ax, classifier)
  ax.set_ylim(-2, 6)
  ax.set_xlim(-3, 3)
  plt.show()

find_strike_zone(aaron_judge)
find_strike_zone(jose_altuve)
find_strike_zone(david_ortiz)

