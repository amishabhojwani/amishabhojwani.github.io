---
title: 'Mushrooms are fun (but deadly)'
date: 2022-03-17
permalink: /posts/2022/03-mushrooms
tags:
  - kaggle
  - machine learning
  - one hot encoding
---

In this blogpost I use a popular kaggle dataset of mushroom traits to build class predictive models to determine if a mushroom is poisonous or edible.

## Setting the scene

Picture this: it has been raining all week and you finally get the perfect weekend to go mushroom foraging. Your mushrooming skills aren't great, but you really want some fresh mushrooms to make a decadent mushroom risotto. The trouble is, will you pick the right mushrooms?

![A purple mushroom](/images/purple_mushy.jpg 'A purple mushroom')

Let's try and train a model to pick the poisonous mushrooms out for us. We don't trust our mushroom judgement. We can work with this dataset from Kaggle [[1](https://www.kaggle.com/datasets/uciml/mushroom-classification)], describing 23 mushroom species from the Lepiota and Agaricus families. 

## What does the data look like?

Let's first import some packages that will be useful in investigating and manipulating the data:

```
# Cleaning and exploring data
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```

Our dataset is labelled, meaning we know if the recorded mushrooms are poisonous or edible. Here's a sneak peak of the data:

![Head of dataset](/amishabhojwani.github.io/_posts/post_images/mushies_head.jpg 'Head of dataset')

|    |class|cap-shape|cap-surface|cap-color|bruises|odor|gill-attachment|gill-spacing|gill-size|gill-color|stalk-shape|stalk-root|stalk-surface-above-ring|stalk-surface-below-ring|stalk-color-above-ring|stalk-color-below-ring|veil-type|veil-color|ring-number|ring-type|spore-print-color|population|habitat|
|----|-----|---------|-----------|---------|-------|----|---------------|------------|---------|----------|-----------|----------|------------------------|------------------------|----------------------|----------------------|---------|----------|-----------|---------|-----------------|----------|-------|
|0   |p    |x        |s          |n        |t      |p   |f              |c           |n        |k         |e          |e         |s                       |s                       |w                     |w                     |p        |w         |o          |p        |k                |s         |u      |
|1   |e    |x        |s          |y        |t      |a   |f              |c           |b        |k         |e          |c         |s                       |s                       |w                     |w                     |p        |w         |o          |p        |n                |n         |g      |
|2   |e    |b        |s          |w        |t      |l   |f              |c           |b        |n         |e          |c         |s                       |s                       |w                     |w                     |p        |w         |o          |p        |n                |n         |m      |
|3   |p    |x        |y          |w        |t      |p   |f              |c           |n        |n         |e          |e         |s                       |s                       |w                     |w                     |p        |w         |o          |p        |k                |s         |u      |
|4   |e    |x        |s          |g        |f      |n   |f              |w           |b        |k         |t          |e         |s                       |s                       |w                     |w                     |p        |w         |o          |e        |n                |a         |g      |

This looks a bit messy. After cleaning up labels by matching them to the data dictionary, we're able to understand the data better. For example, the first column is class with labels 'p' and 'e' standing for 'poisonous' and 'edible' respectively, as seen in the data dictionary on Kaggle. Matching these labels with their value in the dictionary for every column helps us with exploring the data visually. Let's create some plots to look at each mushroom trait by class:

![EDA](post_images/EDA.jpg 'Exploratory data analysis')

These are only some of the plots we can create, but they tell us some important things:

- There is only one viel type and it is distributed equally between classes; it is a column that can be dropped.
- There seems to be a big difference in the amount of bruising between poisonous and non-poisonous mushrooms.
- Odor could be a reliable predictor of mushroom class as it shows edible and poisonous mushrooms tend to have different smells.
- Gill size is generally broader in edible mushrooms and narrower in poisonous. 
- A buff gill colour could be a good indicator of a poisonous mushroom.
- Surface above and below stalk might be able to be differentiated between mushroom classes.
- Spore print colour is quite differentiated between classes.

One peculiarity about this data is that for the stalk root trait, one of the possible labels is: 'missing'. This is probably because it was not recorded for that mushroom, or maybe because there was no stalk root present on the sample. We can impute these values with K-Nearest Neighbour clustering (look at my github repo for an example with this data).

Lets summarise the quality of our data so far:

![Key statistics mushies](post_images/mushies_keystats.jpg 'Key statistics in the data for modelling')

Our data look very clean now, it may be time for some modelling!

## Let's get down to business

So far we have only needed `pandas`, `matplotlib.pyplot` and `seaborn` to clean and visualise our data. Now that we're moving on to feature engineering and modelling, let's define some more imports:

``` 
# Modelling
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# Evaluating models
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn import tree
``` 
Lets see what our features and target variable are:

```
target = 'class'
features = [feature for feature in list(mushroom_data.columns) if feature not in target]

print(features)
```
```
['cap_shape', 'cap_surface', 'cap_color', 'bruises', 'odor', 'gill_attachment', 'gill_spacing', 'gill_size', 'gill_color', 'stalk_shape', 'stalk_root', 'stalk_surface_above_ring', 'stalk_surface_below_ring', 'stalk_color_above_ring', 'stalk_color_below_ring', 'veil_color', 'ring_number', 'ring_type', 'spore_print_color', 'population', 'habitat', 'stalk_imputed']
```

Thats a lot of features! But it's okay, python can handle it. We next have to seperate our data (column-wise), into features (X) and target (y) and divide it (row-wise) so that we can test and train our model. Splitting the data in this way helps us test and validate the 'goodness' of our model, it helps us see that it's actually doing what we want! More on that later.

```
# Define X and y 
X = mushroom_data[features]
y = mushroom_data[target]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.35, random_state = 40)
```

Cool. We now how everything we need to start modelling, except that statistical and Machine Learning models handle data in a numerical format, not in strings like ours is. To coerce or transform our data into numerical there are various options:

- `sklearn.preprocessing.OrdinalEncoder` - establishes and ordinal relationship between labels in a variable and gives them numerical values accordingly.
- `sklearn.preprocessing.LabelEncoder` - can assign numbers to category labels in a variable randomly or in alphabetical order.
- `pd.get_dummies` a.k.a. One Hot Encoder - attributes each label in a category its own column, meaning that every observation has two possible values: either that label corresponds to them (1) or it doesn't (0). This augments the dimensions of our data considerably if we have columns with high cardinality (many of unique labels).

There is no ordinality in our mushroom features, so we don't want to choose the first option. As for the second, it's very possible that an ordinal relationship in the data is implied with how the numbers are assigned to labels. To avoid this, the third option is the best.

One Hot Encoding (OHE) could create errors in some models if we aren't careful. Especially in parametric Machine Learning, you want to avoid features being correlated with one another. If we work through an example it will be easier to understand why. The mushroom trait 'class' has two possible categories: 'poisonous' or 'edible'. Suppose we had 5 observations of mushrooms with two possible gill sizes. OHE encoding this feature would mean we have two new columns, one for each of the types of gill size:

| Observation | Gill_size  | Gill_size_broad  | Gill_size_narrow  |
|-------------|--------|------------------|---------------|
| 1           | broad | 1             | 0             |
| 2           | broad | 1             | 0             |
| 3           | narrow | 0             | 1             |
| 4           | broad | 1             | 0             |
| 5           | narrow | 0             | 1             |

One of the OHE can say exactly the same thing as the other, only the numbers for the labels are reversed. This means the columns are dependent on each other, and they could be very correlated. We want features in our model not be multicollinear, but we do want them to correlate with our target. In this case both of these features correlate perfectly with each other (see the heatmap below), which means we must drop one.

![OHE feature heatmap](post_images/ohe_example.png 'Heatmap of OHE gill size')

Dropping one of the columns created for each feature is repeated for every feature when we do any OHE, except for when the models in question are working with Euclidean space. That get's a bit more tricky, but we can talk about that another day.

OHE features on python is a breeze, even if we have so many variables, it's a simple line of code for each data frame:

```
# Do this for X_train and X_test
X_train_eng = pd.get_dummies(X_train, columns = features, drop_first = True)

# Do this for y_train and y_test
y_train_eng = y_train.map({'edible':0, 'poisonous':1})
```
The last argument in the `get_dummies` function is what helps us eliminate one column for every feature. `.map` will do the same binary transformation for our target variable. With our data now in numerical format, we can start modelling!

### To choose or not to choose a model

Traditionally, mushrooms are identified using mushroom identification guides (actually it's just an obsessive amount of practice, but you have to start somewhere). These guides are much like decision trees, where a question is asked about a mushroom trait, which then takes us down a new line of questioning to identify the type of mushroom, and thus, if it's poisonous. Because of this, I think the most fitting algorithm to run would be a tree-based one. Let's start with running a decision tree. From there we can determine which features contribute the most in the model's decision making power, which will help me see if the the features I thought were interesting when exploring the data are actually useful to predict whether or not a mushroom is poisonous.

```
# Declare and fit the DecisionTree
dt = DecisionTreeClassifier(random_state = 42)
dt.fit(X_train_eng, y_train_eng)
```

From here we can test our models performance on both the training data and the test data. Lets see how we did:

```
###################### TRAIN SET ######################

               precision    recall  f1-score   support

           0       1.00      1.00      1.00      2738
           1       1.00      1.00      1.00      2542

    accuracy                           1.00      5280
   macro avg       1.00      1.00      1.00      5280
weighted avg       1.00      1.00      1.00      5280
 


###################### TEST SET ######################

               precision    recall  f1-score   support

           0       1.00      1.00      1.00      1470
           1       1.00      1.00      1.00      1374

    accuracy                           1.00      2844
   macro avg       1.00      1.00      1.00      2844
weighted avg       1.00      1.00      1.00      2844
```

Um... It seems we have perfect scores! I am especially interested in recall because it will minimise false negatives (FN). In other words, we don't want to eat a mushroom based on the prediction that it's edible when it's actually posionous... things could go *very* wrong.


|           | CONFUSION MATRIX |          |
|-----------|----------|----------|
|           | Actually Poisonous | Actually Edible   |
| Predicted Poisonous | TP        | FP       |
| Predicted Edible    | FN        | TN       |


From the evaluation metrics printed above, we know that this decision tree is reliable (completely 100% for some bizarre reason that we have absolutely identified). Let's have a look at what features are most useful in accurately predicting poisonous mushrooms in our `feat_imp` table:

![Feature importances](post_images/feature_imp.jpg 'Feature importances from a Decision Tree')

The importances are calculated using a metric called Gini Impurity, which returns a normalised probability of misclassification for each variable (see documentation here [[2](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier.feature_importances_)]). Unsurprisingly, a lack of odor is a useful feature in the model (the most useful), and it holds over 60% of the predictve power in identifying poisonous mushrooms. We also see a few of the other features we thought would be interesting,  which means our EDA was very productive.

Let's make a new tree with only the top 12 most important features, since they represent almost 100% of the predictive power of the model. A model that is still good but has less features is more valuable than one with lots. It's always good to remember the KISS rule (Keep it Simple, Stupid). 

```
# Select features of interest in training and test sets
select_interesting = list(feat_imp['Feature'][:12])
X_train_important = X_train_eng[select_interesting]
X_test_important = X_test_eng[select_interesting]

# Declare and fit the Decision Tree
dt_important = DecisionTreeClassifier(random_state = 42)
dt_important.fit(X_train_important, y_train_eng)
```

I won't show you explicitly here again, but we get the exact same perfect evaluation scores output as our last model (with less features!). This is great, we can even see what these decisions look like now:

![Decision Tree](post_images/important_dt.png 'Decision Tree')

This means using this decision tree will tell us if a mushroom is poisonous after asking at most 9 questions about the mushroom traits, thats pretty good!

### Let's think about good practice

For the sake of being thorough, I was having a look at the practicality of having over 100 OHE features like we do in our first model, specifically for tree-based algorithms. OHE categorical features used in training decision trees is thought to be unwise (for discussions on using sparse data for modelling, see: [[3]](https://stats.stackexchange.com/questions/390671/random-forest-regression-with-sparse-data-in-python/430127#430127) [[4]](https://stats.stackexchange.com/questions/231285/dropping-one-of-the-columns-when-using-one-hot-encoding)), so lets have a look at the evaluation output of a logistic regression on the same features, for peace of mind.

```
###################### TRAIN SET ######################

               precision    recall  f1-score   support

           0       0.99      0.99      0.99      2738
           1       0.99      0.99      0.99      2542

    accuracy                           0.99      5280
   macro avg       0.99      0.99      0.99      5280
weighted avg       0.99      0.99      0.99      5280
 


###################### TEST SET ######################

               precision    recall  f1-score   support

           0       0.99      0.99      0.99      1470
           1       0.99      0.99      0.99      1374

    accuracy                           0.99      2844
   macro avg       0.99      0.99      0.99      2844
weighted avg       0.99      0.99      0.99      2844
```

Hmm... There's no difference between our test and train scores for this model either, which means our model is performing well, but the scores arent as perfect as our decision tree. Even the littlest bit of error in our recall could be fatal, implying the model is misclassfying poisonous mushrooms.

It's still surprising that a logistic regression performed worse than a decision tree for this data, when we know that's the type of linear modelling OHE was devised for! I would personally use the decision tree if you're out mushroom picking (or mushrooming, as the cool kids say), especially if you're unsure about your foraging skills!