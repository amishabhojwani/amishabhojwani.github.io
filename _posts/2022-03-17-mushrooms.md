---
title: 'Mushrooms are fun (but deadly)'
date: 2022-03-17
permalink: /posts/mushrooms
tags:
  - kaggle
  - machine learning
  - one hot encoding
---

Picture this: it has been raining all week and you finally get the perfect weekend to go mushroom foraging. Your mushrooming skills aren't great, but you really want some fresh mushrooms to make a decadent mushroom risotto. The trouble is, will you pick the right mushrooms?

![A purple mushroom](images/purple_mushy.jpg 'A purple mushroom')

Let's try and train a model to pick the poisonous mushrooms out for us. We don't trust our mushroom judgement. We can work with this dataset from Kaggle [[1](https://www.kaggle.com/datasets/uciml/mushroom-classification)], describing 23 mushroom species from the Lepiota and Agaricus families. Our dataset is labelled, meaning we know if the recorded mushrooms are poisonous or edible. Here's a sneak peak of the data after importing some packages:

![Head of dataset](images/mushies_head.jpg 'Head of dataset')

This looks a bit messy. After cleaning up labels by matching them to the data dictionary, we're able to understand the data better. For example, the first column was class with labels 'p' for 'poisonous' and 'e' for 'edible'. Matching these labels for every column helps us with exploring the data visually. Let's create some plots to look at each mushroom trait by