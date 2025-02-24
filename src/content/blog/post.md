---
title: 'De Zéro à ChatGpt, Part 1'
excerpt: "Cette série d'articles est dédiée à la création d’un framework de deep learning entièrement fait maison ! Notre objectif final : bâtir une base suffisamment solide pour nous permettre d'implémenter l'architecture Transformer. Dans ce premier article, nous allons plonger dans les fondations, en développant un simple réseau de neurones avec Numpy et Python. "
publishDate: 'Nov 12 2024'
tags:
  - Guide
seo:
  image:
    src: '/p1.jpg'
    alt: Neural Network
isFeatured: false
---



![Neural Network](/RNN_Tutorial.avif)



## Introduction


Si vous n’avez pas passé la dernière décennie enfermé dans une cave, vous avez sûrement entendu parler de termes comme "Intelligence Artificielle", "Deep Learning" ou encore "Réseaux de neurones". Et si, comme moi, vous étudiez l’informatique, je parie que vous en avez entendu parler plus d’une fois.



Bien que ces concepts soient devenus familiers pour beaucoup, leur fonctionnement interne reste un mystère pour de nombreuses personnes. C'était d’ailleurs le cas pour moi avant de me lancer dans le projet de créer un réseau de neurones complexe, et ce, sans utiliser de bibliothèque de deep learning telle que PyTorch ou TensorFlow.



Alors, si vous êtes assez curieux pour plonger dans les profondeurs de l’intelligence artificielle et faire partie de ceux qui sauront l’affronter lors du soulèvement des machines… c’est parti !



## Le neuronne (ou perceptron)

Pour comprendre le deep learning et notamment les réseaux de neurones, il est primordial de s'intéresser à un élément clé de son fonctionnement : le neurone ! J'imagine que tu connais le terme neurone ; eh bien, le neurone en informatique est appelé ainsi car son fonctionnement tend à se rapprocher de celui du neurone biologique : il prend en entrée différentes informations et en ressort une unique.



Voici à quoi ressemble un neurone en intelligence artificielle (on utilisera le terme perceptron par la suite, qui est le terme scientifique) :

![perceptron](/perceptron.png)

Son fonctionnement est simple :
- Le perceptron prend plusieurs valeurs en entrée.
- Chaque valeur est multipliée par son poids, noté wi
- Les valeurs pondérées sont ensuite additionnées entre elles.
- Enfin, une fonction dite d'activation est appliquée au résultat.



Mathematiquement parlant un perceptron est simplement une fonction de cette forme : 

f(x * w0 + y * w1 + z * w2 + biais) = output 

avec f la fonction d'activation. (à adapter en fonction du nombres d'entrée )



Bon, alors c'est super : nous avons une simple fonction mathématique, mais comment cela pourrait-il nous permettre de "prendre le contrôle du monde" et de créer une intelligence supérieure ? Où est l'intelligence dans tout cela ?



Nous allons explorer le "Hello world" de l'IA en utilisant le perceptron pour apprendre le fonctionnement des opérateurs booléens (vrai ou faux) AND et OR. Le mot "apprendre" est utilisé ici car nous allons simplement indiquer les valeurs de sortie que nous souhaitons obtenir en fonction des entrées. Notre neurone va ainsi apprendre le concept des opérations AND et OR et sera capable de les réaliser sans que l'opération elle-même ait été explicitement programmée dans le code.

**AND**


Petit rappel , une operation **AND** retourne 1 si les deux valeurs sont à 1. Voici la table de correspondance : 

| Entrée A | Entrée B | AND        |
|----------|----------|------------|
| 0        | 0        | 0          |
| 0        | 1        | 0          |
| 1        | 0        | 0          |
| 1        | 1        | 1          |

Nous avons donc deux entrées à notre neuronne : A et B.  La fonction de notre perceptron ressemblera à ça : f(A * w1 + B * w2 + b) = output. Notre but est donc maintenant que la sortie de 





## C'est quoi un réseau de neurones ?



## Pourquoi les tenseurs ?

## Notre premiere implémentations