#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importation des bibliothèques nécessaires
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Chargement des données
file_path = '/Users/hachim/Movies/application/fiche.csv'
data = pd.read_csv(file_path)

# Afficher les premières lignes pour explorer les données
print(data.head()) 

# Prétraitement des données
data['date'] = pd.to_datetime(data['date'], format='%d/%m/%Y')
data.drop(columns=['ninsee'], inplace=True)  # Suppression de la colonne 'ninsee' pour simplification

# Extraction de caractéristiques temporelles (Tendance)
data['day'] = data['date'].dt.day
data['month'] = data['date'].dt.month

# Préparation des variables indépendantes et de la variable cible
X = data[['no2', 'o3', 'day', 'month']]  # Sélection des variables indépendantes (Indicateurs de qualité de l'air)
y = data['pm10'] # Sélection de la variable cible (Indicateur de qualité de l'air)

# Division en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardisation des features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modélisation avec une forêt aléatoire (Forêts aléatoires)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Prédiction sur l'ensemble de test
y_pred = model.predict(X_test_scaled)

# Évaluation du modèle
mse = mean_squared_error(y_test, y_pred)  # Calcul de l'erreur quadratique moyenne (MSE)
r2 = r2_score(y_test, y_pred)  # Calcul du coefficient de détermination (R²)

# Affichage des métriques
print(f'MSE: {mse}')  # Affichage de l'erreur quadratique moyenne (MSE)
print(f'R²: {r2}')  # Affichage du coefficient de détermination (R²)

# Visualisation des résultats réels vs prédits
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Valeurs réelles')
plt.ylabel('Prédictions')
plt.title('Valeurs réelles vs Prédictions')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
plt.show()


# Dioxyde d'azote (NO2) : 18.86 µg/m³
# 
# Ozone (O3) : 40.22 µg/m³
# 
# Particules fines (PM10) : 25.62 µg/m³
# 
# Ces résultats montrent que, en moyenne, les niveaux de pollution par l'ozone sont les plus élevés, suivis par les particules fines PM10 et le dioxyde d'azote. L'ozone, un polluant secondaire formé par la réaction du NO2 et d'autres précurseurs sous l'effet du soleil, peut être particulièrement élevé pendant les périodes ensoleillées du printemps et de l'été. Les niveaux moyens de NO2 et PM10 restent inférieurs à ceux de l'O3, mais ils représentent toujours une source importante de pollution atmosphérique, particulièrement dans les zones urbaines et à fort trafic.

# # Conclusion sur la pollution en Île-de-France en avril 2018 :

# En avril 2018, l'Île-de-France a connu des niveaux modérés de pollution atmosphérique, avec l'ozone comme principal polluant. Bien que les moyennes des concentrations de NO2 et PM10 soient relativement plus basses, elles indiquent néanmoins une exposition significative à ces polluants, surtout dans certaines zones. Ces niveaux de pollution peuvent avoir des impacts sur la santé publique, notamment pour les personnes sensibles telles que les enfants, les personnes âgées et celles souffrant de maladies respiratoires. Il est important de poursuivre les efforts pour réduire les émissions de polluants, notamment par la gestion du trafic, l'amélioration de la qualité des carburants et des véhicules, et la promotion des modes de transport moins polluants.

# # Afficher toutes les graphique possibles

# In[2]:


# Conversion de la colonne 'date' en datetime
data['date'] = pd.to_datetime(data['date'], format='%d/%m/%Y')

# Définition du style du graphique à ggplot pour assurer la compatibilité
plt.style.use('ggplot')

# Création de la figure et des axes pour les subplots
fig, ax = plt.subplots(3, 1, figsize=(14, 12))

# Tracé de NO2
ax[0].plot(data['date'], data['no2'], label='NO2', color='blue')
ax[0].set_title('Concentrations quotidiennes de NO2 en Avril 2018')
ax[0].set_ylabel('Concentration de NO2')
ax[0].legend()

# Tracé de O3
ax[1].plot(data['date'], data['o3'], label='O3', color='green')
ax[1].set_title('Concentrations quotidiennes de O3 en Avril 2018')
ax[1].set_ylabel('Concentration de O3')
ax[1].legend()

# Tracé de PM10
ax[2].plot(data['date'], data['pm10'], label='PM10', color='red')
ax[2].set_title('Concentrations quotidiennes de PM10 en Avril 2018')
ax[2].set_ylabel('Concentration de PM10')
ax[2].legend()

plt.tight_layout()
plt.show()


# # Les graphiques de tendance temporelle

# In[3]:


# Re-importation des librairies nécessaires et rechargement des données après réinitialisation de l'environnement d'exécution

# Conversion de la colonne 'date' en format datetime
data['date'] = pd.to_datetime(data['date'], format='%d/%m/%Y')

# Création des graphiques de tendance pour chaque polluant
plt.figure(figsize=(18, 12))

# NO2
plt.subplot(3, 1, 1)
sns.lineplot(x='date', y='no2', data=data)
plt.title('Tendance temporelle de la concentration de NO2 en avril 2018')
plt.ylabel('NO2 (µg/m³)')

# O3
plt.subplot(3, 1, 2)
sns.lineplot(x='date', y='o3', data=data, color='green')
plt.title('Tendance temporelle de la concentration de O3 en avril 2018')
plt.ylabel('O3 (µg/m³)')

# PM10
plt.subplot(3, 1, 3)
sns.lineplot(x='date', y='pm10', data=data, color='orange')
plt.title('Tendance temporelle de la concentration de PM10 en avril 2018')
plt.ylabel('PM10 (µg/m³)')

plt.tight_layout()
plt.show()


# Les graphiques de tendance temporelle ci-dessus montrent l'évolution des concentrations de chaque polluant (NO2, O3, PM10) au cours du mois d'avril 2018 en Île-de-France. On peut observer les variations quotidiennes qui peuvent être influencées par divers facteurs tels que les conditions météorologiques, le trafic, et les activités industrielles.

# # Les histogrammes 

# In[4]:


# Création des histogrammes pour chaque polluant
#import is
plt.figure(figsize=(18, 5))

# NO2
plt.subplot(1, 3, 1)
sns.histplot(data['no2'], bins=20, kde=True)
plt.title('Distribution de NO2')
plt.xlabel('NO2 (µg/m³)')
plt.ylabel('Fréquence')

# O3
plt.subplot(1, 3, 2)
sns.histplot(data['o3'], bins=20, kde=True, color='green')
plt.title('Distribution de O3')
plt.xlabel('O3 (µg/m³)')

# PM10
plt.subplot(1, 3, 3)
sns.histplot(data['pm10'], bins=20, kde=True, color='orange')
plt.title('Distribution de PM10')
plt.xlabel('PM10 (µg/m³)')

plt.tight_layout()
plt.show()


# Les histogrammes ci-dessus montrent la distribution des concentrations pour chaque polluant (NO2, O3, PM10) en Île-de-France au cours du mois d'avril 2018. Ces distributions nous permettent de voir la fréquence des différentes concentrations mesurées pour chaque type de polluant, avec des courbes d'estimation de densité pour mieux apprécier la forme de la distribution.

# # Les boîtes à moustaches

# In[5]:


plt.figure(figsize=(18, 5))

# NO2
plt.subplot(1, 3, 1)
sns.boxplot(x='no2', data=data)
plt.title('Niveaux de NO2 par commune')
plt.xlabel('NO2 (µg/m³)')

# O3
plt.subplot(1, 3, 2)
sns.boxplot(x='o3', data=data, color='green')
plt.title('Niveaux de O3 par commune')
plt.xlabel('O3 (µg/m³)')

# PM10
plt.subplot(1, 3, 3)
sns.boxplot(x='pm10', data=data, color='orange')
plt.title('Niveaux de PM10 par commune')
plt.xlabel('PM10 (µg/m³)')

plt.tight_layout()
plt.show()


# Les boîtes à moustaches (boxplots) ci-dessus présentent une vue d'ensemble des niveaux de pollution (NO2, O3, PM10) à travers l'Île-de-France en avril 2018, mais ils sont affichés de manière unidimensionnelle au lieu de comparer directement entre les communes. Cette représentation permet néanmoins de visualiser la médiane, la dispersion et les valeurs extrêmes pour chaque polluant, offrant une perspective sur la variabilité des concentrations dans la région. Pour une analyse plus précise entre les communes, il serait nécessaire d'ajuster la visualisation pour refléter les différences géographiques ou numériques spécifiques entre les zones.

# # Synthèse des Visualisations :

# 
# - Les graphiques de tendance temporelle ont montré des fluctuations quotidiennes des concentrations de polluants, reflétant l'impact des activités humaines et des conditions météorologiques.
#   
# - Les histogrammes ont révélé la distribution des concentrations de polluants, montrant une prévalence de valeurs moyennes avec des occurrences moins fréquentes de niveaux très élevés ou très bas.
#   
# - Les boîtes à moustaches ont illustré la variabilité des concentrations pour chaque polluant, bien que la comparaison directe entre les communes ait été limitée dans cette visualisation.
# 
#   
# Ces analyses et visualisations offrent un aperçu approfondi de la pollution en Île-de-France en avril 2018, soulignant l'importance d'une surveillance continue et de mesures de mitigation pour gérer la qualité de l'air dans la région

# In[ ]:





# In[ ]:




