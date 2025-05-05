# 🌊 Modélisation et Optimisation d'un Système de Dépollution de Lac

Ce projet propose une modélisation mathématique et une simulation numérique d'un système innovant de dépollution de lac utilisant un bioréacteur à circuit fermé. Le système est décrit par des équations différentielles non-linéaires couplant biomasse et polluants, avec une analyse approfondie de sa dynamique et des stratégies d'optimisation.

## 📌 Contexte Scientifique

**Problématique** :  
Dépolluer efficacement un lac en utilisant un bioréacteur contenant une biomasse qui dégrade les polluants, avec contrôle du débit d'eau Q entre le lac et le réacteur.

**Modèle Mathématique** :
```
dx/dt = μxy - Qx       # Évolution de la biomasse (x)
dy/dt = -μxy + Q(z-y)  # Polluant dans le réacteur (y)
dz/dt = εQ(y-z)        # Polluant dans le lac (z)
```
où :
- μ : taux de croissance de la biomasse
- Q : débit de traitement (variable d'optimisation)
- ε = VR/VL : ratio des volumes réacteur/lac

## 🎯 Objectifs Principaux

1. **Analyse de stabilité** des points d'équilibre pour ε=0 et ε>0
2. **Optimisation du débit Q** pour minimiser le temps de dépollution
3. **Comparaison des méthodes numériques** (Euler, RK4)
4. **Application réaliste** au lac Thai en Chine

## 📦 Structure du Code

```
.
├── main.py                # Script principal avec simulations et visualisations
├── model.py               # Définition du système différentiel
├── numerical_methods.py   # Implémentation des méthodes numériques
├── utils.py               # Fonctions utilitaires et points critiques
└── visualize.py           # Fonctions de visualisation avancées
```

## 🔍 Résultats Clés

### 1. Analyse de Stabilité (ε=0)

### 2. Analyse de Stabilité (ε>0)

### 3. Optimisation du Débit
- **Débit constant optimal** : Q ≈ 0.15 (pour z₀=1, z₁=0.2)
- **Stratégie variable** : Découpage en N intervalles avec Q(t) adaptatif

### 4. Comparaison des Méthodes Numériques
| Méthode         | Ordre | Précision | Stabilité |
|-----------------|-------|-----------|-----------|
| Euler Explicite | 1     | Faible    | Conditionnelle |
| Euler Implicite | 1     | Moyenne   | Inconditionnelle |
| RK4             | 4     | Élevée    | Conditionnelle |

## 🚀 Comment Utiliser le Code

1. **Installation** :
```bash
git clone https://github.com/votre-utilisateur/depollution-lac.git
cd depollution-lac
pip install numpy matplotlib scipy
```

2. **Exécution** :
```bash
python main.py
```

3. **Fonctionnalités** :
- Simulation des dynamiques pour différents paramètres
- Visualisation des portraits de phase 2D/3D
- Optimisation automatique du débit Q
- Comparaison des méthodes numériques

## 📊 Exemples de Visualisations

<p align="center">
  <img src="https://via.placeholder.com/400x300?text=Portrait+de+Phase+3D" alt="Portrait de Phase">
  <img src="https://via.placeholder.com/400x300?text=Optimisation+du+Débit" alt="Optimisation Q">
</p>

## 🌍 Application au Lac Thai (Chine)
- Volume du lac : 4.5×10⁹ m³
- Volume du réacteur : 4.5×10⁵ m³
- **Résultats** :
  - 50% de dépollution en 2000 jours
  - Dépollution complète en 8000 jours

## 📚 Références Scientifiques
- Théorie des systèmes dynamiques non-linéaires
- Méthodes numériques pour EDO (Hairer & Wanner)
- Applications en bioremédiation (Inria)

## 📝 Licence
Projet sous licence MIT - Voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

**Contributions bienvenues** ! Pour toute question ou suggestion, veuillez ouvrir une issue sur le dépôt GitHub.
