# 🌊 Modélisation et Optimisation d'un Système de Dépollution de Lac

## 📦 Structure du Code

```
├── main.py                # Script principal pour exécuter les simulations
└── tools/                 # Dossier contenant tous les modules outils
    ├── __init__.py        # Fichier vide pour rendre tools un package Python
    ├── model.py           # Définition du système différentiel
    ├── numerical_methods.py # Implémentation des méthodes numériques
    ├── utils.py           # Fonctions utilitaires et points critiques
    └── visualize.py       # Fonctions de visualisation avancées
```
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

## 📚 Références Scientifiques
- Théorie des systèmes dynamiques non-linéaires
- Méthodes numériques pour EDO (Hairer & Wanner)
- Applications en bioremédiation (Inria)

## 📝 Licence
Projet sous licence MIT - Voir le fichier [LICENSE](LICENSE) pour plus de détails.
