# DiffuDigit — guide rapide

Un modèle de diffusion (même principe que la formation NVIDIA) qui génère un chiffre manuscrit
à partir d'un bruit aléatoire, avec une petite application web.

## 1. Installer
```
pip install -r requirements.txt
```

## 2. Entraîner le modèle (MNIST est téléchargé automatiquement)
```
python train.py --epochs 10
```
- Sur un PC sans carte graphique : environ 3 à 6 min par époque.
- Plus rapide : Google Colab (Exécution > Modifier le type d'exécution > GPU T4).
- À chaque époque, une grille de chiffres générés est enregistrée dans `samples/`.
- Le modèle est enregistré dans `diffudigit.pt`.

## 3. Lancer l'application
```
python app.py
```
Ouvrir http://127.0.0.1:7860 : choisir un chiffre, cliquer sur « Générer ».
L'application montre l'image générée, les étapes du débruitage et une animation.

## Fichiers
- `model.py` : U-Net + processus de diffusion (DDPM) + Classifier-Free Guidance
- `train.py` : entraînement
- `app.py`   : interface (Gradio)
