import torch
import sys
import time
from model import BigramLanguageModel

# 1. Configuration & Device
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

# 2. Charger le texte pour reconstruire le vocabulaire (indispensable)
# On a besoin des mêmes caractères que pendant l'entraînement
with open('austen.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
itos = { i:ch for i,ch in enumerate(chars) }
decode = lambda l: ''.join([itos[i] for i in l])

# 3. Charger l'architecture du modèle
# On passe le vocab_size qu'on vient de calculer
model = BigramLanguageModel(vocab_size)
model = model.to(device)

# 4. Charger les poids sauvegardés
print("Chargement des poids du modèle...")
try:
    # On charge le fichier généré par ton entraînement
    model.load_state_dict(torch.load('model_final.pt', map_location=device))
    print("Modèle chargé avec succès.")
except FileNotFoundError:
    print("Erreur : 'model_final.pt' est introuvable. Vérifie le nom du fichier.")
    exit()

model.eval() # Mode évaluation (désactive le dropout)

# 5. Génération de texte
print("\n--- Start generating in Jane Austen's style ---\n")

# On commence avec un token vide (0)
context = torch.zeros((1, 1), dtype=torch.long, device=device)

for token_id in model.generate_streaming(context, max_new_tokens=1000):
    char = itos[token_id]
    sys.stdout.write(char)
    sys.stdout.flush()
    time.sleep(0.01)

print("\n--- The end ---")