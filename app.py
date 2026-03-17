import torch
import torch.nn.functional as F

class UltraLogicalHDC:
    def __init__(self, dim=20000, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.dim = dim
        self.device = device
        # Mémoire associative sous forme de matrice pour une recherche instantanée
        self.memory_matrix = torch.empty((0, dim), device=device)
        self.memory_names = []
        self.epsilon = 0.12 # Seuil de confiance statistique en 20k-dim

    def create_concept(self, name):
        """Génère et stocke un vecteur bipolaire atomique."""
        v = torch.randn((1, self.dim), device=self.device).sign()
        self.memory_matrix = torch.cat([self.memory_matrix, v], dim=0)
        self.memory_names.append(name)
        return v

    def bind(self, v1, v2):
        """Liaison (XOR) : préserve la distance, crée une relation."""
        return v1 * v2

    def bundle(self, vectors, normalize=True):
        """Superposition (Addition) : crée un ensemble/contexte."""
        # On somme sans perdre l'intensité des votes avant le sign() final
        summed = torch.sum(torch.cat(vectors, dim=0), dim=0, keepdim=True)
        return summed.sign() if normalize else summed

    def cleanup(self, query_vec):
        """Recherche vectorisée (Matrix Multiplication) : O(1) complexité logique."""
        if self.memory_matrix.shape[0] == 0:
            return "Vide", 0.0
        
        # Similarité cosinus via produit matriciel (v @ M.T)
        # En bipolaire, dot(a, b) / dim est la mesure standard
        scores = torch.matmul(query_vec, self.memory_matrix.t()) / self.dim
        best_score, best_idx = torch.max(scores, dim=1)
        
        match_name = self.memory_names[best_idx.item()]
        return match_name, best_score.item()

# --- SCÉNARIO EXPERT : Raisonnement par Slot Filling ---
hdc = UltraLogicalHDC()

# 1. Définition de l'Ontologie (Concepts de base)
ROLE_SUJET = hdc.create_concept("Rôle:Sujet")
ROLE_OBJET = hdc.create_concept("Rôle:Objet")
JEAN = hdc.create_concept("Jean")
CONTRAT = hdc.create_concept("Contrat_Vente")

# 2. Encodage d'une "Structure de Données" Hyperdimensionnelle
# On lie les rôles aux valeurs : (Rôle * Valeur)
# C'est la base des graphes de connaissances neuronaux.
fait_1 = hdc.bind(ROLE_SUJET, JEAN)
fait_2 = hdc.bind(ROLE_OBJET, CONTRAT)

# Fusion dans un "Mémoire de Travail" globale
memoire_travail = hdc.bundle([fait_1, fait_2])

print(f"--- Moteur HDC Initialisé (Dim: {hdc.dim}) ---")

# 3. Requête Logique : "Qui occupe le rôle de Sujet ?"
# Extraction (Débinding) : memoire * ROLE == VALEUR
query = hdc.bind(memoire_travail, ROLE_SUJET)
nom, score = hdc.cleanup(query)

print(f"Requête : 'Qui est le Sujet ?'")
print(f"Résultat : {nom} (Confiance : {score:.4f})")

# 4. Stress Test : Injection de 40% de bruit (destruction massive)
noisy_query = query.clone()
mask = torch.rand_like(query) < 0.40
noisy_query[mask] *= -1 # On inverse 40% des bits

nom_bruit, score_bruit = hdc.cleanup(noisy_query)
print(f"\n--- Test de Résilience (40% de corruption) ---")
print(f"Résultat sous stress : {nom_bruit} (Score : {score_bruit:.4f})")
