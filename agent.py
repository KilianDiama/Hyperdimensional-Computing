"""
agent.py — HDC logical agent (production-grade, deterministic, extensible)

Invariants:
- Deterministic concept vectors from string keys (SHA-256 → bipolar).
- Clear separation of concerns: kernel / concepts / facts / agent.
- Slot-filling via role debinding + cleanup over entity space.
- Confidence calibration + hard minimum score to reduce hallucinations.
- Scaling-friendly fact storage (amortized append, no O(N) concat per insert).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


# =========================
#  CONFIG & TYPES
# =========================

Device = torch.device


@dataclass(frozen=True)
class HDCConfig:
    dim: int = 20_000
    device: Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    confidence_threshold: float = 0.10  # soft threshold (semantic)
    min_score: float = 0.05             # hard floor (safety)
    top_k: int = 5                      # top-k for cleanup
    # FactStore growth
    initial_capacity: int = 1024
    growth_factor: float = 2.0


# =========================
#  DETERMINISTIC UTILITIES
# =========================

def hash_to_bipolar_vector(key: str, dim: int, device: Device) -> torch.Tensor:
    """
    Deterministic mapping: string key -> bipolar vector in {-1, +1}^dim.

    - Uses SHA-256 in a loop to generate enough bits.
    - Stable across runs, machines, and Python processes.
    """
    needed_bits = dim
    bits: List[int] = []

    seed = key.encode("utf-8")
    while len(bits) < needed_bits:
        h = hashlib.sha256(seed).digest()
        for byte in h:
            for i in range(8):
                bits.append((byte >> i) & 1)
                if len(bits) >= needed_bits:
                    break
            if len(bits) >= needed_bits:
                break
        seed = h  # evolve seed to avoid trivial cycles

    arr = torch.tensor(bits, dtype=torch.float32, device=device).view(1, -1)
    arr = arr * 2.0 - 1.0  # {0,1} -> {-1,+1}
    return arr


# =========================
#  HDC KERNEL
# =========================

class HDCKernel:
    """
    Core HDC operations:
    - binding (XOR in bipolar space)
    - bundling (superposition)
    - similarity (cosine-like for bipolar vectors)
    - cleanup (top-k nearest neighbors)
    """

    def __init__(self, config: HDCConfig):
        self.config = config
        self.dim = config.dim
        self.device = config.device

    # --- Algebra ---

    def bind(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        """
        Binding (XOR in bipolar space).
        v1, v2: (1, D)
        """
        return v1 * v2

    def bundle(self, vectors: List[torch.Tensor], normalize: bool = True) -> torch.Tensor:
        """
        Bundling by sum + sign.
        vectors: list of (1, D)
        """
        if not vectors:
            raise ValueError("bundle() requires at least one vector")

        stacked = torch.cat(vectors, dim=0)  # (N, D)
        summed = stacked.sum(dim=0, keepdim=True)  # (1, D)
        return summed.sign() if normalize else summed

    # --- Similarity & cleanup ---

    def similarity(self, q: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """
        Approximate cosine similarity for bipolar vectors:
        dot(q, m) / dim
        q: (1, D)
        m: (N, D)
        return: (1, N)
        """
        if m.numel() == 0:
            return torch.empty((1, 0), device=self.device)
        return torch.matmul(q, m.t()) / self.dim

    def cleanup(
        self,
        query_vec: torch.Tensor,
        memory_matrix: torch.Tensor,
        names: List[str],
        top_k: int,
    ) -> List[Tuple[str, float]]:
        """
        Vectorized cleanup with top-k.
        - query_vec: (1, D)
        - memory_matrix: (N, D)
        - names: len N
        Returns a sorted list [(name, score), ...] of size ≤ top_k.
        """
        if memory_matrix.shape[0] == 0:
            return []

        scores = self.similarity(query_vec, memory_matrix)  # (1, N)
        scores = scores.squeeze(0)  # (N,)

        k = min(top_k, scores.shape[0])
        best_scores, best_idx = torch.topk(scores, k=k, dim=0)

        results: List[Tuple[str, float]] = []
        for s, idx in zip(best_scores.tolist(), best_idx.tolist()):
            results.append((names[idx], float(s)))
        return results


# =========================
#  CONCEPT STORE
# =========================

class ConceptStore:
    """
    Stores deterministic concept vectors:
    - roles
    - entities
    - relations

    Keys are strings, e.g.:
    - "Role:Sujet"
    - "Entity:Jean"
    - "Relation:Signe"
    """

    def __init__(self, config: HDCConfig, kernel: HDCKernel):
        self.config = config
        self.kernel = kernel
        self.device = config.device
        self.dim = config.dim

        self._concepts: Dict[str, torch.Tensor] = {}

    def get(self, key: str) -> torch.Tensor:
        """
        Returns the vector associated with the key, creates it if necessary.
        """
        if key not in self._concepts:
            self._concepts[key] = hash_to_bipolar_vector(key, self.dim, self.device)
        return self._concepts[key]

    def get_role(self, name: str) -> torch.Tensor:
        return self.get(f"Role:{name}")

    def get_entity(self, name: str) -> torch.Tensor:
        return self.get(f"Entity:{name}")

    def get_relation(self, name: str) -> torch.Tensor:
        return self.get(f"Relation:{name}")

    @property
    def entity_keys(self) -> List[str]:
        return [k for k in self._concepts.keys() if k.startswith("Entity:")]

    @property
    def entity_matrix(self) -> torch.Tensor:
        keys = self.entity_keys
        if not keys:
            return torch.empty((0, self.dim), device=self.device)
        return torch.cat([self._concepts[k] for k in keys], dim=0)


# =========================
#  FACT STORE (SCALING-FRIENDLY)
# =========================

@dataclass
class Fact:
    id: str
    subject: str
    relation: str
    object: str
    time: Optional[str]
    location: Optional[str]
    vector: torch.Tensor  # shape: (1, D)


class FactStore:
    """
    Stores encoded facts:
    - list of Fact
    - matrix (N, D) for fast similarity search

    Uses a capacity-based growth strategy to avoid O(N) concat on each insert.
    """

    def __init__(self, config: HDCConfig):
        self.config = config
        self.device = config.device
        self.dim = config.dim

        self._facts: List[Fact] = []
        self._capacity = max(1, config.initial_capacity)
        self._matrix = torch.empty((self._capacity, self.dim), device=self.device)
        self._size = 0  # number of used rows

    def _grow_if_needed(self) -> None:
        if self._size < self._capacity:
            return
        new_capacity = int(self._capacity * self.config.growth_factor)
        new_matrix = torch.empty((new_capacity, self.dim), device=self.device)
        if self._size > 0:
            new_matrix[: self._size] = self._matrix[: self._size]
        self._matrix = new_matrix
        self._capacity = new_capacity

    def add_fact(self, fact: Fact) -> None:
        self._grow_if_needed()
        self._facts.append(fact)
        self._matrix[self._size : self._size + 1] = fact.vector
        self._size += 1

    @property
    def facts(self) -> List[Fact]:
        return self._facts

    @property
    def matrix(self) -> torch.Tensor:
        return self._matrix[: self._size]

    def get_by_id(self, fact_id: str) -> Optional[Fact]:
        for f in self._facts:
            if f.id == fact_id:
                return f
        return None


# =========================
#  HDC LOGICAL AGENT
# =========================

class HDCAgent:
    """
    Logical HDC agent:
    - Minimal ontology (roles, entities, relations)
    - Fact encoding (Subject, Relation, Object, Time?, Location?)
    - Robust slot-filling
    - Similar fact retrieval
    - Confidence calibration
    """

    def __init__(self, config: Optional[HDCConfig] = None):
        self.config = config or HDCConfig()
        self.kernel = HDCKernel(self.config)
        self.concepts = ConceptStore(self.config, self.kernel)
        self.facts = FactStore(self.config)

        self._init_default_roles()

    # ---------- Ontology ----------

    def _init_default_roles(self) -> None:
        for role in ("Sujet", "Objet", "Temps", "Lieu"):
            _ = self.concepts.get_role(role)

    # ---------- Fact encoding ----------

    def encode_fact(
        self,
        subject: str,
        relation: str,
        obj: str,
        time: Optional[str] = None,
        location: Optional[str] = None,
        fact_id: Optional[str] = None,
    ) -> Fact:
        """
        Encode a fact (Subject, Relation, Object, Time?, Location?) into an HDC vector.
        Returns a Fact with its vector.
        """
        # Roles
        r_subj = self.concepts.get_role("Sujet")
        r_obj = self.concepts.get_role("Objet")
        r_time = self.concepts.get_role("Temps")
        r_loc = self.concepts.get_role("Lieu")

        # Entities / relation
        v_subj = self.concepts.get_entity(subject)
        v_obj = self.concepts.get_entity(obj)
        v_rel = self.concepts.get_relation(relation)

        # Mandatory slots
        slot_subj = self.kernel.bind(r_subj, v_subj)
        slot_obj = self.kernel.bind(r_obj, v_obj)
        slot_rel = v_rel  # global tag

        vectors = [slot_subj, slot_obj, slot_rel]

        # Optional slots
        if time is not None:
            v_time = self.concepts.get_entity(time)
            slot_time = self.kernel.bind(r_time, v_time)
            vectors.append(slot_time)

        if location is not None:
            v_loc = self.concepts.get_entity(location)
            slot_loc = self.kernel.bind(r_loc, v_loc)
            vectors.append(slot_loc)

        fact_vec = self.kernel.bundle(vectors)

        fact = Fact(
            id=fact_id or f"fact_{len(self.facts.facts)}",
            subject=subject,
            relation=relation,
            object=obj,
            time=time,
            location=location,
            vector=fact_vec,
        )
        return fact

    def add_fact(
        self,
        subject: str,
        relation: str,
        obj: str,
        time: Optional[str] = None,
        location: Optional[str] = None,
        fact_id: Optional[str] = None,
    ) -> Fact:
        fact = self.encode_fact(subject, relation, obj, time, location, fact_id)
        self.facts.add_fact(fact)
        return fact

    # ---------- Slot-filling ----------

    def _query_role_from_vector(
        self, fact_vector: torch.Tensor, role_name: str
    ) -> Tuple[Optional[str], float]:
        """
        Debinding: fact * Role ≈ Entity
        Returns (entity_name or None, score).
        """
        role_vec = self.concepts.get_role(role_name)
        query_vec = self.kernel.bind(fact_vector, role_vec)

        entity_keys = self.concepts.entity_keys
        if not entity_keys:
            return None, 0.0

        entity_matrix = self.concepts.entity_matrix  # (N, D)

        candidates = self.kernel.cleanup(
            query_vec,
            entity_matrix,
            entity_keys,
            top_k=self.config.top_k,
        )
        if not candidates:
            return None, 0.0

        best_name, best_score = candidates[0]

        # Confidence calibration: hard + soft thresholds
        threshold = max(self.config.confidence_threshold, self.config.min_score)
        if best_score < threshold:
            return None, best_score

        # "Entity:Jean" -> "Jean"
        entity_name = best_name.split("Entity:", 1)[-1]
        return entity_name, best_score

    def query_subject(self, fact: Fact) -> Tuple[Optional[str], float]:
        return self._query_role_from_vector(fact.vector, "Sujet")

    def query_object(self, fact: Fact) -> Tuple[Optional[str], float]:
        return self._query_role_from_vector(fact.vector, "Objet")

    def query_time(self, fact: Fact) -> Tuple[Optional[str], float]:
        return self._query_role_from_vector(fact.vector, "Temps")

    def query_location(self, fact: Fact) -> Tuple[Optional[str], float]:
        return self._query_role_from_vector(fact.vector, "Lieu")

    # ---------- Similar fact search ----------

    def find_most_similar_fact(
        self, query_vector: torch.Tensor
    ) -> Optional[Tuple[Fact, float]]:
        """
        Returns the stored fact most similar to a query vector.
        """
        if not self.facts.facts:
            return None

        matrix = self.facts.matrix  # (N, D)
        scores = self.kernel.similarity(query_vector, matrix)  # (1, N)
        scores = scores.squeeze(0)  # (N,)

        best_score, best_idx = torch.max(scores, dim=0)
        best_fact = self.facts.facts[best_idx.item()]
        return best_fact, float(best_score.item())

    # ---------- High-level interface ----------

    def ask(self, question_type: str, fact: Fact) -> Tuple[Optional[str], float]:
        """
        question_type ∈ {"sujet", "objet", "temps", "lieu"}
        """
        if question_type == "sujet":
            return self.query_subject(fact)
        elif question_type == "objet":
            return self.query_object(fact)
        elif question_type == "temps":
            return self.query_time(fact)
        elif question_type == "lieu":
            return self.query_location(fact)
        else:
            raise ValueError(f"Unknown question type: {question_type}")


# =========================
#  DEMO / MAIN
# =========================

if __name__ == "__main__":
    config = HDCConfig(dim=20_000, confidence_threshold=0.08, min_score=0.05)
    agent = HDCAgent(config=config)

    print(f"--- HDC Agent initialized (dim={agent.config.dim}, device={agent.config.device}) ---")

    # 1. Encode a fact
    fact = agent.add_fact(
        subject="Jean",
        relation="Signe",
        obj="Contrat_Vente",
        time="2025-06-01",
        location="Paris",
        fact_id="F1",
    )

    print("\nEncoded fact:")
    print(f"  ID      : {fact.id}")
    print(f"  Subject : {fact.subject}")
    print(f"  Relation: {fact.relation}")
    print(f"  Object  : {fact.object}")
    print(f"  Time    : {fact.time}")
    print(f"  Location: {fact.location}")

    # 2. Slot-filling
    subj, s_score = agent.ask("sujet", fact)
    obj, o_score = agent.ask("objet", fact)
    t, t_score = agent.ask("temps", fact)
    loc, l_score = agent.ask("lieu", fact)

    print("\n--- Slot-filling queries ---")
    print(f"Who is the subject?  -> {subj} (score={s_score:.4f})")
    print(f"What is the object?  -> {obj} (score={o_score:.4f})")
    print(f"When?                -> {t} (score={t_score:.4f})")
    print(f"Where?               -> {loc} (score={l_score:.4f})")

    # 3. Stress test: noise on fact vector
    noisy_fact_vec = fact.vector.clone()
    noise_level = 0.40
    mask = torch.rand_like(noisy_fact_vec) < noise_level
    noisy_fact_vec[mask] *= -1

    subj_n, s_score_n = agent._query_role_from_vector(noisy_fact_vec, "Sujet")
    print("\n--- Resilience test (40% corruption on fact) ---")
    print(f"Recovered subject under noise: {subj_n} (score={s_score_n:.4f})")
