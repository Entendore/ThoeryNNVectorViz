"""
Neural Architecture Search with Set Theory & Group Theory foundations.

A research-grade toolkit combining:
- Quiver representation theory for network topology
- Group equivariant layers (Reynolds operator projection)
- Algebraic geometry of weight varieties
- Multi-algorithm NAS (Random, Grid, Evolutionary, Bayesian, RL)
- Vector database for architecture similarity search
- Interactive PySide6 visualization
"""

from __future__ import annotations

import sys
import json
import logging
import random
import math
import time
import itertools
import os
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Any, Callable, Sequence
from abc import ABC, abstractmethod
from math import erf, sqrt

import numpy as np
import networkx as nx
import chromadb
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QGraphicsView, QGraphicsScene, QGraphicsEllipseItem, QGraphicsTextItem,
    QGraphicsLineItem, QLabel, QTextEdit, QComboBox, QSlider, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QListWidget, QListWidgetItem,
    QSplitter, QGroupBox, QFormLayout, QSpinBox, QCheckBox, QMessageBox,
    QProgressBar, QScrollArea, QFileDialog,
    QStatusBar, QMenuBar, QAction, QShortcut
)
from PySide6.QtCore import Qt, Signal, QThread, QTimer, QPointF
from PySide6.QtGui import QBrush, QPen, QColor, QFont, QPainter, QKeySequence, QPolygonF
from collections import defaultdict

# ============================================================
# Configuration & Constants
# ============================================================

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("nas")
logger.setLevel(logging.INFO)

# Numerical stability
EPSILON = 1e-6
CLIP_NORM = 10.0

# Group theory defaults
DEFAULT_PERMUTATION_N = 4
DEFAULT_TRANSLATION_N = 4
DEFAULT_ROTATION_STEPS = 8

# Architecture search bounds
MIN_LAYER_SIZE = 2
MAX_LAYER_SIZE = 256
MAX_HIDDEN_LAYERS = 6

# RL hyperparameters
RL_LEARNING_RATE = 0.01
RL_STD_FLOOR = 0.5
RL_STD_DECAY = 0.97
RL_BASELINE_DECAY = 0.9
RL_ENTROPY_BONUS = 0.01
RL_PROB_FLOOR = 0.05

# Database
EMBEDDING_DIM = 256
COLLECTION_NAME = "nas_db"

# Visualization
LAYER_COLORS = [
    "#4a90e2", "#e24a4a", "#4ae24a", "#e2c84a",
    "#9b4ae2", "#4ae2c8", "#e28b4a", "#7a7a7a",
]
NEURON_RADIUS = 15
EDGE_CULL_DENSE_THRESHOLD = 200

MAX_STORED_NETWORKS = 50

# ============================================================
# Mathematical Foundations
# ============================================================

class QuiverRepresentation:
    """A quiver (directed graph) with a linear representation on edges."""

    def __init__(self) -> None:
        self.quiver: nx.DiGraph = nx.DiGraph()
        self.representation: Dict[Tuple[str, str], np.ndarray] = {}
        self._moduli_cache: Optional[str] = None
        self._invariants_cache: Optional[Dict[str, Any]] = None

    def add_neuron(self, neuron_id: str, layer: int,
                   position: Tuple[float, float],
                   vector_space_dim: int = 1) -> None:
        if vector_space_dim < 1:
            raise ValueError(f"vector_space_dim must be >= 1, got {vector_space_dim}")
        self.quiver.add_node(
            neuron_id,
            layer=layer,
            position=position,
            vector_space=np.zeros(vector_space_dim),
            dim=vector_space_dim,
        )

    def add_connection(self, source: str, target: str,
                       weight_matrix: np.ndarray) -> None:
        if source not in self.quiver or target not in self.quiver:
            raise KeyError(f"Unknown neuron: {source} or {target}")
        self.quiver.add_edge(
            source, target,
            weight=weight_matrix,
            linear_map=weight_matrix,
        )
        self.representation[(source, target)] = weight_matrix

    def _invalidate_caches(self) -> None:
        self._moduli_cache = None
        self._invariants_cache = None

    def compute_moduli_space(self) -> str:
        if self._moduli_cache is not None:
            return self._moduli_cache

        edges = list(self.quiver.edges(data=True))
        if not edges:
            self._moduli_cache = "Moduli Space (empty quiver; dim 0)"
            return self._moduli_cache

        total_dim = int(sum(e[2]['weight'].size for e in edges))
        vertex_dims = [d.get('dim', 1) for _, d in self.quiver.nodes(data=True)]
        gauge_dim = int(sum(d * d for d in vertex_dims))
        n_components = max(1, nx.number_weakly_connected_components(self.quiver))

        moduli_dim = max(0, total_dim - gauge_dim + n_components)
        self._moduli_cache = (
            f"Moduli Space (Param Dim: {total_dim}, "
            f"Gauge Dim: {gauge_dim}, Moduli Dim: {moduli_dim})"
        )
        return self._moduli_cache

    def get_algebraic_invariants(self) -> Dict[str, Any]:
        if self._invariants_cache is not None:
            return self._invariants_cache

        invariants: Dict[str, Any] = {}
        vertices = self.quiver.number_of_nodes()
        edges = self.quiver.number_of_edges()
        invariants['euler_characteristic'] = vertices - edges

        total_dim = sum(data['dim'] for _, data in self.quiver.nodes(data=True))
        invariants['representation_dimension'] = total_dim

        total_rank = 0
        for _, _, data in self.quiver.edges(data=True):
            w = data['weight']
            if w.size > 0:
                try:
                    w2d = w.reshape(w.shape[0], -1) if w.ndim > 2 else w
                    total_rank += int(np.linalg.matrix_rank(w2d))
                except np.linalg.LinAlgError:
                    logger.debug("SVD failed for rank computation")
        invariants['total_weight_rank'] = total_rank

        try:
            cycles = list(nx.simple_cycles(self.quiver))
            invariants['cycle_rank'] = len(cycles)
        except nx.NetworkXError:
            invariants['cycle_rank'] = 0

        invariants['is_weakly_connected'] = nx.is_weakly_connected(self.quiver)
        if invariants['is_weakly_connected']:
            try:
                invariants['diameter'] = nx.diameter(self.quiver.to_undirected())
            except (nx.NetworkXError, nx.NetworkXPointlessConcept):
                invariants['diameter'] = 0
        else:
            invariants['diameter'] = -1

        self._invariants_cache = invariants
        return invariants


class GroupEquivariantLayer:
    """Encapsulates a finite group and its actions on vector spaces."""

    SUPPORTED_GROUPS = ('permutation', 'translation', 'rotation')

    def __init__(self, group_type: str = 'permutation') -> None:
        if group_type not in self.SUPPORTED_GROUPS and group_type != 'none':
            raise ValueError(
                f"Unknown group_type: {group_type}. "
                f"Supported: {self.SUPPORTED_GROUPS + ('none',)}"
            )
        self.group_type = group_type
        self.group_elements: List[Any] = self._generate_group_elements()

    def _generate_group_elements(self) -> List[Any]:
        if self.group_type == 'permutation':
            return list(itertools.permutations(range(DEFAULT_PERMUTATION_N)))
        if self.group_type == 'translation':
            return [np.roll(np.eye(DEFAULT_TRANSLATION_N), shift, axis=1)
                    for shift in range(DEFAULT_TRANSLATION_N)]
        if self.group_type == 'rotation':
            return [
                np.array([[math.cos(t), -math.sin(t)],
                          [math.sin(t),  math.cos(t)]])
                for t in np.linspace(0, 2 * np.pi,
                                     DEFAULT_ROTATION_STEPS, endpoint=False)
            ]
        return [np.eye(2)]

    @staticmethod
    def _permutation_matrix(perm: Tuple[int, ...]) -> np.ndarray:
        n = len(perm)
        P = np.zeros((n, n))
        for i, p in enumerate(perm):
            if not (0 <= p < n):
                raise ValueError(f"Invalid permutation element {p} for size {n}")
            P[i, p] = 1.0
        return P

    def get_irreducible_representations(self) -> List[Dict[str, Any]]:
        if self.group_type == 'permutation':
            n = DEFAULT_PERMUTATION_N
            return [
                {'name': 'trivial', 'dim': 1,
                 'character': np.ones(len(self.group_elements))},
                {'name': 'standard', 'dim': n - 1,
                 'character': np.array(
                     [sum(g == tuple(range(n))) - 1 for g in self.group_elements]
                 )},
                {'name': 'alternating', 'dim': 1,
                 'character': np.array(
                     [1 if self._is_even_permutation(g) else -1
                      for g in self.group_elements]
                 )},
            ]
        if self.group_type == 'translation':
            return [
                {'name': f'irrep_{k}', 'dim': 1,
                 'character': np.array(
                     [np.exp(2j * math.pi * k * i / DEFAULT_TRANSLATION_N)
                      for i in range(DEFAULT_TRANSLATION_N)]
                 )}
                for k in range(DEFAULT_TRANSLATION_N)
            ]
        if self.group_type == 'rotation':
            chars = [float(np.trace(g)) for g in self.group_elements]
            return [{'name': 'rotation_2d', 'dim': 2, 'character': np.array(chars)}]
        return [{'name': 'trivial', 'dim': 1,
                 'character': np.ones(len(self.group_elements))}]

    @staticmethod
    def _is_even_permutation(perm: Tuple[int, ...]) -> bool:
        inv_count = 0
        for i in range(len(perm)):
            for j in range(i + 1, len(perm)):
                if perm[i] > perm[j]:
                    inv_count += 1
        return inv_count % 2 == 0


class AlgebraicGeometryAnalyzer:
    """Analyzes the algebraic variety defined by network weights."""

    def compute_function_space_geometry(
        self, network_weights: Sequence[np.ndarray]
    ) -> Dict[str, Any]:
        total_params = int(sum(w.size for w in network_weights))
        ranks: List[int] = []
        sv_spectra: List[np.ndarray] = []
        determinantal_dims: List[int] = []

        for w in network_weights:
            try:
                w2d = w.reshape(w.shape[0], -1) if w.ndim > 2 else w
                r = int(np.linalg.matrix_rank(w2d))
                ranks.append(r)
                sv = np.linalg.svd(w2d, compute_uv=False)
                sv_spectra.append(sv)
                m, n = w2d.shape
                determinantal_dims.append(r * (m + n - r))
            except np.linalg.LinAlgError:
                ranks.append(0)
                sv_spectra.append(np.array([]))
                determinantal_dims.append(0)

        variety_dim = int(sum(determinantal_dims))
        return {
            'total_parameters': total_params,
            'weight_ranks': ranks,
            'singular_value_spectra': sv_spectra,
            'variety_dimension': max(0, variety_dim),
            'determinantal_dimensions': determinantal_dims,
            'algebraic_complexity': float(np.log10(total_params + 1)),
        }


# ============================================================
# Architecture Generation
# ============================================================

class SetTheoryArchitectureGenerator:
    """Generates architectures based on set-theoretic cardinality constraints."""

    @staticmethod
    def generate_by_cardinality(
        input_dim: int, output_dim: int,
        max_hidden_layers: int = 4,
        cardinality_constraint: str = "balanced"
    ) -> List[int]:
        if cardinality_constraint == "balanced":
            mid = (input_dim + output_dim) // 2
            return ([input_dim]
                    + [max(MIN_LAYER_SIZE, mid + random.randint(-2, 2))
                       for _ in range(max_hidden_layers)]
                    + [output_dim])
        if cardinality_constraint == "decreasing":
            layers = [input_dim]
            current = input_dim
            for _ in range(max_hidden_layers):
                current = max(output_dim + 1, current - random.randint(1, 4))
                layers.append(current)
            layers.append(output_dim)
            return layers
        if cardinality_constraint == "union_partition":
            ps = max(1, input_dim // random.randint(2, 4))
            return ([input_dim]
                    + [ps * random.randint(2, 5) for _ in range(max_hidden_layers)]
                    + [output_dim])
        if cardinality_constraint == "power_set":
            return ([input_dim]
                    + [min(MAX_LAYER_SIZE, max(MIN_LAYER_SIZE,
                            input_dim ** random.choice([1, 2, 3])))
                       for _ in range(max_hidden_layers)]
                    + [output_dim])
        return [input_dim, output_dim]


class GroupTheoryArchitectureGenerator:
    """Generates group-equivariant architectures."""

    NETWORK_TYPES = ('MLP', 'CNN', 'RNN', 'GNN', 'Transformer')

    COMPATIBILITY: Dict[str, Tuple[str, ...]] = {
        'permutation':  ('MLP', 'Transformer', 'GNN'),
        'translation':  ('CNN', 'RNN'),
        'rotation':     ('CNN',),
        'none':         NETWORK_TYPES,
    }

    def generate_equivariant_architecture(
        self, group_type: str, input_dim: int, output_dim: int,
        num_layers: int = 3
    ) -> Tuple[List[int], str, str]:
        compatible = self.COMPATIBILITY.get(group_type, self.NETWORK_TYPES)
        net_type = random.choice(compatible)

        equiv_layer = GroupEquivariantLayer(group_type)
        irreps = equiv_layer.get_irreducible_representations()
        total_irrep_dim = sum(irrep['dim'] for irrep in irreps) or 1

        layers = [input_dim]
        for _ in range(num_layers):
            if group_type == 'permutation':
                layer_dim = total_irrep_dim * random.randint(2, 6)
            elif group_type == 'translation':
                layer_dim = max(1, len(equiv_layer.group_elements)) * random.randint(2, 8)
            elif group_type == 'rotation':
                layer_dim = 2 * random.randint(4, 16)
            else:
                layer_dim = random.randint(8, 32)
            layers.append(max(MIN_LAYER_SIZE, layer_dim))
        layers.append(output_dim)
        return layers, net_type, group_type


# ============================================================
# Search Algorithms
# ============================================================

@dataclass
class ArchitectureCandidate:
    layer_sizes: List[int]
    network_type: str
    group_type: str
    architecture_id: str = field(
        default_factory=lambda: f"arch_{random.randint(10000, 99999)}"
    )
    fitness: float = 0.0
    benchmark_results: Dict[str, Any] = field(default_factory=dict)
    theoretical_properties: Dict[str, Any] = field(default_factory=dict)
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ArchitectureSearchEngine:
    """Multi-strategy architecture search engine."""

    def __init__(self) -> None:
        self.group_generator = GroupTheoryArchitectureGenerator()
        self.population: List[ArchitectureCandidate] = []
        self.history: List[ArchitectureCandidate] = []
        self.best_architecture: Optional[ArchitectureCandidate] = None

    def random_search(
        self, num_candidates: int = 20,
        input_dim: int = 4, output_dim: int = 2
    ) -> List[ArchitectureCandidate]:
        candidates = []
        for _ in range(num_candidates):
            group_type = random.choice(
                ['permutation', 'translation', 'rotation', 'none']
            )
            layers, net_type, g_type = (
                self.group_generator.generate_equivariant_architecture(
                    group_type, input_dim, output_dim, random.randint(2, 4)
                )
            )
            candidates.append(ArchitectureCandidate(
                layer_sizes=layers, network_type=net_type, group_type=g_type
            ))
        return candidates

    def grid_search(
        self, input_dim: int = 4, output_dim: int = 2,
        max_candidates: int = 100
    ) -> List[ArchitectureCandidate]:
        candidates: List[ArchitectureCandidate] = []
        layer_options = [4, 8, 16, 32]
        for num_layers in [2, 3, 4]:
            for layer_sizes_idx in itertools.product(layer_options, repeat=num_layers):
                for group_type, compatible_nets in (
                    GroupTheoryArchitectureGenerator.COMPATIBILITY.items()
                ):
                    for net_type in compatible_nets:
                        layers = [input_dim] + list(layer_sizes_idx) + [output_dim]
                        candidates.append(ArchitectureCandidate(
                            layer_sizes=layers,
                            network_type=net_type,
                            group_type=group_type
                        ))
                        if len(candidates) >= max_candidates:
                            return candidates
        return candidates

    def evolutionary_search(
        self, population_size: int = 20, generations: int = 5,
        mutation_rate: float = 0.3, crossover_rate: float = 0.7,
        input_dim: int = 4, output_dim: int = 2,
        elitism: int = 2
    ) -> List[ArchitectureCandidate]:
        self.population = self.random_search(population_size, input_dim, output_dim)

        for gen in range(generations):
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            if (self.best_architecture is None
                or (self.population
                    and self.population[0].fitness > self.best_architecture.fitness)):
                self.best_architecture = self.population[0]

            elites = self.population[:elitism]
            offspring: List[ArchitectureCandidate] = list(elites)

            while len(offspring) < population_size:
                p1 = self._tournament_select(self.population, k=4)
                p2 = self._tournament_select(self.population, k=4)

                if random.random() < crossover_rate and p1 and p2:
                    child = self._crossover(p1, p2, gen + 1)
                else:
                    child = self._tournament_select(self.population, k=1).copy() if self.population else None

                if child and random.random() < mutation_rate:
                    child = self._mutate(child, gen + 1)

                if child:
                    offspring.append(child)

            self.population = offspring[:population_size]
            if self.population:
                best_in_gen = max(self.population, key=lambda x: x.fitness)
                self.history.append(best_in_gen)
        return self.history

    @staticmethod
    def _tournament_select(
        population: List[ArchitectureCandidate], k: int = 3
    ) -> ArchitectureCandidate:
        if not population:
            raise ValueError("Empty population for tournament")
        k = max(1, min(k, len(population)))
        sample = random.sample(population, k)
        return max(sample, key=lambda x: x.fitness)

    @staticmethod
    def _crossover(
        p1: ArchitectureCandidate, p2: ArchitectureCandidate, gen: int
    ) -> ArchitectureCandidate:
        max_len = max(len(p1.layer_sizes), len(p2.layer_sizes))
        child_layers = []
        for i in range(max_len):
            in_p1 = i < len(p1.layer_sizes)
            in_p2 = i < len(p2.layer_sizes)
            if in_p1 and in_p2:
                child_layers.append(random.choice([p1.layer_sizes[i], p2.layer_sizes[i]]))
            elif in_p1:
                child_layers.append(p1.layer_sizes[i])
            else:
                child_layers.append(p2.layer_sizes[i])
        fitter = p1 if p1.fitness > p2.fitness else p2
        return ArchitectureCandidate(
            layer_sizes=child_layers,
            network_type=fitter.network_type,
            group_type=fitter.group_type,
            generation=gen,
            parent_ids=[p1.architecture_id, p2.architecture_id]
        )

    @staticmethod
    def _mutate(
        candidate: ArchitectureCandidate, gen: int
    ) -> ArchitectureCandidate:
        layers = candidate.layer_sizes.copy()
        mutation = random.choice(['resize', 'add_layer', 'remove_layer', 'change_type'])
        if mutation == 'resize' and len(layers) > 2:
            idx = random.randint(1, len(layers) - 2)
            layers[idx] = max(MIN_LAYER_SIZE, layers[idx] + random.randint(-4, 4))
        elif mutation == 'add_layer' and len(layers) < MAX_HIDDEN_LAYERS + 2:
            layers.insert(random.randint(1, len(layers) - 1),
                          random.choice([4, 8, 16, 32]))
        elif mutation == 'remove_layer' and len(layers) > 3:
            layers.pop(random.randint(1, len(layers) - 2))
        elif mutation == 'change_type':
            compatible = GroupTheoryArchitectureGenerator.COMPATIBILITY.get(
                candidate.group_type, ('MLP',)
            )
            new_type = random.choice(compatible)
            return ArchitectureCandidate(
                layer_sizes=layers,
                network_type=new_type,
                group_type=candidate.group_type,
                generation=gen,
                parent_ids=[candidate.architecture_id]
            )
        return ArchitectureCandidate(
            layer_sizes=layers,
            network_type=candidate.network_type,
            group_type=candidate.group_type,
            generation=gen,
            parent_ids=[candidate.architecture_id]
        )

    def _candidate_to_features(self, cand: ArchitectureCandidate) -> np.ndarray:
        max_layers = MAX_HIDDEN_LAYERS
        layer_feats = np.zeros(max_layers, dtype=np.float64)
        ls = cand.layer_sizes[1:-1]
        layer_feats[:len(ls)] = np.log1p(ls)

        type_map = {'MLP': 1, 'CNN': 2, 'RNN': 3, 'GNN': 4, 'Transformer': 5}
        group_map = {'permutation': 1, 'translation': 2, 'rotation': 3, 'none': 4}
        type_feat = type_map.get(cand.network_type, 0)
        group_feat = group_map.get(cand.group_type, 0)
        return np.concatenate([layer_feats, [type_feat, group_feat]])

    @staticmethod
    def _predict_fitness(
        features: np.ndarray, hist_feats: List[np.ndarray],
        hist_fits: List[float], k: int = 3
    ) -> float:
        if not hist_feats:
            return 0.0
        feats_arr = np.array(hist_feats)
        dists = np.linalg.norm(feats_arr - features, axis=1)
        k = min(k, len(dists))
        nearest = np.argsort(dists)[:k]
        temp = max(dists[nearest].mean(), EPSILON)
        weights = np.exp(-dists[nearest] / temp)
        w_sum = np.sum(weights)
        if w_sum < EPSILON:
            return float(np.mean(np.array(hist_fits)[nearest]))
        return float(np.sum(weights * np.array(hist_fits)[nearest]) / w_sum)

    def _expected_improvement(
        self, pred: float, best_observed: float,
        candidate_features: np.ndarray, hist_feats: List[np.ndarray],
        hist_fits: List[float]
    ) -> float:
        if not hist_feats:
            return pred
        feats_arr = np.array(hist_feats)
        dists = np.linalg.norm(feats_arr - candidate_features, axis=1)
        sigma = max(float(np.std(dists)), EPSILON)
        improvement = max(0.0, pred - best_observed)
        z = improvement / (sigma + EPSILON)
        Phi_z = 0.5 * (1 + erf(z / sqrt(2)))
        return improvement * Phi_z + sigma * np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)

    def bayesian_optimization_search(
        self, num_iterations: int = 20,
        input_dim: int = 4, output_dim: int = 2
    ) -> List[ArchitectureCandidate]:
        candidates = self.random_search(5, input_dim, output_dim)
        for i in range(num_iterations - 5):
            evaluated = [c for c in candidates if c.fitness > 0]
            if evaluated:
                hist_feats = [self._candidate_to_features(c) for c in evaluated]
                hist_fits = [c.fitness for c in evaluated]
                best_observed = max(hist_fits)
                best_cand: Optional[ArchitectureCandidate] = None
                best_ei = -1.0
                for _ in range(10):
                    parent = max(evaluated, key=lambda x: x.fitness)
                    layers, net_type, _ = (
                        self.group_generator.generate_equivariant_architecture(
                            parent.group_type, input_dim, output_dim,
                            max(1, len(parent.layer_sizes) - 2)
                        )
                    )
                    perturb = max(1, int(5 * (1.0 - parent.fitness)))
                    layers = [max(MIN_LAYER_SIZE, l + random.randint(-perturb, perturb))
                              for l in layers]
                    cand = ArchitectureCandidate(
                        layer_sizes=layers,
                        network_type=net_type,
                        group_type=parent.group_type,
                        generation=i,
                        parent_ids=[parent.architecture_id]
                    )
                    pred = self._predict_fitness(
                        self._candidate_to_features(cand), hist_feats, hist_fits
                    )
                    ei = self._expected_improvement(
                        pred, best_observed,
                        self._candidate_to_features(cand), hist_feats, hist_fits
                    )
                    if ei > best_ei:
                        best_ei = ei
                        best_cand = cand
            else:
                best_cand = self.random_search(1, input_dim, output_dim)[0]
                best_cand.generation = i
            if best_cand is not None:
                candidates.append(best_cand)
        return candidates


# ============================================================
# Benchmark Suite
# ============================================================

class BenchmarkSuite:
    """Standardized benchmark tasks for evaluating architectures."""

    @staticmethod
    def xor_benchmark(
        network: 'TheoreticalNeuralNetwork', num_epochs: int = 100
    ) -> Dict[str, float]:
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
        y = np.array([[0], [1], [1], [0]], dtype=np.float64)
        losses = [network.backward(X, y, lr=0.1) for _ in range(num_epochs)]
        final_loss = losses[-1] if losses else 1.0
        accuracy = (1.0 if final_loss < 0.1
                    else (0.5 if final_loss < 0.25 else 0.0))
        rep_dim = network.get_algebraic_invariants().get('representation_dimension', 1)
        return {
            'final_loss': final_loss,
            'accuracy': accuracy,
            'convergence_speed': num_epochs / (final_loss + EPSILON),
            'parameter_efficiency': 1.0 / (rep_dim + EPSILON),
        }

    @staticmethod
    def spiral_benchmark(
        network: 'TheoreticalNeuralNetwork',
        num_samples: int = 100, num_epochs: int = 50
    ) -> Dict[str, float]:
        half = num_samples // 2
        theta = np.linspace(0, 4 * np.pi, half)
        r = np.linspace(0.5, 5.0, half)
        X = np.vstack([
            np.column_stack([r * np.cos(theta), r * np.sin(theta)]),
            np.column_stack([-r * np.cos(theta), -r * np.sin(theta)])
        ])
        y = np.vstack([
            np.zeros((half, 1)),
            np.ones((half, 1))
        ])
        losses = [network.backward(X, y, lr=0.01) for _ in range(num_epochs)]
        final_loss = losses[-1] if losses else 1.0
        return {
            'final_loss': final_loss,
            'accuracy': 1.0 if final_loss < 0.2 else 0.5,
            'convergence_speed': num_epochs / (final_loss + EPSILON),
        }

    @staticmethod
    def graph_isomorphism_benchmark(
        network: 'TheoreticalNeuralNetwork', num_graphs: int = 20
    ) -> Dict[str, float]:
        equiv_ok, total = 0, 0
        in_dim = network.layer_sizes[0]
        for _ in range(num_graphs):
            n = random.randint(4, 8)
            x = np.random.randn(n, in_dim)
            perm = np.random.permutation(n)
            x_perm = x[perm]
            try:
                out, out_perm = network.forward(x), network.forward(x_perm)
                diff = np.linalg.norm(out_perm - out[perm])
                norm = np.linalg.norm(out) + EPSILON
                if diff / norm < 0.1:
                    equiv_ok += 1
                total += 1
            except (ValueError, np.linalg.LinAlgError) as e:
                logger.debug(f"Graph iso benchmark skipped one case: {e}")
        acc = equiv_ok / total if total else 0.0
        return {
            'accuracy': acc,
            'final_loss': 1.0 - acc,
            'convergence_speed': 1.0,
        }

    @staticmethod
    def sequence_prediction_benchmark(
        network: 'TheoreticalNeuralNetwork',
        seq_length: int = 10, num_epochs: int = 50
    ) -> Dict[str, float]:
        X = np.random.randn(seq_length, 4)
        y = np.roll(X, -1, axis=0)
        y[-1] = np.random.randn(4)
        losses = [network.backward(X, y, lr=0.01) for _ in range(num_epochs)]
        final_loss = losses[-1] if losses else 1.0
        return {
            'final_loss': final_loss,
            'accuracy': 1.0 if final_loss < 0.3 else 0.0,
            'convergence_speed': num_epochs / (final_loss + EPSILON),
        }

    @staticmethod
    def run_all_benchmarks(network: 'TheoreticalNeuralNetwork') -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        benchmarks = [
            ('xor',      lambda: BenchmarkSuite.xor_benchmark(network)),
            ('spiral',   lambda: BenchmarkSuite.spiral_benchmark(network)),
            ('graph_iso',lambda: BenchmarkSuite.graph_isomorphism_benchmark(network)),
            ('sequence', lambda: BenchmarkSuite.sequence_prediction_benchmark(network)),
        ]
        for name, fn in benchmarks:
            try:
                results[name] = fn()
            except Exception as e:
                logger.warning(f"Benchmark '{name}' failed: {e}")
                results[name] = {
                    'accuracy': 0.0, 'final_loss': 1.0,
                    'convergence_speed': 0.0
                }
        accuracies = [r.get('accuracy', 0) for r in results.values()
                      if isinstance(r, dict)]
        results['overall_fitness'] = float(np.mean(accuracies)) if accuracies else 0.0
        return results


# ============================================================
# Theoretical Neural Network
# ============================================================

def block_diag(*arrs: np.ndarray) -> np.ndarray:
    if not arrs:
        return np.array([])
    shapes = np.array([a.shape for a in arrs])
    out = np.zeros(np.sum(shapes, axis=0))
    r, c = 0, 0
    for a in arrs:
        out[r:r+a.shape[0], c:c+a.shape[1]] = a
        r += a.shape[0]
        c += a.shape[1]
    return out


class TheoreticalNeuralNetwork:
    """Neural network with explicit quiver-theoretic and group-theoretic structure."""

    def __init__(
        self, layer_sizes: List[int], network_type: str = 'MLP',
        group_type: str = 'permutation',
        network_id: Optional[str] = None
    ) -> None:
        if not layer_sizes or any(s < 1 for s in layer_sizes):
            raise ValueError(f"Invalid layer_sizes: {layer_sizes}")
        self.layer_sizes = list(layer_sizes)
        self.network_type = network_type
        self.group_type = group_type
        self.network_id = network_id or f"net_{random.randint(1000, 9999)}"
        self.quiver_rep = QuiverRepresentation()
        self.equivariant_layer = GroupEquivariantLayer(group_type)
        self.algebraic_analyzer = AlgebraicGeometryAnalyzer()
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        self.activations: List[np.ndarray] = []
        self.zs: List[np.ndarray] = []
        self.moduli_space: Optional[str] = None
        self._ag_cache: Optional[Dict[str, Any]] = None
        self._build_network()

    def _build_network(self) -> None:
        builders = {
            'MLP': self._build_mlp,
            'CNN': self._build_cnn,
            'RNN': self._build_rnn,
            'GNN': self._build_gnn,
            'Transformer': self._build_transformer,
        }
        builder = builders.get(self.network_type, self._build_mlp)
        builder()
        self.quiver_rep._invalidate_caches()
        self._project_weights_to_equivariant()
        self.moduli_space = self.quiver_rep.compute_moduli_space()

    def _add_neuron(self, layer: int, idx: int, x: float, y: float) -> str:
        neuron_id = f"L{layer}_N{idx}"
        self.quiver_rep.add_neuron(neuron_id, layer, (x, y), vector_space_dim=1)
        return neuron_id

    def _add_edge(self, src: str, tgt: str, weight_val: float) -> None:
        self.quiver_rep.add_connection(src, tgt, np.array([[weight_val]], dtype=np.float64))

    def _init_weight(self, out_dim: int, in_dim: int) -> np.ndarray:
        return np.random.randn(out_dim, in_dim) * np.sqrt(2.0 / max(in_dim, 1))

    def _build_mlp(self) -> None:
        for l, s in enumerate(self.layer_sizes):
            for i in range(s):
                self._add_neuron(l, i, l * 150, i * 70)
        for i in range(len(self.layer_sizes) - 1):
            w = self._init_weight(self.layer_sizes[i+1], self.layer_sizes[i])
            b = np.zeros(self.layer_sizes[i+1])
            self.weights.append(w)
            self.biases.append(b)
            for j in range(self.layer_sizes[i]):
                for k in range(self.layer_sizes[i+1]):
                    self._add_edge(f"L{i}_N{j}", f"L{i+1}_N{k}", w[k, j])

    def _build_cnn(self) -> None:
        for l, s in enumerate(self.layer_sizes):
            for i in range(s):
                self._add_neuron(l, i, i * 80, l * 150)
        kernel = 3
        for i in range(len(self.layer_sizes) - 1):
            in_dim, out_dim = self.layer_sizes[i], self.layer_sizes[i+1]
            k = np.random.randn(out_dim, kernel) * np.sqrt(2.0 / max(in_dim, 1))
            w = np.zeros((out_dim, in_dim))
            for o in range(out_dim):
                for j in range(in_dim):
                    w[o, j] = k[o, (j - o) % kernel]
            b = np.zeros(out_dim)
            self.weights.append(w)
            self.biases.append(b)
            for o in range(out_dim):
                for j in range(in_dim):
                    self._add_edge(f"L{i}_N{j}", f"L{i+1}_N{o}", w[o, j])

    def _build_rnn(self) -> None:
        hs = self.layer_sizes[0]
        ts = len(self.layer_sizes)
        for t in range(ts):
            for i in range(hs):
                self._add_neuron(t, i, t * 150, i * 70)
        w_h = self._init_weight(hs, hs)
        self.weights.append(w_h)
        self.biases.append(np.zeros(hs))
        for t in range(ts):
            for j in range(hs):
                for k in range(hs):
                    self._add_edge(f"L{t}_N{j}", f"L{t}_N{k}", w_h[k, j])
                    if t < ts - 1:
                        self._add_edge(f"L{t}_N{j}", f"L{t+1}_N{k}", w_h[k, j])

    def _build_gnn(self) -> None:
        nn = max(2, self.layer_sizes[0])
        G = nx.random_geometric_graph(nn, 0.5, seed=42)
        pos = nx.spring_layout(G, seed=42) if G.number_of_nodes() > 0 else {}
        for l, s in enumerate(self.layer_sizes):
            for i in range(s):
                if l == 0 and i < len(pos):
                    x, y = pos[i] * 400 + 100
                else:
                    x, y = l * 200 + 100, i * 50 + 200
                self._add_neuron(l, i, x, y)
        for i in range(len(self.layer_sizes) - 1):
            w = self._init_weight(nn, nn)
            self.weights.append(w)
            self.biases.append(np.zeros(nn))
            for u, v in G.edges():
                if u < self.layer_sizes[i] and v < self.layer_sizes[i+1]:
                    self._add_edge(f"L{i}_N{u}", f"L{i+1}_N{v}", w[v, u])
                    self._add_edge(f"L{i}_N{v}", f"L{i+1}_N{u}", w[u, v])

    def _build_transformer(self) -> None:
        for l, s in enumerate(self.layer_sizes):
            for i in range(s):
                self._add_neuron(l, i, i * 100, l * 150)
        for i in range(len(self.layer_sizes) - 1):
            w = sum(
                self._init_weight(self.layer_sizes[i+1], self.layer_sizes[i])
                for _ in range(3)
            ) / 3.0
            self.weights.append(w)
            self.biases.append(np.zeros(self.layer_sizes[i+1]))
            for j in range(self.layer_sizes[i]):
                for k in range(self.layer_sizes[i+1]):
                    self._add_edge(f"L{i}_N{j}", f"L{i+1}_N{k}", w[k, j])

    def _get_representation(self, dim: int) -> Optional[Callable]:
        if self.group_type == 'permutation':
            n = DEFAULT_PERMUTATION_N
            def rep_perm(g: Tuple[int, ...]) -> np.ndarray:
                P = self.equivariant_layer._permutation_matrix(g)
                if dim > n:
                    return block_diag(P, np.eye(dim - n))
                elif dim < n:
                    return P[:dim, :dim]
                return P
            return rep_perm
        if self.group_type == 'translation':
            def rep_trans(g: np.ndarray) -> np.ndarray:
                if dim >= DEFAULT_TRANSLATION_N:
                    return block_diag(g, np.eye(dim - DEFAULT_TRANSLATION_N))
                return g[:dim, :dim]
            return rep_trans
        if self.group_type == 'rotation':
            def rep_rot(g: np.ndarray) -> np.ndarray:
                if dim % 2 == 0:
                    return block_diag(*[g] * (dim // 2))
                return block_diag(*([g] * (dim // 2) + [np.ones((1, 1))]))
            return rep_rot
        return None

    def _project_weights_to_equivariant(self) -> None:
        if self.group_type == 'none' or not self.weights:
            return
        n_elements = len(self.equivariant_layer.group_elements)
        if n_elements == 0:
            logger.warning("Empty group_elements; skipping equivariant projection")
            return

        for i in range(len(self.weights)):
            W = self.weights[i]
            rho_in = self._get_representation(W.shape[1])
            rho_out = self._get_representation(W.shape[0])
            if not rho_in or not rho_out:
                continue
            W_proj = np.zeros_like(W)
            for g in self.equivariant_layer.group_elements:
                g_in = rho_in(g)
                g_out = rho_out(g)
                W_proj += g_out @ W @ g_in.T
            W_proj /= n_elements
            self.weights[i] = W_proj

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        self.activations = [x]
        self.zs = []
        for w, b in zip(self.weights, self.biases):
            in_dim = w.shape[1]
            if x.shape[1] != in_dim:
                if x.shape[1] < in_dim:
                    x = np.pad(x, ((0, 0), (0, in_dim - x.shape[1])))
                else:
                    x = x[:, :in_dim]
            z = x @ w.T + b
            a = np.maximum(z, 0)  # ReLU
            self.zs.append(z)
            self.activations.append(a)
            x = a
        return x

    def backward(self, x: np.ndarray, y: np.ndarray, lr: float = 0.01) -> float:
        out = self.forward(x)
        if out.shape != y.shape:
            if out.shape[1] > y.shape[1]:
                y = np.pad(y, ((0, 0), (0, out.shape[1] - y.shape[1])))
            else:
                y = y[:, :out.shape[1]]
        if not self.weights:
            return float(np.mean((out - y) ** 2))

        delta = (out - y) / max(x.shape[0], 1)
        grads_w: List[Optional[np.ndarray]] = [None] * len(self.weights)
        grads_b: List[Optional[np.ndarray]] = [None] * len(self.biases)
        for i in reversed(range(len(self.weights))):
            z = self.zs[i]
            delta = delta * (z > 0).astype(np.float64)
            a_prev = self.activations[i]
            gw = delta.T @ a_prev
            gb = np.sum(delta, axis=0)
            gw_norm = np.linalg.norm(gw)
            if gw_norm > CLIP_NORM:
                gw = gw * (CLIP_NORM / gw_norm)
            grads_w[i] = gw
            grads_b[i] = gb
            delta = delta @ self.weights[i]

        for i in range(len(self.weights)):
            self.weights[i] -= lr * grads_w[i]
            self.biases[i] -= lr * grads_b[i]
        return float(np.mean((out - y) ** 2))

    def get_algebraic_invariants(self) -> Dict[str, Any]:
        return self.quiver_rep.get_algebraic_invariants()

    def get_moduli_space(self) -> Optional[str]:
        return self.moduli_space

    def analyze_with_algebraic_geometry(self) -> Dict[str, Any]:
        if self._ag_cache is None:
            self._ag_cache = self.algebraic_analyzer.compute_function_space_geometry(self.weights)
        return self._ag_cache

    def get_network_summary(self) -> Dict[str, Any]:
        ag = self.analyze_with_algebraic_geometry()
        return {
            'network_id': self.network_id,
            'network_type': self.network_type,
            'layer_sizes': self.layer_sizes,
            'group_type': self.group_type,
            'total_parameters': ag['total_parameters'],
            'algebraic_invariants': self.get_algebraic_invariants(),
            'moduli_space': self.moduli_space,
            'algebraic_geometry': ag,
            'irreducible_representations': self.equivariant_layer.get_irreducible_representations(),
        }


# ============================================================
# Multi-Network Vector Database
# ============================================================

class MultiNetworkVectorDB:
    """ChromaDB-backed vector store for network architectures."""

    EMBEDDING_DIM = EMBEDDING_DIM

    def __init__(self) -> None:
        self.client = chromadb.Client()
        try:
            self.client.delete_collection(name=COLLECTION_NAME)
        except chromadb.errors.InvalidCollectionError:
            pass
        except Exception as e:
            logger.warning(f"Unexpected error deleting collection: {e}")
        self.collection = self.client.create_collection(name=COLLECTION_NAME)
        self.network_metadata: Dict[str, Dict[str, Any]] = {}

    def _network_to_vector(self, network: TheoreticalNeuralNetwork, ag: Optional[Dict[str, Any]] = None) -> List[float]:
        features: List[float] = []
        sv_spectra = ag['singular_value_spectra'] if ag else None
        for i, w in enumerate(network.weights):
            if w.size == 0:
                features.extend([0.0] * 10)
                continue
            try:
                if sv_spectra and i < len(sv_spectra) and sv_spectra[i].size > 0:
                    top_sv = sv_spectra[i][:5]
                else:
                    w2d = w.reshape(w.shape[0], -1) if w.ndim > 2 else w
                    top_sv = np.linalg.svd(w2d, compute_uv=False)[:5]
                
                if len(top_sv) < 5:
                    top_sv = np.pad(top_sv, (0, 5 - len(top_sv)))
                features.extend([
                    float(np.mean(w)), float(np.std(w)),
                    float(np.linalg.norm(w)),
                    float(w.shape[0]),
                    float(w.shape[1] if w.ndim > 1 else 1)
                ])
                features.extend(top_sv.tolist())
            except np.linalg.LinAlgError:
                features.extend([0.0] * 10)
        if len(features) < self.EMBEDDING_DIM:
            features.extend([0.0] * (self.EMBEDDING_DIM - len(features)))
        else:
            features = features[:self.EMBEDDING_DIM]
        return features

    def store_network(
        self, network: TheoreticalNeuralNetwork,
        benchmark_results: Optional[Dict[str, Any]] = None
    ) -> None:
        net_id = network.network_id
        ag = network.analyze_with_algebraic_geometry()
        emb = self._network_to_vector(network, ag)
        inv = network.get_algebraic_invariants()
        
        def default_serializer(o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.integer, np.floating)):
                return o.item()
            return str(o)

        bstr = json.dumps(benchmark_results, default=default_serializer) if benchmark_results else "{}"
        fit = (benchmark_results or {}).get('overall_fitness', 0.0)
        if not isinstance(fit, (int, float)):
            fit = 0.0

        self.network_metadata[net_id] = {
            'network_type': network.network_type,
            'layer_sizes': network.layer_sizes,
            'group_type': network.group_type,
            'invariants': inv,
            'moduli_space': network.get_moduli_space(),
            'algebraic_geometry': ag,
            'irreducible_representations': network.equivariant_layer.get_irreducible_representations(),
            'total_parameters': ag['total_parameters'],
            'benchmark_results': benchmark_results,
        }
        self.collection.upsert(
            embeddings=[emb],
            documents=[f"Network {net_id} ({network.network_type})"],
            metadatas=[{
                "network_id": net_id,
                "network_type": network.network_type,
                "layer_sizes": json.dumps(network.layer_sizes),
                "group_type": network.group_type,
                "total_parameters": int(ag['total_parameters']),
                "euler_characteristic": int(inv.get('euler_characteristic', 0)),
                "cycle_rank": int(inv.get('cycle_rank', 0)),
                "variety_dimension": int(ag.get('variety_dimension', 0)),
                "algebraic_complexity": float(ag.get('algebraic_complexity', 0.0)),
                "benchmark_results": bstr,
                "overall_fitness": float(fit),
            }],
            ids=[net_id]
        )

    def query_similar_networks(
        self, query_net_id: str, n_results: int = 5
    ) -> List[Dict[str, Any]]:
        try:
            res = self.collection.get(ids=[query_net_id])
            emb = res.get('embeddings') or []
            if not emb:
                return []
            res = self.collection.query(
                query_embeddings=[emb[0]],
                n_results=n_results + 1
            )
        except Exception as e:
            logger.warning(f"Query failed: {e}")
            return []
        sims = []
        for idx, net_id in enumerate(res['ids'][0]):
            if net_id == query_net_id:
                continue
            meta = res['metadatas'][0][idx]
            try:
                ls = json.loads(meta.get('layer_sizes', '[]'))
            except json.JSONDecodeError:
                ls = []
            sims.append({
                'network_id': net_id,
                'distance': float(res['distances'][0][idx]),
                'network_type': meta.get('network_type', 'unknown'),
                'layer_sizes': ls,
                'group_type': meta.get('group_type', 'unknown'),
                'total_parameters': int(meta.get('total_parameters', 0) or 0),
                'overall_fitness': float(meta.get('overall_fitness', 0) or 0),
            })
        return sims[:n_results]

    def get_all_networks(self) -> List[Dict[str, Any]]:
        try:
            res = self.collection.get()
        except Exception as e:
            logger.warning(f"get_all_networks failed: {e}")
            return []
        nets = []
        for idx, net_id in enumerate(res['ids']):
            meta = res['metadatas'][idx]
            try:
                ls = json.loads(meta.get('layer_sizes', '[]'))
            except json.JSONDecodeError:
                ls = []
            nets.append({
                'network_id': net_id,
                'network_type': meta.get('network_type', 'unknown'),
                'layer_sizes': ls,
                'group_type': meta.get('group_type', 'unknown'),
                'total_parameters': int(meta.get('total_parameters', 0) or 0),
                'overall_fitness': float(meta.get('overall_fitness', 0) or 0),
            })
        return nets

    def get_network_metadata(self, net_id: str) -> Optional[Dict[str, Any]]:
        return self.network_metadata.get(net_id)

    def clear_database(self) -> None:
        try:
            self.client.delete_collection(name=COLLECTION_NAME)
        except chromadb.errors.InvalidCollectionError:
            pass
        except Exception as e:
            logger.warning(f"Unexpected error in clear_database: {e}")
        self.collection = self.client.create_collection(name=COLLECTION_NAME)
        self.network_metadata.clear()


# ============================================================
# NAS Worker Thread
# ============================================================

class NASWorker(QThread):
    """Background thread for running NAS algorithms."""
    progress = Signal(int, str)
    finished = Signal(list)

    def __init__(
        self, search_algorithm: str, num_iterations: int,
        input_dim: int, output_dim: int
    ) -> None:
        super().__init__()
        self.search_algorithm = search_algorithm
        self.num_iterations = num_iterations
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.search_engine = ArchitectureSearchEngine()
        self.benchmark_suite = BenchmarkSuite()
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def _evaluate_candidate(self, cand: ArchitectureCandidate) -> None:
        try:
            net = TheoreticalNeuralNetwork(
                layer_sizes=cand.layer_sizes,
                network_type=cand.network_type,
                group_type=cand.group_type,
                network_id=cand.architecture_id
            )
            res = self.benchmark_suite.run_all_benchmarks(net)
            cand.fitness = float(res.get('overall_fitness', 0.0))
            cand.benchmark_results = res
            cand.theoretical_properties = net.get_network_summary()
        except Exception as e:
            logger.error(f"Eval failed {cand.architecture_id}: {e}", exc_info=True)
            cand.fitness = 0.0
            cand.benchmark_results = {'overall_fitness': 0.0}

    def run(self) -> None:
        try:
            candidates: List[ArchitectureCandidate] = []
            self.progress.emit(5, "Initializing search...")

            if self.search_algorithm == "RL Search":
                self._run_rl_search(candidates)
            else:
                if self.search_algorithm == "Random Search":
                    self.progress.emit(10, "Generating random architectures...")
                    candidates = self.search_engine.random_search(
                        self.num_iterations, self.input_dim, self.output_dim
                    )
                elif self.search_algorithm == "Grid Search":
                    self.progress.emit(10, "Running grid search...")
                    candidates = self.search_engine.grid_search(
                        self.input_dim, self.output_dim
                    )
                elif self.search_algorithm == "Evolutionary":
                    self.progress.emit(10, "Running evolutionary algorithm...")
                    candidates = self.search_engine.evolutionary_search(
                        population_size=self.num_iterations,
                        generations=3,
                        input_dim=self.input_dim,
                        output_dim=self.output_dim
                    )
                elif self.search_algorithm == "Bayesian Optimization":
                    self.progress.emit(10, "Running Bayesian optimization...")
                    candidates = self.search_engine.bayesian_optimization_search(
                        self.num_iterations, self.input_dim, self.output_dim
                    )
                else:
                    self.progress.emit(10, f"Unknown algorithm '{self.search_algorithm}', defaulting to Random.")
                    candidates = self.search_engine.random_search(self.num_iterations, self.input_dim, self.output_dim)

                total = max(1, len(candidates))
                for i, cand in enumerate(candidates):
                    if self._cancel:
                        self.progress.emit(100, "Search cancelled.")
                        self.finished.emit(candidates[:i])
                        return
                    self.progress.emit(
                        int(10 + (i / total) * 85),
                        f"Evaluating {cand.architecture_id} ({i+1}/{total})..."
                    )
                    self._evaluate_candidate(cand)

            self.progress.emit(100, "Search complete!")
            self.finished.emit(candidates)
        except Exception as e:
            logger.exception(f"NAS worker crashed: {e}")
            self.progress.emit(100, f"Search failed: {e}")
            self.finished.emit([])

    def _run_rl_search(self, candidates: List[ArchitectureCandidate]) -> None:
        """REINFORCE with moving baseline, entropy regularization, std floor."""
        types = ['MLP', 'CNN', 'RNN', 'GNN', 'Transformer']
        groups = ['permutation', 'translation', 'rotation', 'none']

        prefs = {
            'width_mu': 16.0, 'width_std': 8.0,
            'depth_mu': 3.0, 'depth_std': 2.0,
        }
        type_probs = np.ones(len(types)) / len(types)
        group_probs = np.ones(len(groups)) / len(groups)

        baseline = 0.0
        lr = RL_LEARNING_RATE

        for i in range(self.num_iterations):
            if self._cancel:
                break
            self.progress.emit(
                int(10 + (i / self.num_iterations) * 85),
                f"RL Step {i+1}/{self.num_iterations}..."
            )

            depth = max(1, int(np.random.normal(prefs['depth_mu'], prefs['depth_std'])))
            width = max(MIN_LAYER_SIZE, int(np.random.normal(prefs['width_mu'], prefs['width_std'])))
            t_idx = int(np.random.choice(len(types), p=type_probs))
            g_idx = int(np.random.choice(len(groups), p=group_probs))

            net_type, group_type = types[t_idx], groups[g_idx]
            layers = [self.input_dim] + [width] * depth + [self.output_dim]
            cand = ArchitectureCandidate(
                layer_sizes=layers,
                network_type=net_type,
                group_type=group_type,
                generation=i
            )
            self._evaluate_candidate(cand)
            candidates.append(cand)

            reward = cand.fitness
            advantage = reward - baseline
            if len(candidates) > 1:
                recent_rewards = [c.fitness for c in candidates[-10:]]
                adv_std = max(np.std(recent_rewards), EPSILON)
                if adv_std < 1e-3:
                    continue
                advantage = advantage / adv_std

            prefs['width_mu'] += lr * advantage * (width - prefs['width_mu']) / (prefs['width_std']**2 + EPSILON)
            prefs['depth_mu'] += lr * advantage * (depth - prefs['depth_mu']) / (prefs['depth_std']**2 + EPSILON)

            prefs['width_std'] = max(RL_STD_FLOOR, prefs['width_std'] * RL_STD_DECAY)
            prefs['depth_std'] = max(RL_STD_FLOOR, prefs['depth_std'] * RL_STD_DECAY)

            mask_t = np.zeros(len(types))
            mask_t[t_idx] = 1.0
            entropy_grad_t = -(np.log(type_probs + EPSILON) + 1)
            grad_t = advantage * (mask_t - type_probs) + RL_ENTROPY_BONUS * entropy_grad_t
            type_probs = np.clip(type_probs + lr * grad_t, RL_PROB_FLOOR, None)
            type_probs /= np.sum(type_probs)

            mask_g = np.zeros(len(groups))
            mask_g[g_idx] = 1.0
            entropy_grad_g = -(np.log(group_probs + EPSILON) + 1)
            grad_g = advantage * (mask_g - group_probs) + RL_ENTROPY_BONUS * entropy_grad_g
            group_probs = np.clip(group_probs + lr * grad_g, RL_PROB_FLOOR, None)
            group_probs /= np.sum(group_probs)

            baseline = RL_BASELINE_DECAY * baseline + (1 - RL_BASELINE_DECAY) * reward


# ============================================================
# GUI
# ============================================================

class NeuronItem(QGraphicsEllipseItem):
    """Visual representation of a neuron in the architecture graph."""

    def __init__(
        self, neuron_id: str, x: float, y: float,
        radius: int = NEURON_RADIUS, color: str = "#4a90e2",
        show_label: bool = True
    ) -> None:
        super().__init__(x - radius, y - radius, radius * 2, radius * 2)
        self.setBrush(QBrush(QColor(color)))
        self.setPen(QPen(QColor("#ffffff"), 2))
        if show_label:
            text = QGraphicsTextItem(neuron_id.split("_")[-1], self)
            text_rect = text.boundingRect()
            text.setPos(x - text_rect.width() / 2, y - text_rect.height() / 2)
            text.setDefaultTextColor(QColor("white"))
            text.setFont(QFont("Arial", 6, QFont.Bold))


class NASGraphicsView(QGraphicsView):
    """Zoomable, pannable graphics view."""

    def __init__(self, scene: QGraphicsScene) -> None:
        super().__init__(scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

    def wheelEvent(self, event) -> None:
        zoom = 1.15
        if event.angleDelta().y() > 0:
            self.scale(zoom, zoom)
        else:
            self.scale(1 / zoom, 1 / zoom)


class NASApplication(QMainWindow):
    """Main application window for the NAS toolkit."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Neural Architecture Search: Set Theory & Group Theory")
        self.resize(1800, 1100)

        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        self.db = MultiNetworkVectorDB()
        self.networks: Dict[str, TheoreticalNeuralNetwork] = {}
        self.benchmark_suite = BenchmarkSuite()
        self.nas_worker: Optional[NASWorker] = None
        self.search_results: List[ArchitectureCandidate] = []
        self.search_history: List[Dict[str, Any]] = []

        self._build_menu_bar()
        self._setup_shortcuts()
        self.init_ui()
        self.create_default_networks()

    def _build_menu_bar(self) -> None:
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&File")
        export_action = QAction("&Export Results...", self)
        export_action.setShortcut(QKeySequence("Ctrl+E"))
        export_action.triggered.connect(self.export_results)
        file_menu.addAction(export_action)

        import_action = QAction("&Import Results...", self)
        import_action.setShortcut(QKeySequence("Ctrl+I"))
        import_action.triggered.connect(self.import_results)
        file_menu.addAction(import_action)

        file_menu.addSeparator()
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence("Ctrl+Q"))
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        search_menu = menubar.addMenu("&Search")
        self.start_action = QAction("&Start Search", self)
        self.start_action.setShortcut(QKeySequence("Ctrl+S"))
        self.start_action.triggered.connect(self.start_search)
        search_menu.addAction(self.start_action)

        self.cancel_action = QAction("&Cancel Search", self)
        self.cancel_action.setShortcut(QKeySequence("Ctrl+C"))
        self.cancel_action.triggered.connect(self.cancel_search)
        search_menu.addAction(self.cancel_action)

        view_menu = menubar.addMenu("&View")
        redraw_action = QAction("&Redraw Visualization", self)
        redraw_action.setShortcut(QKeySequence("Ctrl+R"))
        redraw_action.triggered.connect(self.redraw_current)
        view_menu.addAction(redraw_action)

        help_menu = menubar.addMenu("&Help")
        about_action = QAction("&About", self)
        about_action.setShortcut(QKeySequence("F1"))
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _setup_shortcuts(self) -> None:
        QShortcut(QKeySequence("Ctrl+G"), self, activated=self.generate_architecture)
        QShortcut(QKeySequence("Ctrl+F"), self, activated=self.find_similar_networks)

    def _show_about(self) -> None:
        QMessageBox.about(
            self, "About NAS",
            "<h3>Neural Architecture Search</h3>"
            "<p>A research toolkit combining quiver representation theory, "
            "group equivariance, and algebraic geometry for NAS.</p>"
            "<p>Algorithms: Random, Grid, Evolutionary, Bayesian, RL (REINFORCE)</p>"
        )

    def init_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._create_left_panel())
        splitter.addWidget(self._create_center_panel())
        splitter.addWidget(self._create_right_panel())
        splitter.setSizes([350, 700, 500])
        main_layout.addWidget(splitter)

    def _create_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        title = QLabel("Architecture Search")
        title.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)

        config_group = QGroupBox("Search Configuration")
        config_layout = QFormLayout(config_group)
        self.search_algo_combo = QComboBox()
        self.search_algo_combo.addItems([
            "Random Search", "Grid Search", "Evolutionary",
            "Bayesian Optimization", "RL Search"
        ])
        config_layout.addRow("Algorithm:", self.search_algo_combo)
        self.num_iterations_spin = QSpinBox()
        self.num_iterations_spin.setRange(5, 50)
        self.num_iterations_spin.setValue(10)
        config_layout.addRow("Iterations:", self.num_iterations_spin)
        self.input_dim_spin = QSpinBox()
        self.input_dim_spin.setRange(2, 16)
        self.input_dim_spin.setValue(4)
        config_layout.addRow("Input Dim:", self.input_dim_spin)
        self.output_dim_spin = QSpinBox()
        self.output_dim_spin.setRange(1, 8)
        self.output_dim_spin.setValue(2)
        config_layout.addRow("Output Dim:", self.output_dim_spin)
        layout.addWidget(config_group)

        btn_row = QHBoxLayout()
        self.search_btn = QPushButton("Start Search")
        self.search_btn.setStyleSheet(
            "background-color: #4a90e2; color: white; padding: 10px; font-weight: bold;"
        )
        self.search_btn.clicked.connect(self.start_search)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_search)
        btn_row.addWidget(self.search_btn)
        btn_row.addWidget(self.cancel_btn)
        layout.addLayout(btn_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        self.progress_label = QLabel("")
        self.progress_label.setVisible(False)
        layout.addWidget(self.progress_label)

        gen_group = QGroupBox("Manual Architecture Generation")
        gen_layout = QFormLayout(gen_group)
        self.gen_method_combo = QComboBox()
        self.gen_method_combo.addItems([
            "Set Theory - Balanced", "Set Theory - Decreasing",
            "Set Theory - Union Partition", "Set Theory - Power Set",
            "Group Theory - Equivariant"
        ])
        gen_layout.addRow("Method:", self.gen_method_combo)
        self.gen_btn = QPushButton("Generate Architecture")
        self.gen_btn.clicked.connect(self.generate_architecture)
        gen_layout.addWidget(self.gen_btn)
        layout.addWidget(gen_group)

        layout.addWidget(QLabel("Network Database:"))
        self.network_list = QListWidget()
        self.network_list.itemSelectionChanged.connect(self.on_network_selected)
        layout.addWidget(self.network_list)

        stats_group = QGroupBox("Database Statistics")
        stats_layout = QFormLayout(stats_group)
        self.total_nets_label = QLabel("0")
        stats_layout.addRow("Total Networks:", self.total_nets_label)
        self.avg_fitness_label = QLabel("0.0")
        stats_layout.addRow("Avg Fitness:", self.avg_fitness_label)
        self.best_fitness_label = QLabel("0.0")
        stats_layout.addRow("Best Fitness:", self.best_fitness_label)
        layout.addWidget(stats_group)

        io_row = QHBoxLayout()
        self.export_btn = QPushButton("Export Results")
        self.export_btn.clicked.connect(self.export_results)
        io_row.addWidget(self.export_btn)
        self.import_btn = QPushButton("Import Results")
        self.import_btn.clicked.connect(self.import_results)
        io_row.addWidget(self.import_btn)
        layout.addLayout(io_row)

        self.clear_db_btn = QPushButton("Clear Database")
        self.clear_db_btn.clicked.connect(self.clear_database)
        layout.addWidget(self.clear_db_btn)
        layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        wrapper = QWidget()
        wl = QVBoxLayout(wrapper)
        wl.setContentsMargins(0, 0, 0, 0)
        wl.addWidget(scroll)
        return wrapper

    def _create_center_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.tab_widget = QTabWidget()
        self.tab_widget.currentChanged.connect(self._on_tab_changed)

        results_tab = QWidget()
        results_layout = QVBoxLayout(results_tab)
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(8)
        self.results_table.setHorizontalHeaderLabels(
            ["ID", "Type", "Group", "Layers", "Params", "Fitness", "XOR Acc", "Spiral Acc"]
        )
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.results_table.setSelectionMode(QTableWidget.SingleSelection)
        self.results_table.itemSelectionChanged.connect(self.on_result_row_selected)
        results_layout.addWidget(self.results_table)
        self.tab_widget.addTab(results_tab, "Search Results")

        benchmark_tab = QWidget()
        benchmark_layout = QVBoxLayout(benchmark_tab)
        self.benchmark_table = QTableWidget()
        self.benchmark_table.setColumnCount(6)
        self.benchmark_table.setHorizontalHeaderLabels(
            ["Network ID", "XOR Loss", "Spiral Loss", "Graph Iso", "Sequence", "Overall"]
        )
        self.benchmark_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        benchmark_layout.addWidget(self.benchmark_table)
        self.tab_widget.addTab(benchmark_tab, "Benchmark Results")

        viz_tab = QWidget()
        viz_layout = QVBoxLayout(viz_tab)
        self.viz_scene = QGraphicsScene(self)
        self.viz_scene.setSceneRect(0, 0, 800, 600)
        self.viz_view = NASGraphicsView(self.viz_scene)
        self.viz_view.setStyleSheet("background-color: #1e1e1e;")
        viz_layout.addWidget(self.viz_view)

        legend = QHBoxLayout()
        legend.addWidget(QLabel("Legend:"))
        pos_dot = QLabel("●"); pos_dot.setStyleSheet("color:#64c8ff;font-size:18px;")
        legend.addWidget(pos_dot)
        legend.addWidget(QLabel(" Positive weight   "))
        neg_dot = QLabel("●"); neg_dot.setStyleSheet("color:#ff6464;font-size:18px;")
        legend.addWidget(neg_dot)
        legend.addWidget(QLabel(" Negative weight"))
        legend.addStretch()
        viz_layout.addLayout(legend)

        self.tab_widget.addTab(viz_tab, "Architecture Viz")

        history_tab = QWidget()
        history_layout = QVBoxLayout(history_tab)
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(5)
        self.history_table.setHorizontalHeaderLabels(
            ["Run #", "Algorithm", "Best Fitness", "Avg Fitness", "# Candidates"]
        )
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        history_layout.addWidget(self.history_table)
        self.tab_widget.addTab(history_tab, "Search History")

        layout.addWidget(self.tab_widget)
        return panel

    def _create_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        title = QLabel("Architecture Details")
        title.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setStyleSheet(
            "background-color: #2e2e2e; color: #00ff00; font-family: monospace;"
        )
        layout.addWidget(self.details_text)

        theory_group = QGroupBox("Theoretical Properties")
        theory_layout = QFormLayout(theory_group)
        self.theory_table = QTableWidget()
        self.theory_table.setColumnCount(2)
        self.theory_table.setHorizontalHeaderLabels(["Property", "Value"])
        self.theory_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        theory_layout.addRow(self.theory_table)
        layout.addWidget(theory_group)

        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_group)
        self.store_btn = QPushButton("Store Current in Database")
        self.store_btn.clicked.connect(self.store_current_network)
        actions_layout.addWidget(self.store_btn)
        self.query_similar_btn = QPushButton("Find Similar Networks")
        self.query_similar_btn.clicked.connect(self.find_similar_networks)
        actions_layout.addWidget(self.query_similar_btn)
        self.redraw_btn = QPushButton("Redraw Visualization")
        self.redraw_btn.clicked.connect(self.redraw_current)
        actions_layout.addWidget(self.redraw_btn)
        layout.addWidget(actions_group)
        return panel

    def _set_search_running(self, running: bool) -> None:
        self.search_btn.setEnabled(not running)
        self.cancel_btn.setEnabled(running)
        for w in [self.gen_btn, self.export_btn, self.import_btn, self.clear_db_btn,
                  self.search_algo_combo, self.num_iterations_spin,
                  self.input_dim_spin, self.output_dim_spin]:
            w.setEnabled(not running)
        self.start_action.setEnabled(not running)
        self.cancel_action.setEnabled(running)

    def start_search(self) -> None:
        if self.nas_worker and self.nas_worker.isRunning():
            self.status_bar.showMessage("A search is already running.", 3000)
            return
        algo = self.search_algo_combo.currentText()
        iters = self.num_iterations_spin.value()
        ind = self.input_dim_spin.value()
        outd = self.output_dim_spin.value()
        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self._set_search_running(True)
        self.status_bar.showMessage(f"Search started: {algo}")

        self.nas_worker = NASWorker(algo, iters, ind, outd)
        self.nas_worker.progress.connect(self.update_progress)
        self.nas_worker.finished.connect(self.on_search_complete)
        self.nas_worker.start()

    def cancel_search(self) -> None:
        if self.nas_worker and self.nas_worker.isRunning():
            self.nas_worker.cancel()
            self.cancel_btn.setEnabled(False)
            self.status_bar.showMessage("Cancelling search...")

    def update_progress(self, value: int, message: str) -> None:
        self.progress_bar.setValue(value)
        self.progress_label.setText(message)
        self.status_bar.showMessage(message)

    def _populate_results_tables(self, candidates: List[ArchitectureCandidate]) -> None:
        self.results_table.setRowCount(len(candidates))
        for i, c in enumerate(candidates):
            self.results_table.setItem(i, 0, QTableWidgetItem(c.architecture_id))
            self.results_table.setItem(i, 1, QTableWidgetItem(c.network_type))
            self.results_table.setItem(i, 2, QTableWidgetItem(c.group_type))
            self.results_table.setItem(i, 3, QTableWidgetItem(str(c.layer_sizes)))
            tp = c.theoretical_properties or {}
            self.results_table.setItem(i, 4, QTableWidgetItem(str(tp.get('total_parameters', 0))))
            self.results_table.setItem(i, 5, QTableWidgetItem(f"{c.fitness:.4f}"))
            xor_acc = (c.benchmark_results or {}).get('xor', {}).get('accuracy', 0)
            spiral_acc = (c.benchmark_results or {}).get('spiral', {}).get('accuracy', 0)
            self.results_table.setItem(i, 6, QTableWidgetItem(f"{xor_acc:.4f}"))
            self.results_table.setItem(i, 7, QTableWidgetItem(f"{spiral_acc:.4f}"))

        self.benchmark_table.setRowCount(len(candidates))
        for i, c in enumerate(candidates):
            self.benchmark_table.setItem(i, 0, QTableWidgetItem(c.architecture_id))
            for j, bn in enumerate(['xor', 'spiral', 'graph_iso', 'sequence'], 1):
                r = (c.benchmark_results or {}).get(bn, {})
                loss = r.get('final_loss', 1.0) if isinstance(r, dict) else 1.0
                try:
                    self.benchmark_table.setItem(i, j, QTableWidgetItem(f"{float(loss):.4f}"))
                except (TypeError, ValueError):
                    self.benchmark_table.setItem(i, j, QTableWidgetItem("N/A"))
            self.benchmark_table.setItem(i, 5, QTableWidgetItem(f"{c.fitness:.4f}"))

    def on_search_complete(self, candidates: List[ArchitectureCandidate]) -> None:
        self._set_search_running(False)
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self.search_results = candidates
        self._populate_results_tables(candidates)

        best_cands = sorted(candidates, key=lambda x: x.fitness, reverse=True)[:5]
        for c in best_cands:
            try:
                net = TheoreticalNeuralNetwork(
                    layer_sizes=c.layer_sizes,
                    network_type=c.network_type,
                    group_type=c.group_type,
                    network_id=c.architecture_id
                )
                self.networks[c.architecture_id] = net
                self.db.store_network(net, c.benchmark_results)
            except Exception as e:
                logger.error(f"Failed storing {c.architecture_id}: {e}")

        if len(self.networks) > MAX_STORED_NETWORKS:
            nets_by_fit = sorted(self.networks.values(), key=lambda n: self.db.get_network_metadata(n.network_id).get('benchmark_results', {}).get('overall_fitness', 0) if self.db.get_network_metadata(n.network_id) else 0, reverse=True)
            self.networks = {n.network_id: n for n in nets_by_fit[:MAX_STORED_NETWORKS]}

        self.update_network_list()
        self.update_stats()
        if candidates:
            best_fit = max(c.fitness for c in candidates)
            avg_fit = float(np.mean([c.fitness for c in candidates])) if candidates else 0.0
            self.search_history.append({
                'run': len(self.search_history) + 1,
                'algorithm': self.search_algo_combo.currentText(),
                'best_fitness': best_fit,
                'avg_fitness': avg_fit,
                'num_candidates': len(candidates)
            })
            self._refresh_history_table()
            self.status_bar.showMessage(
                f"Search complete. Found {len(candidates)} architectures. "
                f"Best fitness: {best_fit:.4f}", 10000
            )
        else:
            self.status_bar.showMessage("Search complete. No candidates returned.", 5000)

    def _on_tab_changed(self, idx: int) -> None:
        if idx in (0, 1):
            self.on_result_row_selected()

    def _refresh_history_table(self) -> None:
        self.history_table.setRowCount(len(self.search_history))
        for i, h in enumerate(self.search_history):
            self.history_table.setItem(i, 0, QTableWidgetItem(str(h['run'])))
            self.history_table.setItem(i, 1, QTableWidgetItem(h['algorithm']))
            self.history_table.setItem(i, 2, QTableWidgetItem(f"{h['best_fitness']:.4f}"))
            self.history_table.setItem(i, 3, QTableWidgetItem(f"{h['avg_fitness']:.4f}"))
            self.history_table.setItem(i, 4, QTableWidgetItem(str(h['num_candidates'])))

    def on_result_row_selected(self) -> None:
        rows = self.results_table.selectionModel().selectedRows()
        if not rows:
            return
        row = rows[0].row()
        arch_id = self.results_table.item(row, 0).text()
        cand = next((c for c in self.search_results if c.architecture_id == arch_id), None)
        if not cand:
            return
        if arch_id not in self.networks:
            try:
                self.networks[arch_id] = TheoreticalNeuralNetwork(
                    layer_sizes=cand.layer_sizes,
                    network_type=cand.network_type,
                    group_type=cand.group_type,
                    network_id=cand.architecture_id
                )
            except Exception as e:
                logger.error(f"Failed building {arch_id}: {e}")
                return
        self.show_network_details(arch_id, self.networks[arch_id], cand.benchmark_results)
        self.draw_network(self.networks[arch_id])

    def generate_architecture(self) -> None:
        method = self.gen_method_combo.currentText()
        ind = self.input_dim_spin.value()
        outd = self.output_dim_spin.value()
        try:
            if "Set Theory" in method:
                con = method.split(" - ", 1)[1].lower().replace(" ", "_")
                layers = SetTheoryArchitectureGenerator.generate_by_cardinality(
                    ind, outd, max_hidden_layers=3, cardinality_constraint=con
                )
                nt, gt = "MLP", "permutation"
            else:
                layers, nt, gt = GroupTheoryArchitectureGenerator().generate_equivariant_architecture(
                    "permutation", ind, outd, 3
                )
            nid = f"gen_{len(self.networks) + 1}"
            net = TheoreticalNeuralNetwork(layers, nt, gt, nid)
            res = self.benchmark_suite.run_all_benchmarks(net)
            self.networks[nid] = net
            self.db.store_network(net, res)
            self.update_network_list()
            self.update_stats()
            self.show_network_details(nid, net, res)
            self.draw_network(net)
            self.status_bar.showMessage(f"Generated and stored network {nid}", 5000)
        except Exception as e:
            self.status_bar.showMessage(f"Generation failed: {e}", 5000)
            logger.exception("Generation failed")

    def show_network_details(
        self, nid: str, net: TheoreticalNeuralNetwork,
        res: Optional[Dict[str, Any]] = None
    ) -> None:
        s = net.get_network_summary()
        d = (f"Network ID: {nid}\nType: {s['network_type']}\n"
             f"Layer Sizes: {s['layer_sizes']}\nGroup Type: {s['group_type']}\n"
             f"Total Parameters: {s['total_parameters']}\n"
             f"Moduli Space: {s['moduli_space']}\n\n")
        if res:
            d += "Benchmark Results:\n"
            for bn, r in res.items():
                if isinstance(r, dict):
                    d += f"  {bn}:\n"
                    for k, v in r.items():
                        try:
                            d += f"    {k}: {float(v):.4f}\n"
                        except (TypeError, ValueError):
                            d += f"    {k}: {v}\n"
                else:
                    d += f"  {bn}: {r}\n"
        d += "\nAlgebraic Invariants:\n"
        for k, v in s['algebraic_invariants'].items():
            d += f"  {k}: {v}\n"
        self.details_text.setText(d)

        self.theory_table.setRowCount(0)
        props = [
            ("Network Type", s['network_type']),
            ("Group Type", s['group_type']),
            ("Total Parameters", str(s['total_parameters'])),
            ("Euler Characteristic", str(s['algebraic_invariants'].get('euler_characteristic', 'N/A'))),
            ("Cycle Rank", str(s['algebraic_invariants'].get('cycle_rank', 'N/A'))),
            ("Variety Dimension", str(s['algebraic_geometry'].get('variety_dimension', 'N/A'))),
            ("Algebraic Complexity", f"{s['algebraic_geometry'].get('algebraic_complexity', 0):.4f}"),
            ("Moduli Space", str(s['moduli_space'])),
            ("Weakly Connected", str(s['algebraic_invariants'].get('is_weakly_connected', 'N/A'))),
            ("Diameter", str(s['algebraic_invariants'].get('diameter', 'N/A'))),
        ]
        self.theory_table.setRowCount(len(props))
        for i, (n, v) in enumerate(props):
            self.theory_table.setItem(i, 0, QTableWidgetItem(n))
            self.theory_table.setItem(i, 1, QTableWidgetItem(v))

    def draw_network(self, net: TheoreticalNeuralNetwork) -> None:
        self.viz_scene.clear()
        q = net.quiver_rep.quiver

        if q.number_of_edges() == 0:
            max_w = 1.0
            edge_cull_threshold = 0.0
        else:
            all_weights = [abs(float(d['weight'].flatten()[0])) for _, _, d in q.edges(data=True) if d.get('weight') is not None]
            max_w = max(all_weights) if all_weights else 1.0
            max_w = max(max_w, EPSILON)
            if q.number_of_edges() > EDGE_CULL_DENSE_THRESHOLD:
                edge_cull_threshold = np.percentile(all_weights, 80)
            else:
                edge_cull_threshold = 0.0

        for u, v, d in q.edges(data=True):
            w = d.get('weight', np.array([[0.0]]))
            wv = float(w.flatten()[0])
            nw = abs(wv) / max_w

            if nw < edge_cull_threshold:
                continue

            alpha = int(np.clip(nw * 255, 20, 200))
            pw = float(np.clip(nw * 3.0, 0.5, 3.0))
            color = QColor(100, 200, 255, alpha) if wv >= 0 else QColor(255, 100, 100, alpha)
            pen = QPen(color, pw)
            ud = q.nodes[u].get('position', (0, 0))
            vd = q.nodes[v].get('position', (0, 0))
            self.viz_scene.addLine(ud[0], ud[1], vd[0], vd[1], pen)
            
            if q.number_of_edges() <= 500:
                self._add_arrowhead(ud[0], ud[1], vd[0], vd[1], color)

        lnodes: Dict[int, List[str]] = defaultdict(list)
        for n, d in q.nodes(data=True):
            lnodes[d.get('layer', 0)].append(n)
        for l, nodes in lnodes.items():
            color = LAYER_COLORS[l % len(LAYER_COLORS)]
            show_labels = len(nodes) <= 8
            for n in nodes:
                d = q.nodes[n]
                x, y = d.get('position', (0, 0))
                self.viz_scene.addItem(NeuronItem(n, x, y, radius=NEURON_RADIUS, color=color, show_label=show_labels))

            if nodes:
                min_y = min(q.nodes[n].get('position', (0, 1000))[1] for n in nodes)
                lx = q.nodes[nodes[0]].get('position', (0, 0))[0]
                txt = self.viz_scene.addText(f"Layer {l}", QFont("Arial", 10, QFont.Bold))
                txt.setDefaultTextColor(QColor("white"))
                txt.setPos(lx - txt.boundingRect().width() / 2, min_y - 40)

    def _add_arrowhead(
        self, x1: float, y1: float, x2: float, y2: float, color: QColor
    ) -> None:
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-3:
            return
        ux, uy = dx / length, dy / length
        size = 8
        px, py = -uy, ux
        tip = QPointF(x2, y2)
        base = QPointF(x2 - ux * size, y2 - uy * size)
        left = QPointF(base.x() + px * size * 0.5, base.y() + py * size * 0.5)
        right = QPointF(base.x() - px * size * 0.5, base.y() - py * size * 0.5)
        item = self.viz_scene.addPolygon(
            QPolygonF([tip, left, right]), QPen(color), QBrush(color)
        )
        item.moveBy(-ux * NEURON_RADIUS * 0.5, -uy * NEURON_RADIUS * 0.5)

    def redraw_current(self) -> None:
        c = self.network_list.currentItem()
        if c:
            nid = c.data(Qt.UserRole) or self._extract_id(c.text())
            if nid in self.networks:
                self.draw_network(self.networks[nid])

    def create_default_networks(self) -> None:
        defs = [
            ([4, 8, 4, 2], "MLP", "permutation"),
            ([8, 16, 8], "CNN", "translation"),
            ([6, 6, 6, 6], "RNN", "translation"),
            ([8, 8, 8], "Transformer", "permutation"),
        ]
        for i, (s, nt, gt) in enumerate(defs):
            try:
                nid = f"default_{i+1}"
                net = TheoreticalNeuralNetwork(s, nt, gt, nid)
                res = self.benchmark_suite.run_all_benchmarks(net)
                self.networks[nid] = net
                self.db.store_network(net, res)
            except Exception as e:
                logger.error(f"Default network {i+1} failed: {e}")
        self.update_network_list()
        self.update_stats()

    def update_network_list(self) -> None:
        self.network_list.clear()
        for nid, net in self.networks.items():
            item = QListWidgetItem(f"[{net.network_type}] {nid}")
            item.setData(Qt.UserRole, nid)
            self.network_list.addItem(item)

    def update_stats(self) -> None:
        nets = self.db.get_all_networks()
        self.total_nets_label.setText(str(len(nets)))
        if nets:
            fits = [n.get('overall_fitness', 0) for n in nets]
            self.avg_fitness_label.setText(f"{float(np.mean(fits)):.4f}")
            self.best_fitness_label.setText(f"{float(np.max(fits)):.4f}")
        else:
            self.avg_fitness_label.setText("0.0")
            self.best_fitness_label.setText("0.0")

    def on_network_selected(self) -> None:
        c = self.network_list.currentItem()
        if not c:
            return
        nid = c.data(Qt.UserRole) or self._extract_id(c.text())
        if nid not in self.networks:
            return
        net = self.networks[nid]
        meta = self.db.get_network_metadata(nid)
        res = meta.get('benchmark_results', {}) if meta else {}
        self.show_network_details(nid, net, res)
        self.draw_network(net)

    @staticmethod
    def _extract_id(text: str) -> str:
        p = text.split("] ", 1)
        return p[1] if len(p) > 1 else text

    def store_current_network(self) -> None:
        c = self.network_list.currentItem()
        if not c:
            self.status_bar.showMessage("Select a network first.", 3000)
            return
        nid = c.data(Qt.UserRole) or self._extract_id(c.text())
        if nid in self.networks:
            try:
                res = self.benchmark_suite.run_all_benchmarks(self.networks[nid])
                self.db.store_network(self.networks[nid], res)
                self.update_stats()
                self.status_bar.showMessage(f"Network '{nid}' stored/updated.", 5000)
            except Exception as e:
                self.status_bar.showMessage(f"Store failed: {e}", 5000)
        else:
            self.status_bar.showMessage(f"Network '{nid}' not in memory.", 3000)

    def find_similar_networks(self) -> None:
        c = self.network_list.currentItem()
        if not c:
            self.status_bar.showMessage("Select a network first.", 3000)
            return
        nid = c.data(Qt.UserRole) or self._extract_id(c.text())
        sims = self.db.query_similar_networks(nid, 5)
        if sims:
            msg = "Similar Networks:\n\n"
            for s in sims:
                msg += (f"  {s['network_id']} [{s['network_type']}]\n"
                        f"    distance: {s['distance']:.4f}, "
                        f"fitness: {s['overall_fitness']:.4f}, "
                        f"layers: {s['layer_sizes']}\n")
            QMessageBox.information(self, "Similar Networks", msg)
        else:
            self.status_bar.showMessage("No similar networks found.", 3000)

    def clear_database(self) -> None:
        if QMessageBox.question(
            self, "Confirm", "Clear entire database?",
            QMessageBox.Yes | QMessageBox.No
        ) == QMessageBox.Yes:
            self.db.clear_database()
            self.networks.clear()
            self.update_network_list()
            self.update_stats()
            self.details_text.clear()
            self.theory_table.setRowCount(0)
            self.viz_scene.clear()
            self.status_bar.showMessage("Database cleared.", 5000)

    def export_results(self) -> None:
        if not self.search_results:
            self.status_bar.showMessage("No search results to export.", 3000)
            return
        fp, _ = QFileDialog.getSaveFileName(
            self, "Save Results", "", "JSON Files (*.json)"
        )
        if fp:
            try:
                with open(fp, 'w') as f:
                    json.dump([c.to_dict() for c in self.search_results], f, indent=4, default=str)
                self.status_bar.showMessage("Results exported successfully.", 5000)
            except (OSError, IOError) as e:
                self.status_bar.showMessage(f"Export failed: {e}", 5000)

    def import_results(self) -> None:
        fp, _ = QFileDialog.getOpenFileName(
            self, "Load Results", "", "JSON Files (*.json)"
        )
        if fp:
            try:
                with open(fp, 'r') as f:
                    data = json.load(f)
                self.search_results = [ArchitectureCandidate(**d) for d in data]
                self._populate_results_tables(self.search_results)
                self.status_bar.showMessage("Results imported successfully.", 5000)
            except (OSError, IOError, json.JSONDecodeError, TypeError) as e:
                self.status_bar.showMessage(f"Import failed: {e}", 5000)

    def closeEvent(self, event) -> None:
        """Clean up worker thread on close."""
        if self.nas_worker and self.nas_worker.isRunning():
            self.nas_worker.cancel()
            if not self.nas_worker.wait(3000):
                self.nas_worker.terminate()
                self.nas_worker.wait(1000)
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NASApplication()
    window.show()
    sys.exit(app.exec())