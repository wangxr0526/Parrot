'''

Adapted from rxnmapper https://github.com/rxn4chemistry/rxnmapper
TODO:
1. 使用Encoder Self Attention以及rxnmapper的映射逻辑映射反应物和产物的原子；
2. 使用Cross Attention预测反应物的反应中心，并且用映射过的反应检测反应物和产物有没有变化，作为过滤的标准

'''
from argparse import ArgumentParser
from functools import partial
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analysis_script.evaluate_reaction2condition_attention import ParrotConditionPredictionModelAnalysis

import torch

from typing import Any, Dict, List, Optional
from pandarallel import pandarallel
import yaml
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from rdkit import Chem
from models.model_layer import BOS
from models.utils import caonicalize_rxn_smiles, identify_attention_token_idx_for_rxn_component, inference_load

from models.parrot_model import ParrotConditionPredictionModel

BAD_TOKS = ["[CLS]", "[SEP]"]  # Default Bad Tokens


def canonicalize_smi(smi: str, remove_atom_mapping=False) -> str:
    """ Convert a SMILES string into its canonicalized form

    Args:
        smi: Reaction SMILES
        remove_atom_mapping: If True, remove atom mapping information from the canonicalized SMILES output

    Returns:
        SMILES reaction, canonicalized, as a string
    """
    mol = Chem.MolFromSmiles(smi)
    if not mol:
        raise NotCanonicalizableSmilesException("Molecule not canonicalizable")
    if remove_atom_mapping:
        for atom in mol.GetAtoms():
            if atom.HasProp("molAtomMapNumber"):
                atom.ClearProp("molAtomMapNumber")
    return Chem.MolToSmiles(mol)


def process_reaction(rxn: str,
                     fragments: str = "",
                     fragment_bond: str = "~") -> str:
    """
    Remove atom-mapping, move reagents to reactants and canonicalize reaction.
    If fragment group information is given, keep the groups together using
    the character defined with fragment_bond.

    Args:
        rxn: Reaction SMILES
        fragments: (optional) fragments information
        fragment_bond:

    Returns: joined_precursors>>joined_products reaction SMILES
    """
    reactants, reagents, products = rxn.split(">")
    try:
        precursors = [canonicalize_smi(r, True) for r in reactants.split(".")]
        if len(reagents) > 0:
            precursors += [
                canonicalize_smi(r, True) for r in reagents.split(".")
            ]
        products = [canonicalize_smi(p, True) for p in products.split(".")]
    except NotCanonicalizableSmilesException:
        return ""
    if len(fragments) > 1 and fragments[1] == "f":
        number_of_precursors = len(precursors)
        groups = fragments[3:-1].split(",")
        new_precursors = precursors.copy()
        new_products = products.copy()
        for group in groups:
            grouped_smi = []
            if group.startswith("^"):
                return ""
            for member in group.split("."):
                member = int(member)
                if member >= number_of_precursors:
                    grouped_smi.append(products[member - number_of_precursors])
                    new_products.remove(products[member -
                                                 number_of_precursors])
                else:
                    grouped_smi.append(precursors[member])
                    new_precursors.remove(precursors[member])
            if member >= number_of_precursors:
                new_products.append(fragment_bond.join(sorted(grouped_smi)))
            else:
                new_precursors.append(fragment_bond.join(sorted(grouped_smi)))
        precursors = new_precursors
        products = new_products
    joined_precursors = ".".join(sorted(precursors))
    joined_products = ".".join(sorted(products))
    return f"{joined_precursors}>>{joined_products}"


def generate_atom_mapped_reaction_atoms(rxn: str,
                                        product_atom_maps,
                                        expected_atom_maps=None):
    """
    Generate atom-mapped reaction from unmapped reaction and
    product-2-reactant atoms mapping vector.
    Args:
        rxn: unmapped reaction
        product_atom_maps: product to reactant atom maps
        expected_atom_maps: (optional) if given return the differences

    Returns: Atom-mapped reaction

    """

    precursors, products = rxn.split(">>")
    precursors_mol = Chem.MolFromSmiles(precursors)
    products_mol = Chem.MolFromSmiles(products)

    precursors_atom_maps = []

    differing_maps = []

    product_mapping_dict = {}

    for i, atom in enumerate(precursors_mol.GetAtoms()):
        if i in product_atom_maps:
            # atom maps start at an index of 1
            corresponding_product_atom_map = product_atom_maps.index(i) + 1
            precursors_atom_maps.append(corresponding_product_atom_map)
            atom.SetProp("molAtomMapNumber",
                         str(corresponding_product_atom_map))

            indices = [
                idx for idx, x in enumerate(product_atom_maps) if x == i
            ]

            if len(indices) > 1:
                for idx in indices[1:]:
                    product_mapping_dict[idx] = corresponding_product_atom_map

            if expected_atom_maps is not None:
                if (i not in expected_atom_maps
                        or corresponding_product_atom_map !=
                        expected_atom_maps.index(i) + 1):
                    differing_maps.append(corresponding_product_atom_map)

    atom_mapped_precursors = Chem.MolToSmiles(precursors_mol)

    for i, atom in enumerate(products_mol.GetAtoms()):
        atom_map = product_mapping_dict.get(i, i + 1)
        atom.SetProp("molAtomMapNumber", str(atom_map))
    atom_mapped_products = Chem.MolToSmiles(products_mol)

    atom_mapped_rxn = atom_mapped_precursors + ">>" + atom_mapped_products

    if expected_atom_maps is not None:
        return atom_mapped_rxn, differing_maps

    return atom_mapped_rxn


def number_tokens(tokens: List[str],
                  special_tokens: List[str] = BAD_TOKS) -> List[int]:
    """Map list of tokens to a list of numbered atoms

    Args:
        tokens: Tokenized SMILES
        special_tokens: List of tokens to not consider as atoms

    Example:
        >>> number_tokens(['[CLS]', 'C', '.', 'C', 'C', 'C', 'C', 'C', 'C','[SEP]'])
                #=> [-1, 0, -1, 1, 2, 3, 4, 5, 6, -1]
    """
    atom_num = 0
    isatm = partial(is_atom, special_tokens=special_tokens)

    def check_atom(t):
        if isatm(t):
            nonlocal atom_num
            ind = atom_num
            atom_num = atom_num + 1
            return ind
        return -1

    out = [check_atom(t) for t in tokens]

    return out


def is_atom(token: str, special_tokens: List[str] = BAD_TOKS) -> bool:
    """Determine whether a token is an atom.

    Args:
        token: Token fed into the transformer model
        special_tokens: List of tokens to consider as non-atoms (often introduced by tokenizer)

    Returns:
        bool: True if atom, False if not
    """
    bad_toks = set(special_tokens)
    normal_atom = token[0].isalpha() or token[0] == "["
    is_bad = token in bad_toks
    return (not is_bad) and normal_atom


def get_mask_for_tokens(tokens: List[str],
                        special_tokens: List[str] = []) -> List[int]:
    """Return a mask for a tokenized smiles, where atom tokens
    are converted to 1 and other tokens to 0.

    e.g. c1ccncc1 would give [1, 0, 1, 1, 1, 1, 1, 0]

    Args:
        smiles: Smiles string of reaction
        special_tokens: Any special tokens to explicitly not call an atom. E.g. "[CLS]" or "[SEP]"

    Returns:
        Binary mask as a list where non-zero elements represent atoms
    """
    check_atom = partial(is_atom, special_tokens=special_tokens)

    atom_token_mask = [1 if check_atom(t) else 0 for t in tokens]
    return atom_token_mask


def get_atom_types_smiles(smiles: str) -> List[int]:
    """Convert each atom in a SMILES into a list of their atomic numbers

    Args:
        smiles: SMILES representation of molecule

    Returns:
        List of atom numbers for each atom in the smiles. Reports atoms in the same order they were passed in the original SMILES
    """
    smiles_mol = Chem.MolFromSmiles(smiles)

    atom_types = [atom.GetAtomicNum() for atom in smiles_mol.GetAtoms()]

    return atom_types


def get_adjacency_matrix(smiles: str):
    """
    Compute adjacency matrix between atoms. Only works for single molecules atm and not for rxns

    Args:
        smiles: SMILES representation of a molecule

    Returns:
        Numpy array representing the adjacency between each atom and every other atom in the molecular SMILES.
        Equivalent to `distance_matrix[distance_matrix == 1]`
    """

    mol = Chem.MolFromSmiles(smiles)
    return Chem.GetAdjacencyMatrix(mol)


def group_with(predicate, xs: List[Any]):
    """Takes a list and returns a list of lists where each sublist's elements are
all satisfied pairwise comparison according to the provided function.
Only adjacent elements are passed to the comparison function

    Original implementation here: https://github.com/slavaGanzin/ramda.py/blob/master/ramda/group_with.py

    Args:
        predicate ( f(a,b) => bool): A function that takes two subsequent inputs and returns True or Fale
        xs: List to group
    """
    out = []
    is_str = isinstance(xs, str)
    group = [xs[0]]

    for x in xs[1:]:
        if predicate(group[-1], x):
            group += [x]
        else:
            out.append("".join(group) if is_str else group)
            group = [x]

    out.append("".join(group) if is_str else group)

    return out


def is_mol_end(a: str, b: str) -> bool:
    """Determine if `a` and `b` are both tokens within a molecule (Used by the `group_with` function).

    Returns False whenever either `a` or `b` is a molecule delimeter (`.` or `>>`)"""
    no_dot = (a != ".") and (b != ".")
    no_arrow = (a != ">>") and (b != ">>")

    return no_dot and no_arrow


def split_into_mols(tokens: List[str]) -> List[List[str]]:
    """Split a reaction SMILES into SMILES for each molecule"""
    split_toks = group_with(is_mol_end, tokens)
    return split_toks


def tokens_to_smiles(tokens: List[str], special_tokens: List[str]) -> str:
    """Combine tokens into valid SMILES string, filtering out special tokens

    Args:
        tokens: Tokenized SMILES
        special_tokens: Tokens to not count as atoms

    Returns:
        SMILES representation of provided tokens, without the special tokens
    """
    bad_toks = set(special_tokens)
    return "".join([t for t in tokens if t not in bad_toks])


def tokens_to_adjacency(tokens: List[str]) -> np.array:
    """Convert a tokenized reaction SMILES into a giant adjacency matrix.

    Note that this is a large, sparse Block Diagonal matrix of the adjacency matrix for each molecule in the reaction.

    Args:
        tokens: Tokenized SMILES representation

    Returns:
        Numpy Array, where non-zero entries in row `i` indicate the tokens that are atom-adjacent to token `i`
    """
    from scipy.linalg import block_diag

    mol_tokens = split_into_mols(tokens)
    altered_mol_tokens = [
        m for m in mol_tokens if "." not in set(m) and ">>" not in set(m)
    ]

    smiles = [tokens_to_smiles(mol, BAD_TOKS) for mol in mol_tokens
              ]  # Cannot process SMILES if it is a '.' or '>>'

    # Calculate adjacency matrix for reaction
    altered_smiles = [s for s in smiles
                      if s not in set([".", ">>"])]  # Only care about atoms
    adjacency_mats = [get_adjacency_matrix(s) for s in altered_smiles
                      ]  # Or filter if we don't need to save the spot
    rxn_mask = block_diag(*adjacency_mats)
    return rxn_mask


class AttentionScorer:

    def __init__(
        self,
        rxn_smiles: str,
        tokens: List[str],
        attentions: np.ndarray,
        special_tokens: List[str] = ["[CLS]", "[SEP]"],
        attention_multiplier: float = 90.0,
        mask_mapped_product_atoms: bool = True,
        mask_mapped_reactant_atoms: bool = True,
        output_attentions: bool = False,
    ):
        """Convenience wrapper for mapping attentions into the atom domain, separated by reactants and products, and introducing neighborhood locality.
        
        Args:
            rxn_smiles: Smiles for reaction
            tokens: Tokenized smiles of reaction of length N
            attentions: NxN attention matrix
            special_tokens: Special tokens used by the model that do not count as atoms
            attention_multiplier: Amount to increase the attention connection from adjacent atoms of a newly mapped product atom to adjacent atoms of the newly mapped reactant atom. 
                Boosts the likelihood of an atom having the same adjacent atoms in reactants and products
            mask_mapped_product_atoms: If true, zero attentions to product atoms that have already been mapped
            mask_mapped_reactant_atoms: If true, zero attentions to reactant atoms that have already been mapped
            output_attentions: If true, output the raw attentions along with generated atom maps
        """
        self.rxn, self.tokens, self.attentions = rxn_smiles, tokens, attentions
        self.special_tokens = special_tokens
        self.N = len(tokens)
        self.attention_multiplier = attention_multiplier
        self.mask_mapped_product_atoms = mask_mapped_product_atoms
        self.mask_mapped_reactant_atoms = mask_mapped_reactant_atoms
        self.output_attentions = output_attentions

        try:
            self.split_ind = tokens.index(
                ">>")  # Index that separates products from reactants
            self._product_inds = slice(self.split_ind + 1, self.N)
            self._reactant_inds = slice(0, self.split_ind)
        except ValueError:
            raise ValueError(
                "rxn smiles is not a complete reaction. Can't find the '>>' to separate the products"
            )

        # Mask of atoms
        self.atom_token_mask = np.array(
            get_mask_for_tokens(self.tokens,
                                self.special_tokens)).astype(np.bool)

        # Atoms numbered in the array
        self.token2atom = np.array(number_tokens(tokens))
        self.atom2token = {
            k: v
            for k, v in zip(self.token2atom, range(len(self.token2atom)))
        }

        # Adjacency graph for all tokens
        self.adjacency_matrix = tokens_to_adjacency(tokens).astype(np.bool)

        self._precursors_atom_types = None
        self._product_atom_types = None
        self._rnums_atoms = None
        self._pnums_atoms = None
        self._nreactant_atoms = None
        self._nproduct_atoms = None
        self._adjacency_matrix_products = None
        self._adjacency_matrix_precursors = None
        self._pxr_filt_atoms = None
        self._rxp_filt_atoms = None
        self._atom_type_mask = None
        self._atom_type_masked_attentions = None

        # Attention multiplication matrix
        self.attention_multiplier_matrix = np.ones_like(
            self.combined_attentions_filt_atoms).astype(float)

    @property
    def atom_attentions(self):
        """The MxM attention matrix, selected for only attentions that are from atoms, to atoms"""
        return self.attentions[self.atom_token_mask].T[self.atom_token_mask].T

    @property
    def adjacent_atom_attentions(self):
        """The MxM attention matrix, where all attentions are zeroed if the attention is not to an adjacent atom."""
        atts = self.atom_attentions.copy()
        mask = np.logical_not(self.adjacency_matrix)
        atts[mask] = 0
        return atts

    @property
    def adjacency_matrix_reactants(self):
        """The adjacency matrix of the reactants"""
        if self._adjacency_matrix_precursors is None:
            self._adjacency_matrix_precursors = self.adjacency_matrix[:len(
                self.rnums_atoms), :len(self.rnums_atoms)]
        return self._adjacency_matrix_precursors

    @property
    def adjacency_matrix_products(self):
        """The adjacency matrix of the products"""
        if self._adjacency_matrix_products is None:
            self._adjacency_matrix_products = self.adjacency_matrix[
                len(self.rnums_atoms):,
                len(self.rnums_atoms):]
        return self._adjacency_matrix_products

    @property
    def atom_type_masked_attentions(self):
        """Generate a """
        if self._atom_type_masked_attentions is None:
            self._atom_type_masked_attentions = np.multiply(
                self.combined_attentions_filt_atoms, self.get_atom_type_mask())
        return self._atom_type_masked_attentions

    @property
    def rxp(self):
        """Subset of attentions relating the reactants to the products"""
        return self.attentions[:self.split_ind, (self.split_ind + 1):]

    @property
    def rxp_filt(self):
        """RXP without the special tokens"""
        return self.rxp[1:, :-1]

    @property
    def rxp_filt_atoms(self):
        """RXP only the atoms, no special tokens"""
        if self._rxp_filt_atoms is None:
            self._rxp_filt_atoms = self.rxp[[i != -1 for i in self.rnums
                                             ]][:,
                                                [i != -1 for i in self.pnums]]
        return self._rxp_filt_atoms

    @property
    def pxr(self):
        """Subset of attentions relating the products to the reactants"""
        i = self.split_ind
        return self.attentions[(i + 1):, :i]

    @property
    def pxr_filt(self):
        """PXR without the special tokens"""
        return self.pxr[:-1, 1:]

    @property
    def pxr_filt_atoms(self):
        """PXR only the atoms, no special tokens"""
        if self._pxr_filt_atoms is None:
            self._pxr_filt_atoms = self.pxr[[i != -1 for i in self.pnums
                                             ]][:,
                                                [i != -1 for i in self.rnums]]
        return self._pxr_filt_atoms

    @property
    def combined_attentions(self):
        """Summed pxr and rxp"""
        return self.pxr + self.rxp.T

    @property
    def combined_attentions_filt(self):
        """Summed pxr_filt and rxp_filt (no special tokens)"""
        return self.pxr_filt + self.rxp_filt.T

    @property
    def combined_attentions_filt_atoms(self):
        """Summed pxr_filt_atoms and rxp_filt_atoms (no special tokens, no "non-atom" tokens)"""
        return self.pxr_filt_atoms + self.rxp_filt_atoms.T

    @property
    def combined_attentions_filt_atoms_same_type(self):
        """Summed pxr_filt_atoms and rxp_filt_atoms (no special tokens, no "non-atom" tokens). All attentions to atoms of a different type are zeroed"""

        atom_type_mask = np.zeros(self.combined_attentions_filt_atoms.shape)
        precursor_atom_types = get_atom_types_smiles("".join(self.rtokens[1:]))
        for i, atom_type in enumerate(
                get_atom_types_smiles("".join(self.ptokens[:-1]))):
            if atom_type > 0:
                atom_type_mask[i, :] = (
                    np.array(precursor_atom_types) == atom_type).astype(int)
        combined_attentions = np.multiply(self.combined_attentions_filt_atoms,
                                          atom_type_mask)
        row_sums = combined_attentions.sum(axis=1)
        normalized_attentions = np.divide(
            combined_attentions,
            row_sums[:, np.newaxis],
            out=np.zeros_like(combined_attentions),
            where=row_sums[:, np.newaxis] != 0,
        )
        return normalized_attentions

    @property
    def pnums(self):
        """Get atom indexes for just the product tokens. 
        
        Numbers in this vector that are >= 0 are atoms, whereas indexes == -1 represent special tokens (e.g., bonds, parens, [CLS])
        """
        return self.token2atom[(self.split_ind + 1):]

    @property
    def pnums_filt(self):
        """Get atom indexes for just the product tokens, without the [SEP]. 
        
        Numbers in this vector that are >= 0 are atoms, whereas indexes == -1 represent special tokens (e.g., bonds, parens, [CLS])
        """
        return self.pnums

    @property
    def pnums_atoms(self):
        """Get atom indexes for just the product ATOMS, without the [SEP]. 
        
        Numbers in this vector that are >= 0 are atoms, whereas indexes == -1 represent special tokens (e.g., bonds, parens, [CLS])
        """
        if self._pnums_atoms is None:
            self._pnums_atoms = np.array([a for a in self.pnums if a != -1])
        return self._pnums_atoms

    @property
    def rnums(self):
        """Get atom indexes for the reactant tokens. 
        
        Numbers in this vector that are >= 0 are atoms, whereas indexes == -1 represent special tokens (e.g., bonds, parens, [CLS])
        """
        return self.token2atom[:self.split_ind]

    @property
    def rnums_filt(self):
        """Get atom indexes for just the reactant tokens, without the [CLS]. 
        
        Numbers in this vector that are >= 0 are atoms, whereas indexes == -1 represent special tokens (e.g., bonds, parens, [CLS])
        """
        return self.rnums[1:]

    @property
    def rnums_atoms(self):
        """Get atom indexes for the reactant ATOMS, without the [CLS]. 
        
        Numbers in this vector that are >= 0 are atoms, whereas indexes == -1 represent special tokens (e.g., bonds, parens, [CLS])
        """
        if self._rnums_atoms is None:
            self._rnums_atoms = np.array([a for a in self.rnums if a != -1])
        return self._rnums_atoms

    @property
    def nreactant_atoms(self):
        """The number of atoms in the reactants"""
        if self._nreactant_atoms is None:
            self._nreactant_atoms = len(self.rnums_atoms)

        return self._nreactant_atoms

    @property
    def nproduct_atoms(self):
        """The number of atoms in the product"""
        if self._nproduct_atoms is None:
            self._nproduct_atoms = len(self.pnums_atoms)

        return self._nproduct_atoms

    @property
    def rtokens(self):
        """Just the reactant tokens"""
        return self.tokens[self._reactant_inds]

    @property
    def rtokens_filt(self):
        """Reactant tokens without special tokens"""
        return self.rtokens[1:]

    @property
    def ptokens(self):
        """Just the product tokens"""
        return self.tokens[self._product_inds]

    @property
    def ptokens_filt(self):
        """Product tokens without special tokens"""
        return self.ptokens[:-1]

    def token_ind(self, atom_num) -> int:
        """Get token index from an atom number
        
        Note that this is not a lossless mapping. -1 represents any special token, but it is always mapped to the token at index [N - 1]
        """
        return self.atom2token[atom_num]

    def atom_num(self, token_ind) -> int:
        """Get the atom number corresponding to a token"""
        return self.token2atom[token_ind]

    def is_atom(self, token_ind) -> int:
        """Check whether a token is an atom"""
        return self.atom_token_mask[token_ind]

    def get_neighboring_attentions(self, atom_num) -> np.ndarray:
        """Get a vector of shape (n_atoms,) representing the neighboring attentions to an atom number. 

        Non-zero attentions are the attentions for neighboring atoms
        """
        return self.atom_attentions[atom_num] * self.adjacency_matrix[atom_num]

    def get_neighboring_atoms(self, atom_num):
        """Get the atom indexes neighboring the desired atom"""
        return np.nonzero(self.adjacency_matrix[atom_num])[0]

    def get_precursors_atom_types(self):
        """Convert reactants into their atomic numbers"""
        if self._precursors_atom_types is None:
            self._precursors_atom_types = get_atom_types_smiles("".join(
                self.rtokens[1:]))
        return self._precursors_atom_types

    def get_product_atom_types(self):
        """Convert products into their atomic indexes"""
        if self._product_atom_types is None:
            self._product_atom_types = get_atom_types_smiles("".join(
                self.ptokens[:-1]))
        return self._product_atom_types

    def get_atom_type_mask(self):
        """Return a mask where only atoms of the same type are True"""
        if self._atom_type_mask is None:
            atom_type_mask = np.zeros(
                self.combined_attentions_filt_atoms.shape)
            precursor_atom_types = self.get_precursors_atom_types()
            for i, atom_type in enumerate(self.get_product_atom_types()):
                atom_type_mask[i, :] = (
                    np.array(precursor_atom_types) == atom_type).astype(int)
            self._atom_type_mask = atom_type_mask
        return self._atom_type_mask

    def _get_combined_normalized_attentions(self):
        """ Get normalized attention matrix from product atoms to candidate reactant atoms. """

        combined_attentions = np.multiply(self.atom_type_masked_attentions,
                                          self.attention_multiplier_matrix)

        row_sums = combined_attentions.sum(axis=1)
        normalized_attentions = np.divide(
            combined_attentions,
            row_sums[:, np.newaxis],
            out=np.zeros_like(combined_attentions),
            where=row_sums[:, np.newaxis] != 0,
        )
        return normalized_attentions

    def generate_attention_guided_pxr_atom_mapping(
            self, absolute_product_inds: bool = False):
        """
        Generate attention guided product to reactant atom mapping.
        Args:
            absolute_product_inds: If True, adjust all indexes related to the product to be relative to that atom's position
                in the entire reaction SMILES
        """

        pxr_mapping_vector = (np.ones(len(self.pnums_atoms)) * -1).astype(int)

        output = {}

        confidences = np.ones(len(self.pnums_atoms))

        mapping_tuples = []

        for i in range(len(self.pnums_atoms)):
            attention_matrix = self._get_combined_normalized_attentions()

            if i == 0 and self.output_attentions:
                output["pxrrxp_attns"] = attention_matrix

            product_atom_to_map = np.argmax(np.max(attention_matrix, axis=1))
            corresponding_reactant_atom = np.argmax(
                attention_matrix, axis=1)[product_atom_to_map]
            confidence = np.max(attention_matrix)

            if np.isclose(confidence, 0.0):
                confidence = 1.0
                corresponding_reactant_atom = pxr_mapping_vector[
                    product_atom_to_map]  # either -1 or already mapped
                break

            pxr_mapping_vector[
                product_atom_to_map] = corresponding_reactant_atom
            confidences[product_atom_to_map] = confidence
            self._update_attention_multiplier_matrix(
                product_atom_to_map, corresponding_reactant_atom)

            if absolute_product_inds:
                adjusted_product_atom = product_atom_to_map + self.nreactant_atoms
            else:
                adjusted_product_atom = product_atom_to_map
            mapping_tuples.append((adjusted_product_atom,
                                   corresponding_reactant_atom, confidence))

        output["pxr_mapping_vector"] = pxr_mapping_vector.tolist()
        output["confidences"] = confidences
        output["mapping_tuples"] = mapping_tuples
        return output

    def _update_attention_multiplier_matrix(self, product_atom: int,
                                            reactant_atom: int):
        """Perform the "neighbor multiplier" step of the atom mapping
        
        Increase the attention connection between the neighbors of specified product atom
        to the neighbors of the specified reactant atom. A stateful operation.

        Args:
            product_atom: Atom index of the product atom (relative to the beginning of the products)
            reactant_atom: Atom index of the reactant atom (relative to the beginning of the reactants)
        """
        if not reactant_atom == -1:
            neighbors_in_products = self.adjacency_matrix_products[
                product_atom]
            neighbors_in_reactants = self.adjacency_matrix_reactants[
                reactant_atom]

            self.attention_multiplier_matrix[np.ix_(
                neighbors_in_products,
                neighbors_in_reactants)] *= float(self.attention_multiplier)

        if self.mask_mapped_product_atoms:
            self.attention_multiplier_matrix[product_atom] = np.zeros(
                len(self.rnums_atoms))
        if self.mask_mapped_reactant_atoms:
            self.attention_multiplier_matrix[:, reactant_atom] = np.zeros(
                len(self.pnums_atoms))

    def __len__(self):
        """Length of provided tokens"""
        return len(self.tokens)

    def __repr__(self):
        return f"AttMapper(`{self.rxn[:50]}...`)"


class Mapper:

    def __init__(self,
                 config_path,
                 gpu=0,
                 mapper_config={},
                 batch_size=8) -> None:

        self.config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

        print(
            '\n########################\Attention Predictor configs:\n########################\n'
        )
        print(yaml.dump(self.config))
        print('\n########################\n')

        model_args = self.config['model_args']
        model_args['pretrained_path'] = os.path.join(
            '..', model_args['pretrained_path'])
        model_args['output_dir'] = os.path.join('..', model_args['output_dir'])
        model_args['best_model_dir'] = os.path.join(
            '..', model_args['best_model_dir'])
        model_args['output_attention'] = model_args.get(
            'output_attention', True)
        model_args['silent'] = True

        dataset_args = self.config['dataset_args']
        dataset_args['dataset_root'] = os.path.join(
            '..', dataset_args['dataset_root'])
        try:
            model_args['use_temperature'] = dataset_args['use_temperature']
            print('\nUsing Temperature:', model_args['use_temperature'])
        except:
            print('\nNo temperature information is specified!')

        # self.input_df = input_df
        self.model_args = model_args
        self.dataset_args = dataset_args
        self.batch_size = batch_size
        self.mapper_config = mapper_config
        self.gpu = gpu
        self.model = self.init_model()

        self.attention_multiplier = mapper_config.get("attention_multiplier",
                                                      90.0)
        self.head = mapper_config.get("head", 3)
        self.layers = mapper_config.get("layers", [10])

    def init_dataset_to_predict(self,
                                input_dataset_path,
                                dataset_flag='val',
                                debug=False):

        input_df = pd.read_csv(input_dataset_path)

        condition_type2cols = {
            'c1': 'catalyst1',
            's1': 'solvent1',
            's2': 'solvent2',
            'r1': 'reagent1',
            'r2': 'reagent2'
        }
        condition_cols = [
            'catalyst1', 'solvent1', 'solvent2', 'reagent1', 'reagent2'
        ]
        for col in condition_cols:
            input_df.loc[input_df[col].isna(), col] = ''

        input_df.loc[input_df['retro_template'].isna(), 'retro_template'] = ''

        if 'dataset' in input_df.columns.tolist():
            input_df = input_df.loc[input_df['dataset'] == dataset_flag]
            # input_df = input_df.loc[input_df[condition_type2cols[
            #     parser_args.eval_condition_type]] != ''].reset_index(drop=True)
            input_df = input_df.loc[
                input_df['retro_template'] != ''].reset_index(drop=True)
        else:
            # input_df = input_df.loc[input_df[condition_type2cols[
            #     parser_args.eval_condition_type]] != ''].reset_index(drop=True)
            input_df = input_df.loc[
                input_df['retro_template'] != ''].reset_index(drop=True)
        if debug:
            input_df = input_df[:1000]

        return input_df

    def init_model(self):
        condition_label_mapping = inference_load(**self.dataset_args)
        self.model_args['decoder_args'].update({
            'tgt_vocab_size':
            len(condition_label_mapping[0]),
            'condition_label_mapping':
            condition_label_mapping
        })

        trained_path = self.model_args['best_model_dir']

        model: ParrotConditionPredictionModelAnalysis = ParrotConditionPredictionModelAnalysis(
            "bert",
            trained_path,
            args=self.model_args,
            use_cuda=True if self.gpu >= 0 else False,
            cuda_device=self.gpu)
        return model

    def convert_batch_to_attns(
        self,
        input_dataset_path='',
        dataset_flag='val',
        debug=False,
        rxn_smiles_list: List[str] = None,
        force_layer: Optional[int] = None,
        force_head: Optional[int] = None,
    ):
        if force_layer is None: use_layers = self.layers
        else: use_layers = [force_layer]

        if force_head is None: use_head = self.head
        else: use_head = force_head
        if rxn_smiles_list is None:
            input_df = self.init_dataset_to_predict(input_dataset_path,
                                                    dataset_flag=dataset_flag,
                                                    debug=debug)

            rxn_smiles_list = input_df['canonical_rxn'].tolist()
        input_df_with_fake_labels = pd.DataFrame({
            'text':
            rxn_smiles_list,
            'labels': [[0] * 7] * len(rxn_smiles_list)
        })

        predicted_conditions, attention_weights, input_tokens = self.model.greedy_search_batch_with_attn(
            input_df_with_fake_labels,
            test_batch_size=self.batch_size,
            normalize=True,
            transpose_end=False)

        encoder_self_attn = attention_weights['encoder_self_attn']
        selected_attns = []
        for attn in encoder_self_attn:
            selected_attn = attn[use_layers, :, :, :]
            selected_attn = selected_attn[:, use_head, :, :]
            selected_attn = torch.mean(selected_attn, dim=[0])
            selected_attns.append(selected_attn)
        return selected_attns, input_tokens, predicted_conditions

    def get_attention_guided_atom_maps(
        self,
        input_dataset_path='',
        rxn_smiles_list=None,
        zero_set_p: bool = True,
        zero_set_r: bool = True,
        canonicalize_rxns: bool = True,
        detailed_output: bool = False,
        absolute_product_inds: bool = False,
        force_layer: Optional[int] = None,
        force_head: Optional[int] = None,
        debug=False,
    ):

        results = []
        if canonicalize_rxns:
            rxns = [process_reaction(rxn) for rxn in rxn_smiles_list]

        attns, input_tokens, _ = self.convert_batch_to_attns(
            rxn_smiles_list=rxn_smiles_list,
            input_dataset_path=input_dataset_path,
            force_layer=force_layer,
            force_head=force_head,
            debug=debug)

        for attn, tokens in zip(attns, input_tokens):
            just_tokens = ['[CLS]'] + tokens + ['[SEP]']
            rxn = ''.join(tokens)
            tokensxtokens_attn = attn.detach().cpu().numpy()
            attention_scorer = AttentionScorer(
                rxn,
                just_tokens,
                tokensxtokens_attn,
                attention_multiplier=self.attention_multiplier,
                mask_mapped_product_atoms=zero_set_p,
                mask_mapped_reactant_atoms=zero_set_r,
                output_attentions=
                detailed_output,  # Return attentions when detailed output requested
            )

            output = attention_scorer.generate_attention_guided_pxr_atom_mapping(
                absolute_product_inds=absolute_product_inds)

            result = {
                "mapped_rxn":
                generate_atom_mapped_reaction_atoms(
                    rxn, output["pxr_mapping_vector"]),
                "confidence":
                np.prod(output["confidences"]),
            }
            if detailed_output:
                result["pxr_mapping_vector"] = output["pxr_mapping_vector"]
                result["pxr_confidences"] = output["confidences"]
                result["mapping_tuples"] = output["mapping_tuples"]
                result["pxrrxp_attns"] = output["pxrrxp_attns"]
                result["tokensxtokens_attns"] = tokensxtokens_attn
                result["tokens"] = just_tokens

            results.append(result)
        return results


def extract_mapping_and_conf(result: Dict):
    mapping_tuples: List = result['mapping_tuples']
    confidence = result['confidence']
    mapping_tuples.sort(key=lambda x: x[0])
    map_to_validate = [x[1] for x in mapping_tuples]
    return map_to_validate, confidence


def get_mapping_accuracy(pred: List[int], gts: List[List]):
    pred_map, conf = pred
    for gt in gts:
        pred_map = np.array(pred_map)
        gt = np.array(gt)
        if pred_map.shape == gt.shape:
            if (pred_map == gt).all():
                return True, conf

    return False, None


if __name__ == '__main__':

    debug = True
    rxnmapper_uspto_dataset_path = './eval_data/eval_use_data/uspto_rxnmapper'
    uspto_50k_eval_set = pd.read_json(
        os.path.join(rxnmapper_uspto_dataset_path, 'Validation',
                     'eval_schneider.json'))

    correct_maps = uspto_50k_eval_set['correct_maps'].tolist()
    rxns = uspto_50k_eval_set['rxn'].tolist()
    if debug:
        correct_maps = correct_maps[:10]
        rxns = rxns[:10]

    mapper = Mapper(config_path='../configs/config_inference_use_uspto.yaml')

    all_accuracy_results = []
    best_head_layer = None
    best_accuracy = 0
    for head in range(0, 4):
        for layer in range(0, 12):
            results = [
                mapper.get_attention_guided_atom_maps(
                    rxn_smiles_list=[rxn],
                    debug=debug,
                    force_head=head,
                    force_layer=layer,
                    detailed_output=True,
                )[0] for rxn in tqdm(rxns)
            ]

            results_to_validate = [
                extract_mapping_and_conf(x) for x in results
            ]

            accuracy_results = [
                get_mapping_accuracy(results_to_validate[i], correct_maps[i])
                for i in range(len(results_to_validate))
            ]

            is_correct, confs = map(list, zip(*accuracy_results))
            accuracy = np.array(is_correct).sum() / len(is_correct)
            confs = [x for x in confs if x]
            mean_confs = np.array(confs).mean() if confs else None

            all_accuracy_results.append((layer, head, accuracy, mean_confs))

            if mean_confs:
                print('Accuracy: {:.4f}, Confidence: {:.6f}'.format(
                    accuracy, mean_confs))
            else:
                print('Accuracy: {:.4f}, Confidence: {}'.format(
                    accuracy, mean_confs))

            if best_accuracy < accuracy:
                best_accuracy = accuracy
                best_head_layer = {
                    'head': head,
                    'layer': layer,
                    'accuracy': best_accuracy
                }

            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            print('Current Best head layer', best_head_layer, '')
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print('Best head layer', best_head_layer)
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    all_accuracy_results_df = pd.DataFrame(
        all_accuracy_results,
        columns=['layer', 'head', 'accuracy', 'confidence'])
    all_accuracy_results_df.sort_values(by=['accuracy', 'confidence'],
                                        ascending=False)
    all_accuracy_results_df.to_csv(
        './eval_data/eval_results/mapper_evaluation_results.csv', index=False)
