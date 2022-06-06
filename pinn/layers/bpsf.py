# -*- coding: utf-8 -*-

"""(Keras) Layers for building Behler-Parrinello symmetry functions

This module implements the base layers `*_SF`, which computes the symmetry
functions. The layers handelling the computation of atom-centered fingerprints
and the caching of Jacobian are implemented in `networks.bpnn.BPFingerprint`.
"""

import numpy as np
import tensorflow as tf
from pinn.utils import pi_named
from .basis import GaussianBasis


def sf2fp(i_rind, a_rind, sf):
    """Helper function to concatenat"""
    n_sf = sf.shape[-1]
    fp = tf.scatter_nd(tf.expand_dims(i_rind, 1), sf,
                       [tf.reduce_max(a_rind)+1, n_sf])
    return fp

@pi_named("triplet_jacobian")
def _triplet_jacobian(i_rind, ind_ij, ind_ik):
    """Helper function for jacobian indices in G3 and G4 SFs

    Returns:
       jacob_ind: indices of the (sparse) jacobian matrix
    """
    p_ind, p_uniq_idx = tf.unique(tf.concat([ind_ij, ind_ik], axis=0))
    i_rind = tf.math.unsorted_segment_max(
        tf.concat([i_rind, i_rind], axis=0), p_uniq_idx, tf.shape(p_ind)[0]
    )
    jacob_ind = tf.stack([p_ind, i_rind], axis=1)
    return jacob_ind


@pi_named("triplet_filter")
def _triplet_filter(ind_2, ind_3, elems, i, j, k):
    """Helper function for atom selection in G3 and G4 symmetry functions

    Returns:
       i_rind: relative indices of i atoms within the selected species
       ind_ij: indices of pair ij
       ind_ik: indices of pair ik
    """
    ind_ij = ind_3[:, 0]
    ind_ik = ind_3[:, 1]
    i_rind = tf.gather(ind_2[:, 0], ind_ij)
    t_filter = []  # build the filter if necessary
    if i != "ALL":
        t_filter.append(tf.equal(tf.gather(elems, tf.gather(ind_2[:, 0], ind_ij)), i))
        a_rind = tf.cumsum(tf.cast(tf.equal(elems, i), tf.int32)) - 1
    else:  # a_rind: relative indices of atoms within all "i-species"
        a_rind = tf.cumsum(tf.ones_like(elems, tf.int32)) - 1
    if j != "ALL":
        t_filter.append(tf.equal(tf.gather(elems, tf.gather(ind_2[:, 1], ind_ij)), j))
    if k != "ALL":
        t_filter.append(tf.equal(tf.gather(elems, tf.gather(ind_2[:, 1], ind_ik)), k))
    if t_filter:
        t_filter = tf.reduce_all(t_filter, axis=0)
        t_ind = tf.cast(tf.where(t_filter)[:, 0], tf.int32)
        ind_ij = tf.gather(ind_ij, t_ind)
        ind_ik = tf.gather(ind_ik, t_ind)
        i_rind = tf.gather(a_rind, tf.gather(i_rind, t_ind))
    return i_rind, a_rind, ind_ij, ind_ik


class G2_SF(tf.keras.layers.Layer):
    def __init__(self, Rs, eta, i="ALL", j="ALL"):
        """
        Args:
            Rs (list of floats): Gaussian centers
            eta (list of floats): Gaussian widths
            i (str): species i
            j (str): species j
        """
        super(G2_SF, self).__init__()
        self.basis = GaussianBasis(center=Rs, gamma=eta)
        self.i = i
        self.j = j

    def call(self, ind_2, dist, elems, fc):
        """
        Args:
            ind_2: (N_pair x 2) indices for each pair
            dist: (N_pair) array of distance
            elems: (N_atom) elements for each atom
            fc: (N_pair) cutoff functio  n

        Returns:
            fp: a (n_atom x n_fingerprint) tensor of fingerprints
                where n_atom is the number of central atoms defined by "i"
            jacob_ind: a (n_pair x 2) tensor
                each row correspond to the (p_ind, i_rind) of the pair
                p_ind => the relative position of this pair within all pairs
                i_rind => the index of the central atom for this pair
        """
        p_filter = []  # build the filter if necessary
        i_rind = ind_2[:, 0]
        if self.i != "ALL":
            p_filter.append(tf.equal(tf.gather(elems, ind_2[:, 0]), self.i))
            a_rind = tf.cumsum(tf.cast(tf.equal(elems, self.i), tf.int32)) - 1
        else:  # a_rind: relative indices of atoms within all "i-species"
            a_rind = tf.cumsum(tf.ones_like(elems, tf.int32)) - 1
        if self.j != "ALL":
            p_filter.append(tf.equal(tf.gather(elems, ind_2[:, 1]), self.j))
        if p_filter:
            p_filter = tf.reduce_all(p_filter, axis=0)
            p_ind = tf.cast(tf.where(p_filter)[:, 0], tf.int32)
            dist = tf.gather(dist, p_ind)
            fc = tf.gather(fc, p_ind)
            i_rind = tf.gather(a_rind, tf.gather(i_rind, p_ind))
        else:
            p_ind = tf.cumsum(tf.ones_like(i_rind))-1

        sf = self.basis(dist, fc)
        fp = sf2fp(i_rind, a_rind, sf)
        jacob_ind = tf.stack([p_ind, i_rind], axis=1)
        return fp, jacob_ind


class G3_SF(tf.keras.layers.Layer):
    """BP-style G3 symmetry functions."""

    def __init__(self, lambd, zeta, eta, cutoff, rc, i="ALL", j="ALL", k="ALL"):
        """
        Args:
            lambd (list of floats): lambda parameter of G3 SF
            zeta (list of floats): zeta parameter of G3 SF
            eta (list of floats): Gaussian widths
            i (str): species i
            j (str): species j
            k (str): species k
        """
        super(G3_SF, self).__init__()
        self.lambd = tf.cast(lambd, tf.keras.backend.floatx())
        self.zeta = tf.cast(zeta, tf.keras.backend.floatx())
        self.basis = GaussianBasis(center=np.zeros_like(eta), gamma=eta)
        self.cutoff = cutoff
        self.rc = rc
        self.i = i
        self.j = j
        self.k = k

    def call(self, ind_2, ind_3, dist, diff, elems, fc):
        """

        Args:
            ind_2: (N_pair x 2) indices for each pair
            ind_3: (N_triplet x 2) indices for each triplet
            dist: (N_pair) array of distance
            diff: (N_pair) array of bond vectors
            elems: (N_atom) elements for each atom
            fc: (N_pair) cutoff functio  n

        Returns:
            fp: a (n_atom x n_fingerprint) tensor of fingerprints
                where n_atom is the number of central atoms defined by "i"
            jacob_ind: a (n_pair x 2) tensor
                each row correspond to the (p_ind, i_rind) of the pair
                p_ind => the relative position of this pair within all pairs
                i_rind => the index of the central atom for this pair
        """

        i_rind, a_rind, ind_ij, ind_ik = _triplet_filter(
            ind_2, ind_3, elems, self.i, self.j, self.k
        )

        # NOTE(YS): here diff_jk is calculated through diff_ik - diff_ij instead
        # of retrieving the distance calcualted in cell_list_nl. This makes is
        # easier to get the jacobian with grad(sf, [diff_ij, diff_ik]). However,
        # this probably introduce some waste of computation.
        diff_ij = tf.gather(diff, ind_ij)
        diff_ik = tf.gather(diff, ind_ik)
        diff_jk = diff_ik - diff_ij
        dist_jk = tf.norm(diff_jk, axis=1)
        t_ind = tf.where(dist_jk < self.rc)[:, 0]
        dist_jk = tf.gather(dist_jk, t_ind)
        fc_jk = self.cutoff(dist_jk)
        # other distances/vectors are gathered from the neighbor list
        ind_ij = tf.gather(ind_ij, t_ind)
        ind_ik = tf.gather(ind_ik, t_ind)
        i_rind = tf.gather(i_rind, t_ind)
        diff_ij = tf.gather(diff, ind_ij)
        diff_ik = tf.gather(diff, ind_ik)
        dist_ij = tf.gather(dist, ind_ij)
        dist_ik = tf.gather(dist, ind_ik)
        fc_ij = tf.gather(fc, ind_ij)
        fc_ik = tf.gather(fc, ind_ik)

        cos_ijk = tf.einsum("id,id->i", diff_ij, diff_ik) / dist_ij / dist_ik
        sf = (
            2 ** (1 - self.zeta[None,:])
            * (1 + self.lambd[None,:] * cos_ijk[:,None]) ** self.zeta[None,:]
            * self.basis(dist_ij, fc_ij)
            * self.basis(dist_ik, fc_ik)
            * self.basis(dist_jk, fc_jk)
        )

        fp = sf2fp(i_rind, a_rind, sf)
        jacob_ind = _triplet_jacobian(i_rind, ind_ij, ind_ik)
        return fp, jacob_ind


class G4_SF(tf.keras.layers.Layer):
    """BP-style G4 symmetry functions."""

    def __init__(self, lambd, zeta, eta, i="ALL", j="ALL", k="ALL"):
        """
        Args:
            lambd (list of floats): lambda parameter of G4 SF
            zeta (list of floats): zeta parameter of G4 SF
            eta (list of floats): Gaussian widths
            i (str): species i
            j (str): species j
            k (str): species k
        """
        super(G4_SF, self).__init__()
        self.lambd = tf.cast(lambd, tf.keras.backend.floatx())
        self.zeta = tf.cast(zeta, tf.keras.backend.floatx())
        self.basis = GaussianBasis(center=np.zeros_like(eta), gamma=eta)
        self.i = i
        self.j = j
        self.k = k

    def call(self, ind_2, ind_3, dist, diff, elems, fc):
        """

        Args:
            ind_2: (N_pair x 2) indices for each pair
            ind_3: (N_triplet x 2) indices for each triplet
            dist: (N_pair) array of distance
            diff: (N_pair) array of bond vectors
            elems: (N_atom) elements for each atom
            fc: (N_pair) cutoff functio  n

        Returns:
            fp: a (n_atom x n_fingerprint) tensor of fingerprints
                where n_atom is the number of central atoms defined by "i"
            jacob_ind: a (n_pair x 2) tensor
                each row correspond to the (p_ind, i_rind) of the pair
                p_ind => the relative position of this pair within all pairs
                i_rind => the index of the central atom for this pair
        """

        i_rind, a_rind, ind_ij, ind_ik = _triplet_filter(
            ind_2, ind_3, elems, self.i, self.j, self.k
        )

        # gather distances/vectors from the neighbor list
        diff_ij = tf.gather(diff, ind_ij)
        diff_ik = tf.gather(diff, ind_ik)
        dist_ij = tf.gather(dist, ind_ij)
        dist_ik = tf.gather(dist, ind_ik)
        fc_ij = tf.gather(fc, ind_ij)
        fc_ik = tf.gather(fc, ind_ik)

        cos_ijk = tf.einsum("id,id->i", diff_ij, diff_ik) / dist_ij / dist_ik
        sf = (
            2 ** (1 - self.zeta[None,:])
            * (1 + self.lambd[None,:] * cos_ijk[:,None]) ** self.zeta[None,:]
            * self.basis(dist_ij, fc_ij)
            * self.basis(dist_ik, fc_ik)
        )

        fp = sf2fp(i_rind, a_rind, sf)
        jacob_ind = _triplet_jacobian(i_rind, ind_ij, ind_ik)
        return fp, jacob_ind
