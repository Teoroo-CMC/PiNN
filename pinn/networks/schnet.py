def schnet_network(tensors):
    """ Network function for
    SchNet: https://doi.org/10.1063/1.5019779

    TODO: Implement this

    Args: 
        tensors: input data (nested tensor from dataset).
        gamma (float): "width" of the radial basis.
        miu_max (float): minimal distance of the radial basis.
        miu_min (float): maximal distance of the radial basis.
        n_basis (int): number of radial basis.
        n_atomic (int): number of nodes to be used in atomic layers.
        n_cfconv (int): number of nodes to be used in cfconv layers.
        T (int): number of interaction blocks.
        pre_level (int): flag for preprocessing:
            0 for no preprocessing;
            1 for preprocess till the cell list nl;
            2 for preprocess all filters (cannot do force training).
    Returns:
        - preprocessed nested tensors if n<0
        - prediction tensor if n>=0
    """
    pass
