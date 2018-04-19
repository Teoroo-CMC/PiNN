import numpy as np
class traj_parser():
    def __init__(self, traj, p_filter):
        self.size = len(traj)
        self.n_atoms = max([len(atoms) for atoms in traj])
        c_mat = []
        p_mat = []
        for atoms in traj:
            n_pad = self.n_atoms - len(atoms)
            c_mat.append(np.pad(atoms.get_positions(),
                                     [[0, n_pad], [0, 0]], 'constant'))
            p_mat.append(np.pad(p_filter.parse(atoms),
                                     [[0, n_pad], [0, 0]], 'constant'))
        self.c_mat = np.array(c_mat)
        self.p_mat = np.array(p_mat)
        self.e_mat = np.array([atoms.get_potential_energy() for atoms in traj])

    def get_input(self, index):
        c_in = self.c_mat[index]
        p_in = self.p_mat[index]
        e_in = self.e_mat[index]
        feed_dict = {'c_in': c_in,
                     'p_in': p_in,
                     'e_in': e_in}
        return feed_dict


class npz_parser():
    def __init__(self, fname):
        data = np.load(fname)
        self.c_mat = data['c_mat']
        self.p_mat = data['p_mat']
        self.e_mat = data['e_mat']
        self.size = self.c_mat.shape[0]
        self.n_atoms = self.c_mat.shape[1]

    def get_input(self, index):
        feed_dict = {'c_in': self.c_mat[index],
                     'p_in': self.p_mat[index],
                     'e_in': self.e_mat[index]}
        return feed_dict
