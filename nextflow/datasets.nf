#!/usr/bin/env nextflow

params.md17_tags = 'uracil,naphthalene,aspirin,malonaldehyde,ethanol,toluene'
params.rmd17_tags = 'aspirin,azobenzene,benzene,ethanol,malonaldehyde,naphthalene,paracetamol,salicylic,toluene,uracil'

process gen_qm9 {
  publishDir "datasets/public"
  
  output:
  path("qm9.{yml,tfr}")

  script:
  """
  #!/usr/bin/env python
  
  import requests, tarfile, io
  import numpy as np
  from pinn.io.qm9 import _qm9_spec
  from pinn.io.base import list_loader
  from pinn.io import write_tfrecord
  from ase.data import atomic_numbers
  
  qm9_url = "https://ndownloader.figshare.com/files/3195389"
  qm9_bytes = requests.get(qm9_url, allow_redirects=True).content
  qm9_fobj = io.BytesIO(qm9_bytes)
  qm9_fobj.seek(0)
  qm9_tar = tarfile.open(fileobj=qm9_fobj, mode='r:bz2')
  names = qm9_tar.getnames()

  exclude_url = "https://figshare.com/ndownloader/files/3195404"
  exclude_bytes = requests.get(exclude_url, allow_redirects=True).content
  exclude_fobj = io.TextIOWrapper(io.BytesIO(exclude_bytes))
  exclude = [int(line.split()[0]) for line in exclude_fobj.readlines()[9:-1]]
  names = [name for name in names if int(name[-10:-4]) not in exclude]
  
  # custom loader -- convert without file IO
  _labels = ['tag', 'index', 'A', 'B', 'C', 'mu', 'alpha', 'homo',
             'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']
  _label_ind = {k: i for i, k in enumerate(_labels)}
  label_map  = {'e_data': 'U0', 'd_data': 'mu'}
  
  @list_loader(ds_spec=_qm9_spec(label_map))
  def load_qm9_tar(name):
      f = io.TextIOWrapper(qm9_tar.extractfile(name))
      lines = f.readlines()
      elems = [atomic_numbers[l.split()[0]] for l in lines[2:-3]]
      coord = [[i.replace('*^', 'E') for i in l.split()[1:4]]
               for l in lines[2:-3]]
      elems = np.array(elems, np.int32)
      coord = np.array(coord, float)
      data = {'elems': elems, 'coord': coord}
  
      for k, v in label_map.items():
          data[k] = float(lines[1].split()[_label_ind[v]])
      return data
  
  qm9_ds = load_qm9_tar(names)
  write_tfrecord('qm9.yml', qm9_ds)
  """
}

process gen_md17_single {
  publishDir "datasets/public"

  input:
  val(md17_tag)

  output:
  path("md17-*.{yml,tfr}")

  script:
  """
  #!/usr/bin/env python
  
  import numpy as np
  import requests, zipfile, io
  from pinn.io import load_numpy
  from pinn.io import write_tfrecord
  from ase.io import read
  
  md17_url = "http://www.quantum-machine.org/gdml/data/npz/md17"
  all_ds = None
  
  tag_url = f"{md17_url}_${md17_tag}.npz"
  tag_bytes = requests.get(tag_url, allow_redirects=True).content
  tag_fobj = io.BytesIO(tag_bytes)
  tag_npz = np.load(tag_fobj)
  tag_size = tag_npz['R'].shape[0]
  tag_ds = load_numpy({
      'elems': np.repeat(tag_npz['z'][None,:], tag_size, axis=0),
      'coord': tag_npz['R'],
      'e_data': tag_npz['E'][:,0],
      'f_data': tag_npz['F']
  })
  write_tfrecord(f'md17-${md17_tag}.yml', tag_ds)
  """
}

workflow gen_md17 {
  Channel
    .fromList(params.md17_tags.tokenize(','))
    |gen_md17_single

  emit:
  gen_md17_single.out
}
  
process gen_rmd17_single {
  publishDir "datasets/public"

  input:
  val(rmd17_tag)

  output:
  path("rmd17-*.{yml,tfr}")

  script:
  """
  #!/usr/bin/env python
  
  import numpy as np
  import requests, zipfile, io, tarfile
  from pinn.io import load_numpy
  from pinn.io import write_tfrecord
  import os
  
  rmd17_url = "https://figshare.com/ndownloader/articles/12672038/versions/3"
  zip_bytes = requests.get(rmd17_url, allow_redirects=True).content
  zip_bytes_io = io.BytesIO(zip_bytes)
  zip_bytes_io.seek(0)
  zip_fobj = zipfile.ZipFile(zip_bytes_io)
  tar_path = zip_fobj.extract(zip_fobj.infolist()[0])
  tar_fobj = tarfile.open(tar_path)
  npz_obj = tar_fobj.extractfile('rmd17/npz_data/rmd17_${rmd17_tag}.npz')
  tag_npz = np.load(npz_obj)
  tag_size = tag_npz['coords'].shape[0]
  tag_ds = load_numpy({
      'elems': np.repeat(tag_npz['nuclear_charges'][None,:], tag_size, axis=0),
      'coord': tag_npz['coords'],
      'e_data': tag_npz['energies'],
      'f_data': tag_npz['forces']
  })
  write_tfrecord(f'rmd17-${rmd17_tag}.yml', tag_ds)
  tar_fobj.close()
  zip_fobj.close()
  os.remove(tar_path)
  """
}

workflow gen_rmd17 {
  Channel
    .fromList(params.rmd17_tags.tokenize(','))
    |gen_rmd17_single

  emit:
  gen_rmd17_single.out
}

process gen_mp2018 {
  publishDir "datasets/public"
  
  output:
  path("mpc.{yml,tfr}")

  script:
"""
#!/usr/bin/env python

import requests, zipfile, io, json
import numpy as np
from pinn.io.base import list_loader
from pinn.io import write_tfrecord
from ase.data import atomic_numbers
from ase.cell import Cell

mpc_url = "https://figshare.com/ndownloader/files/15087992"
mpc_bytes = requests.get(mpc_url, allow_redirects=True).content
mpc_fobj = io.BytesIO(mpc_bytes)
mpc_fobj.seek(0)

mpc_zip = zipfile.ZipFile(mpc_fobj)
data_path = mpc_zip.extract('mp.2018.6.1.json')
with open(data_path) as f:
    data = json.load(f)

assert len(data) == 69239

@list_loader(pbc=True)
def load_mpc(datum):
    struct = datum['structure']
    energy = datum['formation_energy_per_atom']
    lines = struct.split('\\n')
    lines = list(map(lambda line: line.strip(), lines))
    for line in lines:
        if '_cell_length_a' in line:
            a = line.split()[1]
        if '_cell_length_b' in line:
            b = line.split()[1]
        if '_cell_length_c' in line:
            c = line.split()[1]
        if '_cell_angle_alpha' in line:
            alpha = line.split()[1]
        if '_cell_angle_beta' in line:
            beta = line.split()[1]
        if '_cell_angle_gamma' in line:
            gamma = line.split()[1]
        if '_cell_volume' in line:
            volume = line.split()[1]
    cell = Cell.fromcellpar(list(map(float, [a, b, c, alpha, beta, gamma])))
    lattice_matrix = cell.array
    lines = lines[lines.index('_atom_site_occupancy')+1:]
    lines = filter(lambda line: line, lines)
    atomtypes = []
    coords = []

    for line in lines:
        l = line.split()
        atomtypes.append(atomic_numbers[l[0]])
        coords.append(list(map(float, l[3:6])))

    return {
    'coord': np.array(coords),
    'elems': np.array(atomtypes),
    'cell': lattice_matrix,
    'e_data': np.array(energy) * len(coords)
  }

mpc_ds = load_mpc(data)
write_tfrecord('mpc.yml', mpc_ds)
"""
}

process gen_mp2021 {
  publishDir "datasets/public"
  
  output:
  path("mpf.{yml,tfr}")

  script:
"""
#!/usr/bin/env python

import requests, zipfile, io, json
import pickle
import numpy as np
from pinn.io.base import list_loader
from pinn.io import write_tfrecord
from ase.data import atomic_numbers
from ase.cell import Cell

mpf_url = "https://figshare.com/ndownloader/articles/19470599/versions/3"
mpf_bytes = requests.get(mpf_url, allow_redirects=True).content
mpf_fobj = io.BytesIO(mpf_bytes)
mpf_fobj.seek(0)

mpf_zip = zipfile.ZipFile(mpf_fobj)
part1 = mpf_zip.extract('block_0.p')
part2 = mpf_zip.extract('block_1.p')

with open(part1, 'rb') as f:
    data1 = pickle.load(f)
with open(part2, 'rb') as f:
    data2 = pickle.load(f)
data = {**data1, **data2}

flatten_data = []
for datum in data.values():
    for struct, energy, forces, stress in zip(datum['structure'], datum['energy'], datum['force'], datum['stress']):
      if np.any(np.linalg.norm(forces, axis=-1) > 100):
        continue
      flatten_data.append(
            {'structure': struct,
            'energy': energy,
            'forces': forces, 
            'stress': stress
            }
        )

@list_loader(pbc=True, force=True)
def load_mpf(datum):
    struct = datum['structure']
    energy = datum['energy']
    forces = datum['forces']
    stress = datum['stress']
    cell = struct.lattice.matrix
    atomtypes = []
    coords = []
    for site in struct.sites:
        atomtypes.append(atomic_numbers[site.specie.name])
        coords.append([site.x, site.y, site.z])

    return {
    'coord': np.array(coords),
    'elems': np.array(atomtypes),
    'cell': cell,
    'e_data': np.array(energy),
    'f_data': np.array(forces),
    # 's_data': np.array(stress)
  }

mpf_ds = load_mpf(flatten_data)
write_tfrecord('mpf.yml', mpf_ds)

"""
}
