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
  label_map  = {'e_data': 'U0'}
  
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