import subprocess
import string
import numpy as np

base = '/Users/dimitris/Repositories/matrix/group2'

for s in [1]:
    for r in [1,2,3,4,5,7,10]:
        for g in [2,3,4,5]:
            data_file = '{}/data/simdata{}.txt'.format(base,s)
            output_file = '{}/{}rep/{}groups/_clust{}'.format(base,r,g,s)
            samples = ' '.join(['sample' + str(item) for item in range(1,r+1)*g + np.repeat(range(0,g), r)*10])
            groups = ' ' .join([letter for letter in string.uppercase[:g] for _ in range(r)])
                        
            cmd = 'clust.py {} -s {} -g {} -o {} -e'.format(data_file, samples, groups, output_file)
            print cmd
            
            p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
            
        p.wait()
        