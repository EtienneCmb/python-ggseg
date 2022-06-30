import numpy as np
import ggseg

data = {'Left-Lateral-Ventricle': 12289.6,
        'Left-Thalamus': 8158.3,
        'Left-Caudate': 3463.3,
        'Left-Putamen': 4265.3,
        'Left-Pallidum': 1620.9,
        '3rd-Ventricle': 1635.6,
        '4th-Ventricle': 1115.6,
}

ggseg.plot_aseg(data, cmap='Spectral',
                background='k', edgecolor='w', bordercolor='gray',
                ylabel='Volume (mm3)', title='Title of the figure')
