__version__ = '0.1'


def _svg_parse_(path):
    import re
    import numpy as np

    from matplotlib.path import Path

    commands = {'M': Path.MOVETO,
                'L': Path.LINETO,
                'Q': Path.CURVE3,
                'C': Path.CURVE4,
                'Z': Path.CLOSEPOLY}
    vertices = []
    codes = []
    cmd_values = re.split("([A-Za-z])", path)[1:]  # Split over commands.
    for cmd, values in zip(cmd_values[::2], cmd_values[1::2]):
        # Numbers are separated either by commas, or by +/- signs (but not at
        # the beginning of the string).
        if cmd.upper() in ['M', 'L', 'Q', 'C']:
            points = [e.split(',') for e in values.split(' ') if e != '']
            points = [list(map(float, each)) for each in points]
        else:
            points = [(0., 0.)]
        points = np.reshape(points, (-1, 2))
        # if cmd.islower():
        #    points += vertices[-1][-1]
        for i in range(0, len(points)):
            codes.append(commands[cmd.upper()])
        vertices.append(points)
    return np.array(codes), np.concatenate(vertices)


def _add_colorbar_(ax, cmap, norm, ec, labelsize, ylabel, cbsize=1, cbpad=1,
                   cbar=True, location='right'):
    if not cbar: return None
    assert location in ['right', 'bottom']
    import matplotlib
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # get parameters
    if location == 'right':
        ax_loc, cb_orient = 'right', 'vertical'
    elif location == 'bottom':
        ax_loc, cb_orient = 'bottom', 'horizontal'

    divider = make_axes_locatable(ax)
    cax = divider.append_axes(ax_loc, size=f'{cbsize}%', pad=cbpad)

    cb1 = matplotlib.colorbar.ColorbarBase(
        cax, cmap=cmap, norm=norm, orientation=cb_orient, ticklocation=location
    )
    cb1.ax.tick_params()
    if location == 'right':
        cb1.ax.set_ylabel(ylabel)
    elif location == 'bottom':
        cb1.ax.set_xlabel(ylabel)


def _render_data_(data, wd, cmap, norm, ax, edgecolor):
    import os.path as op
    import matplotlib.patches as patches
    from matplotlib.path import Path
    for k, v in data.items():
        fp = op.join(wd, k)
        if op.isfile(fp):
            p = open(fp).read()
            codes, verts = _svg_parse_(p)
            path = Path(verts, codes)
            c = cmap(norm(v))
            ax.add_patch(patches.PathPatch(path, facecolor=c,
                                           edgecolor=edgecolor, lw=.5))
        else:
            # print('%s not found' % fp)
            pass


def _create_figure_(files, figsize, background, title, fontsize, edgecolor,
                    ax=None):
    import numpy as np
    import matplotlib.pyplot as plt

    codes, verts = _svg_parse_(' '.join(files))

    xmin, ymin = verts.min(axis=0) - 1
    xmax, ymax = verts.max(axis=0) + 1
    yoff = 0
    ymin += yoff
    verts = np.array([(x, y + yoff) for x, y in verts])

    # create axis (if needed)
    if ax is None:
        fig = plt.figure(figsize=figsize, facecolor=background)
        ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect=1,
                          xlim=(xmin, xmax),  # centering
                          ylim=(ymax, ymin),  # centering, upside down
                          xticks=[], yticks=[])  # no ticks
    else:
        plt.xlim(xmin, xmax)
        plt.ylim(ymax, ymin)
        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect(1)

    # set title (if needed)
    if title:
        ax.set_title(title, y=1.03, x=0.55, fontweight='bold')

    return ax


def _render_regions_(files, ax, facecolor, edgecolor):
    from matplotlib.path import Path
    import matplotlib.patches as patches

    codes, verts = _svg_parse_(' '.join(files))
    path = Path(verts, codes)

    ax.add_patch(patches.PathPatch(path, facecolor=facecolor,
                                   edgecolor=edgecolor, lw=.1))


def _get_cmap_(cmap, values, vminmax=[]):
    import matplotlib

    if not len(values):
        values = [0, 1]

    cmap = matplotlib.cm.get_cmap(cmap)
    if vminmax == []:
        vmin, vmax = min(values), max(values)
    else:
        vmin, vmax = vminmax
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    return cmap, norm


def ggplot_dk(data, cmap='Spectral', background='k', edgecolor='w', ylabel='',
              figsize=(15, 15), bordercolor='w', vminmax=[], title='',
              fontsize=15):
    """Plot cortical ROI data based on a Desikan-Killiany (DK) parcellation.

    Parameters
    ----------
    data : dict
            Data to be plotted. Should be passed as a dictionary where each key
            refers to a region from the cortical Desikan-Killiany atlas. The
            full list of applicable regions can be found in the folder
            ggseg/data/dk.
    cmap : matplotlib colormap, optional
            The colormap for specified image.
            Default='Spectral'.
    vminmax : list, optional
            Lower and upper bound for the colormap, passed to matplotlib.colors.Normalize
    background : matplotlib color, if not provided, defaults to black
    edgecolor : matplotlib color, if not provided, defaults to white
    bordercolor : matplotlib color, if not provided, defaults to white
    ylabel : str, optional
            Label to display next to the colorbar
    figsize : list, optional
            Dimensions of the final figure, passed to matplotlib.pyplot.figure
    title : str, optional
            Title displayed above the figure, passed to matplotlib.axes.Axes.set_title
    fontsize: int, optional
            Relative font size for all elements (ticks, labels, title)
    """
    import matplotlib.pyplot as plt
    import os.path as op
    from glob import glob
    import ggseg

    wd = op.join(op.dirname(ggseg.__file__), 'data', 'dk')

    # A figure is created by the joint dimensions of the whole-brain outlines
    whole_reg = ['lateral_left', 'medial_left', 'lateral_right',
                 'medial_right']
    files = [open(op.join(wd, e)).read() for e in whole_reg]
    ax = _create_figure_(files, figsize, background, title, fontsize, edgecolor)

    # Each region is outlined
    reg = glob(op.join(wd, '*'))
    files = [open(e).read() for e in reg]
    _render_regions_(files, ax, bordercolor, edgecolor)

    # For every region with a provided value, we draw a patch with the color
    # matching the normalized scale
    cmap, norm = _get_cmap_(cmap, data.values(), vminmax=vminmax)
    _render_data_(data, wd, cmap, norm, ax, edgecolor)

    # DKT regions with no provided values are rendered in gray
    data_regions = list(data.keys())
    dkt_regions = [op.splitext(op.basename(e))[0] for e in reg]
    NA = set(dkt_regions).difference(data_regions).difference(whole_reg)
    files = [open(op.join(wd, e)).read() for e in NA]
    _render_regions_(files, ax, 'gray', edgecolor)

    # A colorbar is added
    _add_colorbar_(ax, cmap, norm, edgecolor, fontsize*0.75, ylabel)

    plt.show()


def ggplot_jhu(data, cmap='Spectral', background='k', edgecolor='w', ylabel='',
               figsize=(17, 5), bordercolor='w', vminmax=[], title='',
               fontsize=15):
    """Plot WM ROI data based on the Johns Hopkins University (JHU) white
    matter atlas.

    Parameters
    ----------
    data : dict
            Data to be plotted. Should be passed as a dictionary where each key
            refers to a region from the Johns Hopkins University white Matter
            atlas. The full list of applicable regions can be found in the
            folder ggseg/data/jhu.
    cmap : matplotlib colormap, optional
            The colormap for specified image.
            Default='Spectral'.
    vminmax : list, optional
            Lower and upper bound for the colormap, passed to matplotlib.colors.Normalize
    background : matplotlib color, if not provided, defaults to black
    edgecolor : matplotlib color, if not provided, defaults to white
    bordercolor : matplotlib color, if not provided, defaults to white
    ylabel : str, optional
            Label to display next to the colorbar
    figsize : list, optional
            Dimensions of the final figure, passed to matplotlib.pyplot.figure
    title : str, optional
            Title displayed above the figure, passed to matplotlib.axes.Axes.set_title
    fontsize: int, optional
            Relative font size for all elements (ticks, labels, title)
    """

    import matplotlib.pyplot as plt
    import ggseg
    import os.path as op
    from glob import glob

    wd = op.join(op.dirname(ggseg.__file__), 'data', 'jhu')

    # A figure is created by the joint dimensions of the whole-brain outlines
    whole_reg = ['NA']
    files = [open(op.join(wd, e)).read() for e in whole_reg]
    ax = _create_figure_(files, figsize, background, title, fontsize, edgecolor)

    # Each region is outlined
    reg = glob(op.join(wd, '*'))
    files = [open(e).read() for e in reg]
    _render_regions_(files, ax, bordercolor, edgecolor)

    # For every region with a provided value, we draw a patch with the color
    # matching the normalized scale
    cmap, norm = _get_cmap_(cmap, data.values(), vminmax=vminmax)
    _render_data_(data, wd, cmap, norm, ax, edgecolor)

    # JHU regions with no provided values are rendered in gray
    NA = ['CSF']
    files = [open(op.join(wd, e)).read() for e in NA]
    _render_regions_(files, ax, 'gray', edgecolor)

    # A colorbar is added
    _add_colorbar_(ax, cmap, norm, edgecolor, fontsize*0.75, ylabel)

    plt.show()


def ggplot_aseg(data, cmap='Spectral', background='k', edgecolor='w', ylabel='',
                figsize=(15, 5), bordercolor='w', vminmax=[],
                title='', fontsize=15):
    """Plot subcortical ROI data based on the FreeSurfer `aseg` atlas

    Parameters
    ----------
    data : dict
            Data to be plotted. Should be passed as a dictionary where each key
            refers to a region from the FreeSurfer `aseg` atlas. The full list
            of applicable regions can be found in the folder ggseg/data/aseg.
    cmap : matplotlib colormap, optional
            The colormap for specified image.
            Default='Spectral'.
    vminmax : list, optional
            Lower and upper bound for the colormap, passed to matplotlib.colors.Normalize
    background : matplotlib color, if not provided, defaults to black
    edgecolor : matplotlib color, if not provided, defaults to white
    bordercolor : matplotlib color, if not provided, defaults to white
    ylabel : str, optional
            Label to display next to the colorbar
    figsize : list, optional
            Dimensions of the final figure, passed to matplotlib.pyplot.figure
    title : str, optional
            Title displayed above the figure, passed to matplotlib.axes.Axes.set_title
    fontsize: int, optional
            Relative font size for all elements (ticks, labels, title)
    """
    import matplotlib.pyplot as plt
    import os.path as op
    from glob import glob
    import ggseg

    wd = op.join(op.dirname(ggseg.__file__), 'data', 'aseg')
    reg = [op.basename(e) for e in glob(op.join(wd, '*'))]

    # Select data from known regions (prevents colorbar from going wild)
    known_values = []
    for k, v in data.items():
        if k in reg:
            known_values.append(v)

    whole_reg = ['Coronal', 'Sagittal']
    files = [open(op.join(wd, e)).read() for e in whole_reg]

    # A figure is created by the joint dimensions of the whole-brain outlines
    ax = _create_figure_(files, figsize, background,  title, fontsize, edgecolor)

    # Each region is outlined
    reg = glob(op.join(wd, '*'))
    files = [open(e).read() for e in reg]
    _render_regions_(files, ax, bordercolor, edgecolor)

    # For every region with a provided value, we draw a patch with the color
    # matching the normalized scale
    cmap, norm = _get_cmap_(cmap, known_values, vminmax=vminmax)
    _render_data_(data, wd, cmap, norm, ax, edgecolor)

    # The following regions are ignored/displayed in gray
    NA = ['Cerebellum-Cortex', 'Cerebellum-White-Matter', 'Brain-Stem']
    files = [open(op.join(wd, e)).read() for e in NA]
    _render_regions_(files, ax, '#111111', edgecolor)

    # A colorbar is added
    _add_colorbar_(ax, cmap, norm, edgecolor, fontsize*0.75, ylabel)

    plt.show()


def ggplot_vep(
        data, cmap='Spectral', background='w', edgecolor='k', ylabel='',
        figsize=(10, 8), bordercolor='grey', vminmax=[], title='', fontsize=15,
        ax=None, cbar=True, cbar_loc='right', views=['internal', 'external'],
        hemispheres=['left', 'right']
    ):
    """Plot using the VEP atlas.

    Parameters
    ----------
    data : dict
        Keys are brain region names and values are attached to each parcel.
        To dissociate between left and right hemispheres, append '_left', '_l'
        or '_lh' (same for right)
    views : list
        List of views. Use either 'internal', 'external', 'frontal'
    hemispheres : list
        Use either 'left' and/or 'right'
    """
    import matplotlib.pyplot as plt
    import os.path as op
    from glob import glob
    import ggseg
    import warnings

    # -------------------------------------------------------------------------
    # define patterns to recognize left / right hemispheres
    l_pat = ['left', '_l', '_lh']
    r_pat = ['right', '_r', '_rh']

    # -------------------------------------------------------------------------
    # get working directory
    wd = op.join(op.dirname(ggseg.__file__), 'data', 'vep')

    # build orientations
    orientations = []
    for v in views:
        for h in hemispheres:
            orientations.append(f"{v}_{h}")

    # build cortex files
    cortex_file = [open(op.join(wd, o, "cortex")).read() for o in orientations]

    # build all files
    all_files = []
    for o in orientations:
        all_files += [e for e in glob(op.join(wd, o, '*'))]
    all_files = [open(e).read() for e in all_files if 'cortex' not in e]

    # build data files
    data_n, ignored = {}, []
    for k, v in data.items():
        # split in region / hemisphere names
        reg = k.split('_')[0]

        # determine which hemisphere to use
        if any([h in k for h in l_pat]):
            hemi = 'left'
        elif any([h in k for h in r_pat]):
            hemi = 'right'
        else:
            ignored.append(k)
            continue

        # skip if hemisphere not plotted
        if hemi not in hemispheres:
            ignored.append(k)
            continue

        # else, append views
        for vi in views:
            data_n[op.join(wd, f"{vi}_{hemi}", reg)] = v

    # print ignored regions
    if len(ignored):
        warnings.warn(
            f"Some regions have been ignored either because it's not\npossible "
            f"to infer the hemisphere either because the hemisphere\nis not "
            f"plotted :\n{', '.join(ignored)}"
        )

    # -------------------------------------------------------------------------
    # A figure is created by the joint dimensions of the whole-brain outlines
    ax = _create_figure_(
        cortex_file, figsize, background,  title, fontsize, edgecolor, ax=ax
    )

    # gray background cortex
    _render_regions_(cortex_file, ax, bordercolor, edgecolor)

    # Each region is outlined
    _render_regions_(all_files, ax, "lightgray", edgecolor)

    # For every region with a provided value, we draw a patch with the color
    # matching the normalized scale
    cmap, norm = _get_cmap_(cmap, data_n.values(), vminmax=vminmax)
    _render_data_(data_n, '', cmap, norm, ax, edgecolor)

    # A colorbar is added
    _add_colorbar_(
        ax, cmap, norm, edgecolor, fontsize * 0.75, ylabel, cbsize=4, cbpad=.5,
        cbar=cbar, location=cbar_loc
    )

    return plt.gcf()


def ggplot_marsatlas(
        data, lr=True, cmap='Spectral_r', background='w', edgecolor='k',
        ylabel='', figsize=(10, 8), bordercolor='grey', vminmax=[], title='',
        fontsize=15, ax=None, cbar=True, cbar_loc='right'
    ):
    import matplotlib.pyplot as plt
    import os.path as op
    from glob import glob
    import ggseg

    # possible orientations
    if lr:
        orientations = [
            'external_left', 'internal_left', 'external_right',
            'internal_right'
        ]
    else:
        orientations = ['external_left', 'internal_left']

    # get names of all brain regions
    wd = op.join(op.dirname(ggseg.__file__), 'data', 'marsatlas')
    reg = [op.basename(e) for e in glob(op.join(wd, '*'))]

    # rebuild data for matching with orientations
    data_n = {}
    for k, v in data.items():
        if lr:
            orient = 'left' if 'L_' in k else 'right'
            for ori in [f"external_{orient}", f"internal_{orient}"]:
                new_key = f"{k.split('_')[-1]}_{ori}"
                if new_key in reg:
                    data_n[new_key] = v
        else:
            for ori in orientations:
                new_key = f"{k}_{ori}"
                if new_key in reg:
                    data_n[new_key] = v

        # for ori in orientations:
        #     new_key = f"{k}_{ori}"
        #     if new_key in reg:
        #         data_n[new_key] = v

    # Select data from known regions (prevents colorbar from going wild)
    known_values = []
    for k, v in data_n.items():
        if k in reg:
            known_values.append(v)

    whole_reg = [f"cortex_{ori}" for ori in orientations]
    files = [open(op.join(wd, e)).read() for e in whole_reg]

    # A figure is created by the joint dimensions of the whole-brain outlines
    ax = _create_figure_(
        files, figsize, background,  title, fontsize, edgecolor, ax=ax
    )

    # gray background cortex
    _render_regions_(files, ax, bordercolor, edgecolor)

    # Each region is outlined
    reg = glob(op.join(wd, '*'))
    files = [open(e).read() for e in reg if 'cortex' not in e]
    _render_regions_(files, ax, "lightgray", edgecolor)

    # For every region with a provided value, we draw a patch with the color
    # matching the normalized scale
    cmap, norm = _get_cmap_(cmap, known_values, vminmax=vminmax)
    _render_data_(data_n, wd, cmap, norm, ax, edgecolor)

    # A colorbar is added
    _add_colorbar_(
        ax, cmap, norm, edgecolor, fontsize * 0.75, ylabel, cbsize=3, cbpad=1,
        cbar=cbar, location=cbar_loc
    )

    return plt.gcf()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os
    import numpy as np


    # data = {
    #     'Mdl': 11,
    #     'PMdl': 15,
    #     'PMdm': 9,
    #     'Mdm': 23,
    #     'PFcdm': 17,
    #     'PMrv': 5,
    # }
    # data = {
    #     'L_Mv': 10,
    #     "R_Insula": 21,
    #     "L_Sdm": 12,
    #     'L_PFCvm': 3,
    #     'R_VCl': 22
    # }
    # ggplot_marsatlas(
    #     data, lr=True, cmap='Spectral_r', bordercolor='gray', background='w',
    #     edgecolor='k', ylabel='Power (a.u)'
    # )

    # data = {
    #     'Mdl': 11,
    #     'PMdl': 15,
    #     'PMdm': 9,
    #     'Mdm': 23,
    #     'PFcdm': 17,
    #     'PMrv': 5,
    # }

    # ggplot_vep(
    #     data, cmap='Spectral_r', bordercolor='gray', background='w',
    #     edgecolor='k', ylabel='Power (a.u)'
    # )

    data = {
        '1_left': 11,
        '2_right': 33,
        '1_right': 44,
        '30_left': 25
    }

    # files = os.listdir('data/vep')
    # _data = np.random.rand(len(files))
    # data = {}
    # for n_f, f in enumerate(files):
    #     if 'cortex' in f: continue
    #     data[f] = _data[n_f]

    ggplot_vep(
        data, cmap='plasma', ylabel='Power (a.u)',
        views=['internal', 'external'], hemispheres=['left', 'right'],
        cbar_loc='bottom'
    )
    plt.show()